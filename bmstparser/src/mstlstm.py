from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
import codecs


class MSTParserLSTM:
    def __init__(self, pos, rels, w2i, options):
        self.model = Model()
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'leaky': (lambda x: bmax(.1*x, x))}
        self.activation = self.activations[options.activation]
        self.dropout = False if options.dropout==0.0 else True
        self.vocab = {word: ind+1 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+1 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        edim = options.we
        self.wlookup = self.model.add_lookup_parameters((len(w2i) + 1, edim))

        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp if len(line.split(' '))>2}
            external_embedding_fp.close()
            self.evocab = {word: i + 1 for i, word in enumerate(external_embedding)}

            edim = len(external_embedding.values()[0])
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 1, edim))
            self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 1, edim))
            self.elookup.set_updated(False)
            self.elookup.init_row(0, [0]*edim)
            for word in external_embedding.keys():
                self.elookup.init_row(self.evocab[word], external_embedding[word])
                if word in self.vocab:
                    self.wlookup.init_row(self.vocab[word], external_embedding[word])

            print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(external_embedding),'words'

        self.plookup = self.model.add_lookup_parameters((len(pos) + 1, options.pe))
        self.rlookup = self.model.add_lookup_parameters((len(rels), options.re))
        self.hidLayerFOH = self.model.add_parameters((options.hidden_units, options.lstm_dims * 2))
        self.hidLayerFOM = self.model.add_parameters((options.hidden_units, options.lstm_dims * 2))
        self.hidBias = self.model.add_parameters((options.hidden_units))
        self.outLayer = self.model.add_parameters((1, options.hidden_units))
        self.rhidLayerFOH = self.model.add_parameters((options.hidden_units, 2 * options.lstm_dims))
        self.rhidLayerFOM = self.model.add_parameters((options.hidden_units, 2 * options.lstm_dims))
        self.rhidBias = self.model.add_parameters((options.hidden_units))
        self.routLayer = self.model.add_parameters((len(self.irels), options.hidden_units))
        self.routBias = self.model.add_parameters((len(self.irels)))
        self.deep_lstms = BiRNNBuilder(options.layer, edim + options.pe, options.lstm_dims * 2, self.model, VanillaLSTMBuilder)

    def  __getExpr(self, sentence, modifier):
        h = concatenate_cols([self.activation(sentence[i].headfov + sentence[modifier].modfov + self.hidBias.expr()) for i in range(len(sentence))])
        return (reshape(self.outLayer.expr() * h,(len(sentence),1)))

    def __evaluate(self, sentence):
        return [self.__getExpr(sentence, i) for i in xrange(len(sentence))]

    def __evaluateLabel(self, sentence, i, j):
       return self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def getInputLayer(self, sentence, train):
        embed = []
        for entry in sentence:
            wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.options.we > 0 else None
            ewordvec = self.elookup[int(self.evocab.get(entry.norm, 0))] if self.options.we > 0 else None
            posvec = self.plookup[int(self.pos[entry.pos])] if self.options.pe > 0 else None
            vec = concatenate(filter(None, [wordvec+ewordvec, posvec]))
            if self.dropout:
                vec = dropout(vec, self.options.dropout)
            embed.append(vec)
        return embed

    def Predict(self, conll_path, greedy):
        with codecs.open(conll_path, 'r', encoding='utf-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                lstm_vecs = self.deep_lstms.transduce(self.getInputLayer(conll_sentence, False))

                for i, entry in enumerate(conll_sentence):
                    entry.vec = lstm_vecs[i]
                    entry.headfov = self.hidLayerFOH.expr() * entry.vec
                    entry.modfov = self.hidLayerFOM.expr() * entry.vec
                    entry.rheadfov = self.rhidLayerFOH.expr() * entry.vec
                    entry.rmodfov = self.rhidLayerFOM.expr() * entry.vec

                if greedy:
                    scores = self.__evaluate(conll_sentence)
                    for modifier, entry in enumerate(conll_sentence[1:]):
                        entry.pred_parent_id = np.argmax(scores[modifier+1].value())
                        s = self.__evaluateLabel(conll_sentence, entry.pred_parent_id, modifier + 1).value()
                        conll_sentence[modifier + 1].pred_relation = self.irels[max(enumerate(s), key=itemgetter(1))[0]]
                else:
                    scores = self.__evaluate(conll_sentence)
                    scores = np.array([s.npvalue().T[0] for s in scores]).T
                    heads = decoder.parse_proj(scores)

                    for entry, head in zip(conll_sentence, heads):
                        entry.pred_parent_id = head
                        entry.pred_relation = '_'

                    for modifier, head in enumerate(heads[1:]):
                        scores = self.__evaluateLabel(conll_sentence, head, modifier+1).value()
                        conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]
                renew_cg()
                yield sentence

    def Train(self, conll_path, t):
        self.deep_lstms.set_dropout(self.options.dropout)
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)
            start, lss, total,loss_vec = time.time(), 0, 0, []
            for iSentence, sentence in enumerate(shuffledData):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                lstm_vecs = self.deep_lstms.transduce(self.getInputLayer(conll_sentence, True))
                for i, entry in enumerate(conll_sentence):
                    entry.vec = lstm_vecs[i]
                    entry.headfov = self.hidLayerFOH.expr() * entry.vec
                    entry.modfov = self.hidLayerFOM.expr() * entry.vec
                    entry.rheadfov = self.rhidLayerFOH.expr() * entry.vec
                    entry.rmodfov = self.rhidLayerFOM.expr() * entry.vec
                    if self.dropout:
                        entry.headfov = dropout(entry.headfov, self.options.dropout)
                        entry.modfov = dropout(entry.modfov, self.options.dropout)
                        entry.rheadfov = dropout(entry.rheadfov, self.options.dropout)
                        entry.rmodfov = dropout(entry.rmodfov, self.options.dropout)
                scores = self.__evaluate(conll_sentence)
                for modifier, entry in enumerate(conll_sentence[1:]):
                    loss_vec.append(pickneglogsoftmax(scores[modifier+1], entry.parent_id))
                    loss_vec.append(pickneglogsoftmax(self.__evaluateLabel(conll_sentence, entry.parent_id, modifier+1)))

                if len(loss_vec)>=2*self.options.batch:
                    err = esum(loss_vec)/len(loss_vec)
                    err.scalar_value()
                    lss+= err.value()
                    total+=1
                    if total%10==0:
                        print 'Processing sentence number:', iSentence+1, 'Loss:', float(lss)/total , 'Time', time.time() - start
                        lss,total,start=0,0,time.time()
                    err.backward()
                    self.trainer.update()
                    renew_cg()
                    loss_vec,t = [],t+1
                    if self.options.anneal:
                        decay_steps = min(1.0, float(t)/50000)
                        lr = self.options.lr * 0.75 ** decay_steps
                        self.trainer.learning_rate = lr

        renew_cg()
        print 'current learning rate', self.trainer.learning_rate,'t:',t
        self.deep_lstms.disable_dropout()
        return t
