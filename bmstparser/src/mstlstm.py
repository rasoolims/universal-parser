from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
import codecs


class MSTParserLSTM:
    def __init__(self, vocab, pos, rels, w2i, options):
        self.model = Model()
        random.seed(1)
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'tanh3': (lambda x: tanh(cwise_multiply(cwise_multiply(x, x), x)))}
        self.activation = self.activations[options.activation]

        self.costaugFlag = options.costaugFlag
        self.dropout = False if options.dropout==0.0 else True
        self.dropout_prob = options.dropout
        self.batch = options.batch
        self.ldims = options.lstm_dims
        self.wdims = options.we
        self.pdims = options.pe
        self.rdims = options.re
        self.layers = options.layer
        self.wordsCount = vocab
        self.vocab = {word: ind+3 for word, ind in w2i.iteritems()}
        self.pos = {word: ind+3 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels

        self.external_embedding, self.edim = None, 0
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding,'r')
            external_embedding_fp.readline()
            self.external_embedding = {line.split(' ')[0] : [float(f) for f in line.strip().split(' ')[1:]] for line in external_embedding_fp}
            external_embedding_fp.close()

            self.edim = len(self.external_embedding.values()[0])
            self.noextrn = [0.0 for _ in xrange(self.edim)]
            self.extrnd = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self.elookup = self.model.add_lookup_parameters((len(self.external_embedding) + 3, self.edim))
            for word, i in self.extrnd.iteritems():
                self.elookup.init_row(i, self.external_embedding[word])
            self.extrnd['*PAD*'] = 1
            self.extrnd['*INITIAL*'] = 2

            print 'Load external embedding. Vector dimensions', self.edim

        self.deep_lstms = BiRNNBuilder(self.layers, self.wdims + self.pdims + self.edim, self.ldims*2, self.model, VanillaLSTMBuilder)
        self.hidden_units = options.hidden_units

        self.vocab['*PAD*'] = 1
        self.pos['*PAD*'] = 1
        self.vocab['*INITIAL*'] = 2
        self.pos['*INITIAL*'] = 2

        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.plookup = self.model.add_lookup_parameters((len(pos) + 3, self.pdims))
        self.rlookup = self.model.add_lookup_parameters((len(rels), self.rdims))
        self.hidLayerFOH = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidLayerFOM = self.model.add_parameters((self.hidden_units, self.ldims * 2))
        self.hidBias = self.model.add_parameters((self.hidden_units))
        self.outLayer = self.model.add_parameters((1, self.hidden_units))
        self.rhidLayerFOH = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
        self.rhidLayerFOM = self.model.add_parameters((self.hidden_units, 2 * self.ldims))
        self.rhidBias = self.model.add_parameters((self.hidden_units))
        self.routLayer = self.model.add_parameters((len(self.irels), self.hidden_units))
        self.routBias = self.model.add_parameters((len(self.irels)))

    def  __getExpr(self, sentence, modifier):
        h = concatenate_cols([self.activation(sentence[i].headfov + sentence[modifier].modfov + self.hidBias.expr()) for i in range(len(sentence))])
        return transpose(self.outLayer.expr() * h)

    def __evaluate(self, sentence):
        return [self.__getExpr(sentence, i) for i in xrange(len(sentence))]

    def get_all_scores(self, sentence):
        scores = concatenate_cols(self.__evaluate(sentence))
        return reshape(scores, (len(sentence), len(sentence))).npvalue()

    def __evaluateLabel(self, sentence, i, j):
       return self.routLayer.expr() * self.activation(sentence[i].rheadfov + sentence[j].rmodfov + self.rhidBias.expr()) + self.routBias.expr()

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def getInputLayer(self, sentence, train):
        embed = []
        for entry in sentence:
            c = float(self.wordsCount.get(entry.norm, 0))
            dropFlag = (random.random() < (c / (0.25 + c))) if train else False
            wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0)) if dropFlag else 0] if self.wdims > 0 else None
            posvec = self.plookup[int(self.pos[entry.pos])] if self.pdims > 0 else None
            evec = None

            if self.external_embedding is not None:
                evec = self.elookup[self.extrnd.get(entry.form, self.extrnd.get(entry.norm, 0)) if (dropFlag or (random.random() < 0.5)) else 0]
            vec = concatenate(filter(None, [wordvec, posvec, evec]))
            if self.dropout:
                vec = dropout(vec, self.dropout_prob)
            embed.append(vec)
        return embed

    def Predict(self, conll_path):
        with codecs.open(conll_path, 'r', encoding='utf-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                lstm_vecs = self.deep_lstms.transduce(self.getInputLayer(conll_sentence, False))

                for i, entry in enumerate(conll_sentence):
                    entry.vec = lstm_vecs[i]
                    if self.dropout:
                        entry.vec = dropout(entry.vec, self.dropout_prob)
                    entry.headfov = self.hidLayerFOH.expr() * entry.vec
                    entry.modfov = self.hidLayerFOM.expr() * entry.vec
                    entry.rheadfov = self.rhidLayerFOH.expr() * entry.vec
                    entry.rmodfov = self.rhidLayerFOM.expr() * entry.vec

                scores = self.get_all_scores(conll_sentence)
                heads = decoder.parse_proj(scores.T)

                for entry, head in zip(conll_sentence, heads):
                    entry.pred_parent_id = head
                    entry.pred_relation = '_'

                dump = False
                for modifier, head in enumerate(heads[1:]):
                    scores = self.__evaluateLabel(conll_sentence, head, modifier+1).value()
                    conll_sentence[modifier+1].pred_relation = self.irels[max(enumerate(scores), key=itemgetter(1))[0]]

                renew_cg()
                if not dump:
                    yield sentence

    def Train(self, conll_path):
        start = time.time()

        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)
            loss_vec = []
            for iSentence, sentence in enumerate(shuffledData):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                lstm_vecs = self.deep_lstms.transduce(self.getInputLayer(conll_sentence, True))
                for i, entry in enumerate(conll_sentence):
                    entry.vec = lstm_vecs[i]
                    if self.dropout:
                        entry.vec = dropout(entry.vec, self.dropout_prob)
                    entry.headfov = self.hidLayerFOH.expr() * entry.vec
                    entry.modfov = self.hidLayerFOM.expr() * entry.vec
                    entry.rheadfov = self.rhidLayerFOH.expr() * entry.vec
                    entry.rmodfov = self.rhidLayerFOM.expr() * entry.vec
                scores = self.__evaluate(conll_sentence)
                for modifier, entry in enumerate(conll_sentence[1:]):
                    loss_vec.append(pickneglogsoftmax(scores[modifier+1], entry.parent_id))
                    loss_vec.append(pickneglogsoftmax(self.__evaluateLabel(conll_sentence, entry.parent_id, modifier+1), self.rels[entry.relation]))

                if len(loss_vec)>=self.batch:
                    err = esum(loss_vec)/len(loss_vec)
                    err.scalar_value()
                    print 'Processing sentence number:', iSentence+1, 'Loss:', err.value() , 'Time', time.time() - start
                    err.backward()
                    renew_cg()
                    loss,loss_vec,start = 0, [],time.time()
        if len(loss_vec) > 0:
            err = esum(loss_vec) / len(loss_vec)
            err.scalar_value()
            print 'Processing sentence number:', iSentence+1, 'Loss:', err.value(), 'Time', time.time() - start
            err.backward()
            self.trainer.update()
            renew_cg()
            loss, loss_vec, start = 0, [], time.time()
        self.trainer.update_epoch()
