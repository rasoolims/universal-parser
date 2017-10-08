from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
import codecs
from linalg import *

class MSTParserLSTM:
    def __init__(self, pos, rels, w2i, options):
        self.model = Model()
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'leaky': (lambda x: bmax(.1 * x, x))}
        self.activation = self.activations[options.activation]
        self.dropout = False if options.dropout == 0.0 else True
        self.vocab = {word: ind + 1 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 1 for ind, word in enumerate(pos)}
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        edim = options.we
        self.wlookup = self.model.add_lookup_parameters((len(w2i) + 1, edim))
        self.elookup = None
        if options.external_embedding is not None:
            external_embedding_fp = open(options.external_embedding, 'r')
            external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                  external_embedding_fp if len(line.split(' ')) > 2}
            external_embedding_fp.close()
            self.evocab = {word: i + 1 for i, word in enumerate(external_embedding)}

            edim = len(external_embedding.values()[0])
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 1, edim), init=ConstInitializer(0))
            self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 1, edim))
            self.elookup.set_updated(False)
            self.elookup.init_row(0, [0] * edim)
            for word in external_embedding.keys():
                self.elookup.init_row(self.evocab[word], external_embedding[word])
                if word in self.vocab:
                    self.wlookup.init_row(self.vocab[word], external_embedding[word])

            print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(
                external_embedding), 'words'

        self.plookup = self.model.add_lookup_parameters((len(pos) + 1, options.pe))
        self.rlookup = self.model.add_lookup_parameters((len(rels), options.re))
        self.deep_lstms = BiRNNBuilder(options.layer, edim + options.pe, options.rnn * 2, self.model, VanillaLSTMBuilder)
        for i in range(len(self.deep_lstms.builder_layers)):
            builder = self.deep_lstms.builder_layers[i]
            b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
            b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
            self.deep_lstms.builder_layers[i] = (b0, b1)
        w_mlp_arc = orthonormal_initializer(options.arc_mlp, options.rnn * 2)
        w_mlp_label = orthonormal_initializer(options.label_mlp, options.rnn * 2)
        self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.rnn * 2), init= NumpyInitializer(w_mlp_arc))
        self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init = ConstInitializer(0))
        self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.rnn * 2), init= NumpyInitializer(w_mlp_label))
        self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init = ConstInitializer(0))
        self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.rnn * 2), init= NumpyInitializer(w_mlp_arc))
        self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init = ConstInitializer(0))
        self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.rnn * 2), init= NumpyInitializer(w_mlp_label))
        self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init = ConstInitializer(0))
        self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp+1), init = ConstInitializer(0))
        self.u_label = self.model.add_parameters((options.label_mlp+1, len(self.irels) * (options.label_mlp+1)), init = ConstInitializer(0))

    def __evaluate(self, H, M):
        M2 = concatenate([M, inputTensor(np.ones((1, M.dim()[0][1]), dtype=np.float32))])
        return transpose(H)*(self.w_arc.expr()*M2)

    def __evaluateLabel(self, i, j, HL, ML):
        H2 = concatenate([HL, inputTensor(np.ones((1, HL.dim()[0][1]), dtype=np.float32))])
        M2 = concatenate([ML, inputTensor(np.ones((1, ML.dim()[0][1]), dtype=np.float32))])
        h, m = transpose(H2), transpose(M2)
        return reshape(transpose(h[i]) * self.u_label.expr(), (len(self.irels), self.options.label_mlp+1)) * m[j]

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def getInputLayer(self, sentence, train):
        embed = []
        for entry in sentence:
            wordvec = self.wlookup[int(self.vocab.get(entry.norm, 0))] if self.options.we > 0 else None
            ewordvec = self.elookup[int(self.evocab.get(entry.norm, 0))] if self.elookup else None
            posvec = self.plookup[int(self.pos[entry.pos])] if self.options.pe > 0 else None
            vec = concatenate(filter(None, [wordvec + ewordvec if ewordvec else wordvec, posvec]))
            if self.dropout and train:
                vec = dropout(vec, self.options.dropout)
            embed.append(vec)
        return embed

    def getLstmLayer(self, sentence, train):
        h = concatenate_cols(self.deep_lstms.transduce(self.getInputLayer(sentence, train)))
        H = self.activation(affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.arc_mlp_dep_b.expr(),self.arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.label_mlp_dep_b.expr(),self.label_mlp_dep.expr(), h]))
        if self.dropout and train:
            d = self.options.dropout
            H, M, HL, ML = dropout(H, d), dropout(M, d), dropout(HL, d), dropout(ML, d)
        return H, M, HL, ML

    def Predict(self, conll_path, greedy, non_proj):
        with codecs.open(conll_path, 'r', encoding='utf-8') as conllFP:
            for iSentence, sentence in enumerate(read_conll(conllFP)):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                H, M, HL, ML = self.getLstmLayer(conll_sentence, False)

                if greedy:
                    scores = self.__evaluate(H, M).value()
                    for modifier, entry in enumerate(conll_sentence[1:]):
                        entry.pred_parent_id = np.argmax(scores[modifier + 1])
                        s = self.__evaluateLabel(entry.pred_parent_id, modifier + 1,HL, ML).value()
                        conll_sentence[modifier + 1].pred_relation = self.irels[max(enumerate(s), key=itemgetter(1))[0]]
                else:
                    scores = self.__evaluate(H, M)
                    probs = softmax(transpose(scores)).npvalue().T
                    scores = scores.npvalue().T
                    heads = None
                    if non_proj:
                        heads = decoder.arc_argmax(probs, len(conll_sentence))
                    else:
                        heads = decoder.parse_proj(scores)
                    for entry, head in zip(conll_sentence, heads):
                        entry.pred_parent_id = head
                        entry.pred_relation = '_'

                    for modifier, head in enumerate(heads[1:]):
                        scores = self.__evaluateLabel(head, modifier + 1, HL, ML).value()
                        conll_sentence[modifier + 1].pred_relation = self.irels[
                            max(enumerate(scores), key=itemgetter(1))[0]]
                renew_cg()
                yield sentence

    def Train(self, conll_path, t):
        self.deep_lstms.set_dropout(self.options.dropout)
        with open(conll_path, 'r') as conllFP:
            shuffledData = list(read_conll(conllFP))
            random.shuffle(shuffledData)
            start, lss, total, loss_vec = time.time(), 0, 0, []
            for iSentence, sentence in enumerate(shuffledData):
                conll_sentence = [entry for entry in sentence if isinstance(entry, utils.ConllEntry)]
                H, M, HL, ML = self.getLstmLayer(conll_sentence, True)
                scores = self.__evaluate(H, M)
                for modifier, entry in enumerate(conll_sentence[1:]):
                    loss_vec.append(pickneglogsoftmax(scores[modifier + 1], entry.parent_id))
                    loss_vec.append(
                        pickneglogsoftmax(self.__evaluateLabel(entry.parent_id, modifier + 1, HL, ML), self.rels[entry.relation]))

                if len(loss_vec) >= 2 * self.options.batch:
                    err = esum(loss_vec) / len(loss_vec)
                    err.scalar_value()
                    lss += err.value()
                    total += 1
                    if total % 10 == 0:
                        print 'Processing sentence number:', iSentence + 1, 'Loss:', float(
                            lss) / total, 'Time', time.time() - start
                        lss, total, start = 0, 0, time.time()
                    err.backward()
                    self.trainer.update()
                    renew_cg()
                    loss_vec, t = [], t + 1
                    if self.options.anneal:
                        decay_steps = min(1.0, float(t) / 50000)
                        lr = self.options.lr * 0.75 ** decay_steps
                        self.trainer.learning_rate = lr

        renew_cg()
        print 'current learning rate', self.trainer.learning_rate, 't:', t
        self.deep_lstms.disable_dropout()
        return t
