from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder
import numpy as np
import codecs
from linalg import *

class MSTParserLSTM:
    def __init__(self, pos, rels, w2i, options, from_model=None):
        self.model = Model()
        self.PAD = 1
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'leaky': (lambda x: bmax(.1 * x, x))}
        self.activation = self.activations[options.activation]
        self.dropout = False if options.dropout == 0.0 else True
        self.vocab = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.pos = {word: ind + 2 for ind, word in enumerate(pos)}
        self.rels = {word: ind + 1 for ind, word in enumerate(rels)}
        self.root_id = self.rels['root']
        self.irels = ['PAD'] + rels
        self.PAD_REL = 0
        edim = options.we
        if not from_model:
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim))
            self.elookup = None
            if options.external_embedding is not None:
                external_embedding_fp = open(options.external_embedding, 'r')
                external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                      external_embedding_fp if len(line.split(' ')) > 2}
                external_embedding_fp.close()
                self.evocab = {word: i + 2 for i, word in enumerate(external_embedding)}

                edim = len(external_embedding.values()[0])
                self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim), init=ConstInitializer(0))
                self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 2, edim))
                self.elookup.set_updated(False)
                self.elookup.init_row(0, [0] * edim)
                for word in external_embedding.keys():
                    self.elookup.init_row(self.evocab[word], external_embedding[word])

                print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(external_embedding),\
                    'words, number of training words', len(w2i) + 2

            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))
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
            self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp+1), options.label_mlp+1), init = ConstInitializer(0))

            self.a_wlookup = np.ndarray(shape=(options.we, len(w2i)+2), dtype=float)
            self.a_plookup = np.ndarray(shape=(options.pe, len(pos)+2), dtype=float)
            self.a_arc_mlp_head = np.ndarray(shape=(options.arc_mlp, options.rnn * 2), dtype=float)
            self.a_arc_mlp_head_b = np.ndarray(shape=(options.arc_mlp,), dtype=float)
            self.a_label_mlp_head = np.ndarray(shape=(options.label_mlp, options.rnn * 2), dtype=float)
            self.a_label_mlp_head_b = np.ndarray(shape=(options.label_mlp,), dtype=float)
            self.a_arc_mlp_dep = np.ndarray(shape=(options.arc_mlp, options.rnn * 2), dtype=float)
            self.a_arc_mlp_dep_b = np.ndarray(shape=(options.arc_mlp,), dtype=float)
            self.a_label_mlp_dep = np.ndarray(shape=(options.label_mlp, options.rnn * 2), dtype=float)
            self.a_label_mlp_dep_b =  np.ndarray(shape=(options.label_mlp,), dtype=float)
            self.a_w_arc = np.ndarray(shape=(options.arc_mlp,options.arc_mlp+1), dtype=float)
            self.a_u_label = np.ndarray(shape=(len(self.irels) * (options.label_mlp + 1), options.label_mlp + 1), dtype=float)

            self.a_lstms = []
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                this_layer = []
                for j in range(len(params)):
                    dim = params[j].expr().dim()
                    if (j+1)%3==0: # bias
                        this_layer.append(np.ndarray(shape=(dim[0][0],), dtype=float))
                    else:
                        this_layer.append(np.ndarray(shape=(dim[0][0],dim[0][1]), dtype=float))
                self.a_lstms.append(this_layer)
        else:
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim), init=NumpyInitializer(from_model.a_wlookup))
            if from_model.evocab:
                self.evocab = from_model.evocab
                self.elookup =  self.model.add_lookup_parameters((len(self.evocab) + 2, edim), init=NumpyInitializer(from_model.elookup.expr().npvalue()))
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe), init=NumpyInitializer(from_model.a_plookup))
            self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.rnn * 2), init=NumpyInitializer(from_model.a_arc_mlp_head))
            self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,),init=NumpyInitializer(from_model.a_arc_mlp_head_b))
            self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.rnn * 2),init=NumpyInitializer(from_model.a_label_mlp_head))
            self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,),init=NumpyInitializer(from_model.a_label_mlp_head_b))
            self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.rnn * 2),init=NumpyInitializer(from_model.a_arc_mlp_dep))
            self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,),init=NumpyInitializer(from_model.a_arc_mlp_dep_b))
            self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.rnn * 2), init = NumpyInitializer(from_model.a_label_mlp_dep))
            self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init = NumpyInitializer(from_model.a_label_mlp_dep_b))
            self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp + 1), init = NumpyInitializer(from_model.a_w_arc))
            self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp + 1), options.label_mlp + 1),init = NumpyInitializer(from_model.a_u_label))
            self.deep_lstms = BiRNNBuilder(options.layer, edim + options.pe, options.rnn * 2, self.model, VanillaLSTMBuilder)
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                for j in range(len(params)):
                    params[j].set_value(from_model.a_lstms[i][j])

    def moving_avg(self, r1, r2):
        self.a_wlookup = r1 * self.a_wlookup + r2 * self.wlookup.expr().npvalue()
        self.a_plookup = r1 * self.a_plookup + r2 * self.plookup.expr().npvalue()
        self.a_arc_mlp_head = r1 * self.a_arc_mlp_head + r2 * self.arc_mlp_head.expr().npvalue()
        self.a_arc_mlp_head_b = r1 * self.a_arc_mlp_head_b + r2 * self.arc_mlp_head_b.expr().npvalue()
        self.a_label_mlp_head = r1 * self.a_label_mlp_head + r2 * self.label_mlp_head.expr().npvalue()
        self.a_label_mlp_head_b = r1 * self.a_label_mlp_head_b + r2 * self.label_mlp_head_b.expr().npvalue()
        self.a_arc_mlp_dep = r1 * self.a_arc_mlp_dep + r2 * self.arc_mlp_dep.expr().npvalue()
        self.a_arc_mlp_dep_b = r1 * self.a_arc_mlp_dep_b + r2 * self.arc_mlp_dep_b.expr().npvalue()
        self.a_label_mlp_dep = r1 * self.a_label_mlp_dep + r2 * self.label_mlp_dep.expr().npvalue()
        self.a_label_mlp_dep_b = r1 * self.a_label_mlp_dep_b + r2 * self.label_mlp_dep_b.expr().npvalue()
        self.a_w_arc = r1 * self.a_w_arc + r2 * self.w_arc.expr().npvalue()
        self.a_u_label = r1 * self.a_u_label + r2 * self.u_label.expr().npvalue()

        for i in range(len(self.deep_lstms.builder_layers)):
            builder = self.deep_lstms.builder_layers[i]
            params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
            for j in range(len(params)):
                self.a_lstms[i][j] = r1 * self.a_lstms[i][j] + r2 * params[j].expr().npvalue()

        renew_cg()

    def bilinear(self, M, W, H, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
        # x,y: (input_size x seq_len) x batch_size
        if bias_x:
            M = concatenate([M, inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        if bias_y:
            H = concatenate([H, inputTensor(np.ones((1, seq_len), dtype=np.float32))])

        nx, ny = input_size + bias_x, input_size + bias_y
        # W: (num_outputs x ny) x nx
        lin = W * M
        if num_outputs > 1:
            lin = reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
        blin = transpose(H) * lin
        if num_outputs > 1:
            blin = reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
        # seq_len_y x seq_len_x if output_size == 1
        # seq_len_y x num_outputs x seq_len_x else
        return blin

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
            if ewordvec: wordvec = wordvec + ewordvec
            posvec = self.plookup[int(self.pos[entry.pos])] if self.options.pe > 0 else None
            vec = concatenate(filter(None, [wordvec, posvec]))
            if train:
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

    def rnn_batch(self, sens, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words,pwords,pos = sens[0], sens[1], sens[2]
        wembed = [lookup_batch(self.wlookup, words[i]) for i in range(len(words))]
        pwembed = [lookup_batch(self.elookup, pwords[i]) for i in range(len(pwords))]
        posembed = [lookup_batch(self.plookup, pos[i]) for i in range(len(pos))]

        inputs = [concatenate([wembed[i]+pwembed[i], posembed[i]]) for i in range(len(words))]
        if train:
            inputs = [dropout(input, self.options.dropout) for input in inputs]
            self.deep_lstms.set_dropout(self.options.dropout)
        else:
            self.deep_lstms.disable_dropout()
        h_out = self.deep_lstms.transduce(inputs)
        h = concatenate_cols(h_out)
        H = self.activation(affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

        if self.dropout and train:
            d = self.options.dropout
            H, M, HL, ML = dropout(H, d), dropout(M, d), dropout(HL, d), dropout(ML, d)
        return H, M, HL, ML

    def Predict(self, conll_path, greedy, non_proj):
        self.deep_lstms.disable_dropout()
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

    def build_graph(self, mini_batch, t=1, train=True):
        H, M, HL, ML = self.rnn_batch(mini_batch, train)
        arc_scores = self.bilinear(M, self.w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1],1, True, False)
        rel_scores = self.bilinear(ML, self.u_label.expr(), HL, self.options.label_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1], len(self.irels), True, True)
        flat_scores = reshape(arc_scores, (mini_batch[0].shape[0],), mini_batch[0].shape[0]* mini_batch[0].shape[1])
        flat_rel_scores = reshape(rel_scores, (mini_batch[0].shape[0], len(self.irels)), mini_batch[0].shape[0]* mini_batch[0].shape[1])
        heads = np.reshape(mini_batch[3], (-1,), 'F') if train else np.reshape(arc_scores.npvalue().argmax(0), (-1,), 'F')
        partial_rel_scores = pick_batch(flat_rel_scores, heads)
        masks = np.reshape(mini_batch[5], (-1,), 'F')
        mask_1D_tensor = inputTensor(masks, batched=True)
        n_tokens = np.sum(masks)
        if train:
            gold_relations = np.reshape(mini_batch[4], (-1,), 'F')
            arc_losses = pickneglogsoftmax_batch(flat_scores, heads)
            arc_loss = sum_batches(arc_losses*mask_1D_tensor)/n_tokens
            rel_losses = pickneglogsoftmax_batch(partial_rel_scores, gold_relations)
            rel_loss = sum_batches(rel_losses*mask_1D_tensor)/n_tokens
            err = 0.5 * (arc_loss + rel_loss)
            err.scalar_value()
            loss = err.value()
            err.backward()
            self.trainer.update()
            renew_cg()
            ratio = min(0.9999, float(t) / (9 + t))
            self.moving_avg(ratio, 1 - ratio)
            return t + 1, loss
        else:
            arc_probs = np.transpose(np.reshape(softmax(flat_scores).npvalue(), (mini_batch[0].shape[0],  mini_batch[0].shape[0],  mini_batch[0].shape[1]), 'F'))
            rel_probs = np.transpose(np.reshape(softmax(transpose(flat_rel_scores)).npvalue(),
                                                (len(self.irels), mini_batch[0].shape[0], mini_batch[0].shape[0], mini_batch[0].shape[1]), 'F'))
            outputs = []

            for msk, arc_prob, rel_prob in zip(np.transpose(mini_batch[5]), arc_probs, rel_probs):
                # parse sentences one by one
                msk[0] = 1.
                sent_len = int(np.sum(msk))
                arc_pred = decoder.arc_argmax(arc_prob, sent_len, msk)
                rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
                rel_pred = decoder.rel_argmax(rel_prob, sent_len, self.PAD,self.root_id)
                outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))
            renew_cg()
            return outputs



