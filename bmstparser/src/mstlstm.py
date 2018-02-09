from dynet import *
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder, gzip
import numpy as np
import codecs
from linalg import *

class MSTParserLSTM:
    def __init__(self, pos, rels, w2i, chars, langs, options, from_model=None):
        self.model = Model()
        self.PAD = 1
        self.options = options
        self.trainer = AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': tanh, 'sigmoid': logistic, 'relu': rectify, 'leaky': (lambda x: bmax(.1 * x, x))}
        self.activation = self.activations[options.activation]
        self.dropout = False if options.dropout == 0.0 else True
        self.vocab = {word: ind + 2 for word, ind in w2i.iteritems()}
        self.langs = {lang: ind + 2 for ind,lang in enumerate(langs)}
        self.pos = {word: ind + 2 for ind, word in enumerate(pos)}
        self.rels = {word: ind + 1 for ind, word in enumerate(rels)}
        self.chars = {c: i + 2 for i, c in enumerate(chars)}
        self.root_id = self.rels['root']
        self.irels = ['PAD'] + rels
        self.PAD_REL = 0
        edim = options.we
        if not from_model:
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim))
            self.lang_lookup = self.model.add_lookup_parameters((len(langs) + 2, options.le))
            self.elookup = None
            if options.external_embedding is not None:
                external_embedding_fp = gzip.open(options.external_embedding, 'r')
                external_embedding = {line.split(' ')[0]: [float(f) for f in line.strip().split(' ')[1:]] for line in
                                      external_embedding_fp if len(line.split(' ')) > 2}
                external_embedding_fp.close()
                self.evocab = {word: i + 2 for i, word in enumerate(external_embedding)}

                edim = len(external_embedding.values()[0])
                self.elookup = self.model.add_lookup_parameters((len(external_embedding) + 2, edim))
                self.elookup.set_updated(False)
                self.elookup.init_row(0, [0] * edim)
                for word in external_embedding.keys():
                    self.elookup.init_row(self.evocab[word], external_embedding[word])
                    if word == '_UNK_':
                        self.elookup.init_row(0, external_embedding[word])

                print 'Initialized with pre-trained embedding. Vector dimensions', edim, 'and', len(external_embedding),\
                    'words, number of training words', len(w2i) + 2

            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))

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
            input_dim = edim + options.pe + options.le if self.options.use_pos else edim
            self.deep_lstms = BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model, VanillaLSTMBuilder)
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
                b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
                self.deep_lstms.builder_layers[i] = (b0, b1)

            if options.use_char:
                self.clookup = self.model.add_lookup_parameters((len(chars) + 2, options.ce))
                self.char_lstm = BiRNNBuilder(1, options.ce + options.le, edim, self.model, VanillaLSTMBuilder)

            self.a_wlookup = np.ndarray(shape=(options.we, len(w2i)+2), dtype=float)
            self.a_wlookup.fill(0)
            self.a_lang_lookup = np.ndarray(shape=(options.le, len(langs) + 2), dtype=float)
            self.a_lang_lookup.fill(0)
            self.a_plookup = np.ndarray(shape=(options.pe, len(pos)+2), dtype=float)
            self.a_plookup.fill(0)
            self.a_arc_mlp_head = np.ndarray(shape=(options.arc_mlp, options.le + options.rnn * 2), dtype=float)
            self.a_arc_mlp_head.fill(0)
            self.a_arc_mlp_head_b = np.ndarray(shape=(options.arc_mlp,), dtype=float)
            self.a_arc_mlp_head_b.fill(0)
            self.a_label_mlp_head = np.ndarray(shape=(options.label_mlp, options.le + options.rnn * 2), dtype=float)
            self.a_label_mlp_head.fill(0)
            self.a_label_mlp_head_b = np.ndarray(shape=(options.label_mlp,), dtype=float)
            self.a_label_mlp_head_b.fill(0)
            self.a_arc_mlp_dep = np.ndarray(shape=(options.arc_mlp, options.le + options.rnn * 2), dtype=float)
            self.a_arc_mlp_dep.fill(0)
            self.a_arc_mlp_dep_b = np.ndarray(shape=(options.arc_mlp,), dtype=float)
            self.a_arc_mlp_dep_b.fill(0)
            self.a_label_mlp_dep = np.ndarray(shape=(options.label_mlp, options.le + options.rnn * 2), dtype=float)
            self.a_label_mlp_dep.fill(0)
            self.a_label_mlp_dep_b =  np.ndarray(shape=(options.label_mlp,), dtype=float)
            self.a_label_mlp_dep_b.fill(0)
            self.a_w_arc = np.ndarray(shape=(options.arc_mlp,options.arc_mlp+1), dtype=float)
            self.a_w_arc.fill(0)
            self.a_u_label = np.ndarray(shape=(len(self.irels) * (options.label_mlp + 1), options.label_mlp + 1), dtype=float)
            self.a_u_label.fill(0)

            self.a_lstms = []
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                this_layer = []
                for j in range(len(params)):
                    dim = params[j].expr().dim()
                    if (j+1)%3==0: # bias
                        x = np.ndarray(shape=(dim[0][0],), dtype=float)
                        x.fill(0)
                        this_layer.append(x)
                    else:
                        x = np.ndarray(shape=(dim[0][0],dim[0][1]), dtype=float)
                        x.fill(0)
                        this_layer.append(x)
                self.a_lstms.append(this_layer)

            if options.use_char:
                self.a_clookup = np.ndarray(shape=(options.ce, len(chars) + 2), dtype=float)
                self.ac_lstms = []
                for i in range(len(self.char_lstm.builder_layers)):
                    builder = self.char_lstm.builder_layers[i]
                    params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                    this_layer = []
                    for j in range(len(params)):
                        dim = params[j].expr().dim()
                        if (j + 1) % 3 == 0:  # bias
                            this_layer.append(np.ndarray(shape=(dim[0][0],), dtype=float))
                        else:
                            this_layer.append(np.ndarray(shape=(dim[0][0], dim[0][1]), dtype=float))
                    self.ac_lstms.append(this_layer)
        else:
            self.wlookup = self.model.add_lookup_parameters((len(w2i) + 2, edim), init=NumpyInitializer(from_model.a_wlookup))
            self.lang_lookup = self.model.add_lookup_parameters((len(langs) + 2, options.le), init=NumpyInitializer(from_model.a_lang_lookup))
            if from_model.evocab:
                self.evocab = from_model.evocab
                self.elookup =  self.model.add_lookup_parameters((len(self.evocab) + 2, edim), init=NumpyInitializer(from_model.elookup.expr().npvalue()))
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe), init=NumpyInitializer(from_model.a_plookup))
            self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.le + options.rnn * 2), init=NumpyInitializer(from_model.a_arc_mlp_head))
            self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,),init=NumpyInitializer(from_model.a_arc_mlp_head_b))
            self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.le + options.rnn * 2),init=NumpyInitializer(from_model.a_label_mlp_head))
            self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,),init=NumpyInitializer(from_model.a_label_mlp_head_b))
            self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.le + options.rnn * 2),init=NumpyInitializer(from_model.a_arc_mlp_dep))
            self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,),init=NumpyInitializer(from_model.a_arc_mlp_dep_b))
            self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.le + options.rnn * 2), init = NumpyInitializer(from_model.a_label_mlp_dep))
            self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init = NumpyInitializer(from_model.a_label_mlp_dep_b))
            self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp + 1), init = NumpyInitializer(from_model.a_w_arc))
            self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp + 1), options.label_mlp + 1),init = NumpyInitializer(from_model.a_u_label))
            input_dim = edim + options.pe + options.le if self.options.use_pos else edim
            self.deep_lstms = BiRNNBuilder(options.layer, input_dim, options.rnn * 2, self.model, VanillaLSTMBuilder)
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                for j in range(len(params)):
                    params[j].set_value(from_model.a_lstms[i][j])

            if options.use_char:
                self.clookup = self.model.add_lookup_parameters((len(chars) + 2, options.ce),  init=NumpyInitializer(from_model.a_clookup))
                self.char_lstm = BiRNNBuilder(1, options.ce + options.le, edim, self.model, VanillaLSTMBuilder)
                for i in range(len(self.char_lstm.builder_layers)):
                    builder = self.char_lstm.builder_layers[i]
                    params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                    for j in range(len(params)):
                        params[j].set_value(from_model.ac_lstms[i][j])

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                le_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                if self.options.use_pos:
                    tag_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                    scale = 4. / (2. * word_mask + tag_mask + le_mask+ 1e-12) if not self.options.use_char else 6. / (4. * word_mask + tag_mask + le_mask + 1e-12)
                    word_mask *= scale
                    tag_mask *= scale
                    le_mask *= scale
                    word_mask = inputTensor(word_mask, batched=True)
                    tag_mask = inputTensor(tag_mask, batched=True)
                    le_mask = inputTensor(le_mask, batched=True)
                    ret.append((word_mask, tag_mask, le_mask))
                else:
                    scale = 2. / (2. * word_mask + 1e-12) if not self.options.use_char else 4. / (4. * word_mask + 1e-12)
                    word_mask *= scale
                    word_mask = inputTensor(word_mask, batched=True)
                    ret.append(word_mask)
            return ret

        self.generate_emb_mask = _emb_mask_generator

    def moving_avg(self, r1, r2):
        self.a_wlookup = r1 * self.a_wlookup + r2 * self.wlookup.expr().npvalue()
        self.a_lang_lookup = r1 * self.a_lang_lookup + r2 * self.lang_lookup.expr().npvalue()
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

        if self.options.use_char:
            self.a_clookup = r1 * self.a_clookup + r2 * self.clookup.expr().npvalue()
            for i in range(len(self.char_lstm.builder_layers)):
                builder = self.char_lstm.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                for j in range(len(params)):
                    self.ac_lstms[i][j] = r1 * self.ac_lstms[i][j] + r2 * params[j].expr().npvalue()

        renew_cg()

    def bilinear(self, M, W, H, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
        if bias_x:
            M = concatenate([M, inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        if bias_y:
            H = concatenate([H, inputTensor(np.ones((1, seq_len), dtype=np.float32))])

        nx, ny = input_size + bias_x, input_size + bias_y
        lin = W * M
        if num_outputs > 1:
            lin = reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
        blin = transpose(H) * lin
        if num_outputs > 1:
            blin = reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
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

    def bi_rnn(self, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
        for fb, bb in self.deep_lstms.builder_layers:
            f, b = fb.initial_state(), bb.initial_state()
            fb.set_dropouts(dropout_x, dropout_h)
            bb.set_dropouts(dropout_x, dropout_h)
            if batch_size is not None:
                fb.set_dropout_masks(batch_size)
                bb.set_dropout_masks(batch_size)
            fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
            inputs = [concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs


    def rnn_mlp(self, sens, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words, pwords, pos, chars, langs, clangs = sens[0], sens[1], sens[2], sens[5], sens[6], sens[7]
        if self.options.use_char:
            cembed = [concatenate([lookup_batch(self.clookup, c),lookup_batch(self.lang_lookup, l)]) for c, l in zip(chars, clangs)]
            char_fwd, char_bckd = self.char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1],\
                                  self.char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnn = reshape(concatenate_cols([char_fwd, char_bckd]), (self.options.we, words.shape[0]*words.shape[1]))
            cnn_reps = [list() for _ in range(len(words))]
            for i in range(words.shape[0]):
                cnn_reps[i] = pick_batch(crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

            wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.elookup, pwords[i]) + cnn_reps[i] for i in range(len(words))]
        else:
            wembed = [lookup_batch(self.wlookup, words[i]) + lookup_batch(self.elookup, pwords[i]) for i in range(len(words))]
        posembed = [lookup_batch(self.plookup, pos[i]) for i in range(len(pos))] if self.options.use_pos else None
        lembeds = [lookup_batch(self.lang_lookup, langs[i]) for i in range(len(langs))]

        if not train:
            inputs = [concatenate([w, pos, lembed]) for w, pos, lembed in zip(wembed, posembed, lembeds)] if self.options.use_pos else wembed
        else:
            emb_masks = self.generate_emb_mask(words.shape[0], words.shape[1])
            inputs = [concatenate([cmult(w, wm), cmult(pos, posm), cmult(lembed, lem)]) for w, pos,lembed, (wm, posm, lem) in zip(wembed, posembed, lembeds, emb_masks)] if self.options.use_pos\
                else [cmult(w, wm) for w, wm in zip(wembed, emb_masks)]

        d = self.options.dropout
        h_out = self.bi_rnn(inputs, words.shape[1], d if train else 0, d if train else 0) #self.deep_lstms.transduce(inputs)
        h_out = [concatenate([lembeds[i], h_out[i]]) for i in range(len(h_out))]
        h = dropout_dim(concatenate_cols(h_out), 1, d) if train else concatenate_cols(h_out)
        H = self.activation(affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

        if train:
            H, M, HL, ML = dropout_dim(H, 1, d), dropout_dim(M, 1, d), dropout_dim(HL, 1, d), dropout_dim(ML, 1, d)
        return H, M, HL, ML

    def build_graph(self, mini_batch, t=1, train=True):
        H, M, HL, ML = self.rnn_mlp(mini_batch, train)
        arc_scores = self.bilinear(M, self.w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1],1, True, False)
        rel_scores = self.bilinear(ML, self.u_label.expr(), HL, self.options.label_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1], len(self.irels), True, True)
        flat_scores = reshape(arc_scores, (mini_batch[0].shape[0],), mini_batch[0].shape[0]* mini_batch[0].shape[1])
        flat_rel_scores = reshape(rel_scores, (mini_batch[0].shape[0], len(self.irels)), mini_batch[0].shape[0]* mini_batch[0].shape[1])
        masks = np.reshape(mini_batch[-1], (-1,), 'F')
        mask_1D_tensor = inputTensor(masks, batched=True)
        n_tokens = np.sum(masks)
        if train:
            heads = np.reshape(mini_batch[3], (-1,), 'F')
            partial_rel_scores = pick_batch(flat_rel_scores, heads)
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

            for msk, arc_prob, rel_prob in zip(np.transpose(mini_batch[-1]), arc_probs, rel_probs):
                # parse sentences one by one
                msk[0] = 1.
                sent_len = int(np.sum(msk))
                arc_pred = decoder.arc_argmax(arc_prob, sent_len, msk)
                rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
                rel_pred = decoder.rel_argmax(rel_prob, sent_len, self.PAD_REL, self.root_id)
                outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))
            renew_cg()
            return outputs