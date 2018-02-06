import dynet as dy
from utils import read_conll, write_conll, ConllEntry
from collections import defaultdict
from operator import itemgetter
import time, random, decoder, gzip, pickle
import numpy as np
import codecs, os, sys, math
from linalg import *

class MSTParserLSTM:
    def __init__(self, pos, rels, options, words, chars, model_path=None):
        self.model = dy.Model()
        self.PAD = 1
        self.options = options
        self.trainer = dy.AdamTrainer(self.model, options.lr, options.beta1, options.beta2)
        self.activations = {'tanh': dy.tanh, 'sigmoid': dy.logistic, 'relu': dy.rectify, 'leaky': (lambda x: dy.bmax(.1 * x, x))}
        self.activation = self.activations[options.activation]
        self.dropout = False if options.dropout == 0.0 else True
        self.pos = {p: ind + 2 for ind, p in enumerate(pos)}
        self.rels = {word: ind + 1 for ind, word in enumerate(rels)}
        self.root_id = self.rels['root']
        self.irels = ['PAD'] + rels
        self.PAD_REL = 0
        edim = options.we

        if model_path:
            with open(model_path, 'r') as paramsfp:
                lang2id, deep_lstm_params, char_lstm_params, clookup_params, proj_mat_params, plookup_params, lang_lookup_params, arc_mlp_head_params, arc_mlp_head_b_params, label_mlp_head_params, label_mlp_head_b_params, arc_mlp_dep_params, arc_mlp_dep_params, arc_mlp_dep_b_params, arc_mlp_dep_b_params, label_mlp_dep_params, label_mlp_dep_b_params, w_arc_params, u_label_params = pickle.load(paramsfp)

        if model_path:
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe), init=dy.NumpyInitializer(plookup_params))
        else:
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, options.pe))

        self.chars = dict()
        self.evocab = dict()
        self.clookup = dict()
        self.char_lstm = dict()
        self.proj_mat = dict()
        external_embedding = dict()
        word_index = 2
        for f in os.listdir(options.external_embedding):
            lang = f[:-3]
            efp = gzip.open(options.external_embedding + '/' + f, 'r')
            external_embedding[lang] =  dict()
            for line in efp:
                spl = line.strip().split(' ')
                if len(spl) > 2:
                    w = spl[0]
                    if w in words[lang]:
                        external_embedding[lang][w] = [float(f) for f in spl[1:]]
            efp.close()

            self.evocab[lang] = {word: i + word_index for i, word in enumerate(external_embedding[lang])}
            word_index += len(self.evocab[lang])

            if len(external_embedding[lang]) > 0:
                edim = len(external_embedding[lang].values()[0])
            self.chars[lang] = {c: i + 2 for i, c in enumerate(chars[lang])}

            print 'Loaded vector', edim, 'and', len(external_embedding[lang]), 'for', lang
            if model_path:
                self.clookup[lang] = self.model.add_lookup_parameters((len(chars[lang]) + 2, options.ce), init=dy.NumpyInitializer(clookup_params[lang]))
            else:
                self.clookup[lang] = self.model.add_lookup_parameters((len(chars[lang]) + 2, options.ce))

            if not options.tune_net: self.clookup[lang].set_updated(False)

            self.char_lstm[lang] = dy.BiRNNBuilder(1, options.ce, edim, self.model, dy.VanillaLSTMBuilder)
            if model_path:
                for i in range(len(self.char_lstm[lang].builder_layers)):
                    builder = self.char_lstm[lang].builder_layers[i]
                    params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                    for j in range(len(params)):
                        params[j].set_value(char_lstm_params[lang][i][j])
                        if not options.tune_net: params[j].set_updated(False)

            if model_path:
                self.proj_mat[lang] = self.model.add_parameters((edim + options.pe, edim + options.pe),
                                                                init=dy.NumpyInitializer(proj_mat_params[lang]))
            else:
                self.proj_mat[lang] = self.model.add_parameters((edim + options.pe, edim + options.pe))


            if not options.tune_net: self.proj_mat[lang].set_updated(False)

        self.elookup = self.model.add_lookup_parameters((word_index, edim))
        self.num_all_words = word_index
        self.elookup.set_updated(False)
        self.elookup.init_row(0, [0] * edim)
        self.elookup.init_row(1, [0] * edim)
        for lang in self.evocab.keys():
            for word in external_embedding[lang].keys():
                self.elookup.init_row(self.evocab[lang][word], external_embedding[lang][word])

        self.lang2id = {lang: i for i, lang in enumerate(sorted(list(words.keys())))}
        print self.lang2id
        if model_path:
            self.lang_lookup = self.model.add_lookup_parameters((len(self.lang2id), options.le), init=dy.NumpyInitializer(lang_lookup_params))
        else:
            self.lang_lookup = self.model.add_lookup_parameters((len(self.lang2id), options.le))

        input_dim = edim + options.pe if self.options.use_pos else edim

        self.deep_lstms = dy.BiRNNBuilder(options.layer, input_dim + options.le, options.rnn * 2, self.model, dy.VanillaLSTMBuilder)
        if not model_path:
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                b0 = orthonormal_VanillaLSTMBuilder(builder[0], builder[0].spec[1], builder[0].spec[2])
                b1 = orthonormal_VanillaLSTMBuilder(builder[1], builder[1].spec[1], builder[1].spec[2])
                self.deep_lstms.builder_layers[i] = (b0, b1)
        else:
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                for j in range(len(params)):
                    params[j].set_value(deep_lstm_params[i][j])
                    if not options.tune_net: params[j].set_updated(False)

        w_mlp_arc = orthonormal_initializer(options.arc_mlp, options.rnn * 2)
        w_mlp_label = orthonormal_initializer(options.label_mlp, options.rnn * 2)
        if not model_path:
            self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.rnn * 2), init= dy.NumpyInitializer(w_mlp_arc))
            self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init = dy.ConstInitializer(0))
            self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.rnn * 2), init= dy.NumpyInitializer(w_mlp_label))
            self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init = dy.ConstInitializer(0))
            self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.rnn * 2), init= dy.NumpyInitializer(w_mlp_arc))
            self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init = dy.ConstInitializer(0))
            self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.rnn * 2), init= dy.NumpyInitializer(w_mlp_label))
            self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init = dy.ConstInitializer(0))
            self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp+1), init = dy.ConstInitializer(0))
            self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp+1), options.label_mlp+1), init = dy.ConstInitializer(0))
        else:
            self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, options.rnn * 2),
                                                          init=dy.NumpyInitializer(arc_mlp_head_params))
            self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init=dy.NumpyInitializer(arc_mlp_head_b_params))
            self.label_mlp_head = self.model.add_parameters((options.label_mlp, options.rnn * 2),
                                                            init=dy.NumpyInitializer(label_mlp_head_params))
            self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init=dy.NumpyInitializer(label_mlp_head_b_params))
            self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, options.rnn * 2),
                                                         init=dy.NumpyInitializer(arc_mlp_dep_params))
            self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init=dy.NumpyInitializer(arc_mlp_dep_b_params))
            self.label_mlp_dep = self.model.add_parameters((options.label_mlp, options.rnn * 2),
                                                           init=dy.NumpyInitializer(label_mlp_dep_params))
            self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init=dy.NumpyInitializer(label_mlp_dep_b_params))
            self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp + 1), init=dy.NumpyInitializer(w_arc_params))
            self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp + 1), options.label_mlp + 1),
                                                     init=dy.NumpyInitializer(u_label_params))

        self.lm_w = self.model.add_parameters((2, options.arc_mlp))
        self.lm_b = self.model.add_parameters((2,), init=dy.ConstInitializer(-math.log(2)))

        self.lmL_w = self.model.add_parameters((2, options.label_mlp))
        self.lmL_b = self.model.add_parameters((2,), init=dy.ConstInitializer(-math.log(2)))

        def _emb_mask_generator(seq_len, batch_size):
            ret = []
            for _ in xrange(seq_len):
                word_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                if self.options.use_pos:
                    tag_mask = np.random.binomial(1, 1. - self.options.dropout, batch_size).astype(np.float32)
                    scale = 3. / (2. * word_mask + tag_mask + 1e-12)
                    word_mask *= scale
                    tag_mask *= scale
                    word_mask = dy.inputTensor(word_mask, batched=True)
                    tag_mask = dy.inputTensor(tag_mask, batched=True)
                    ret.append((word_mask, tag_mask))
                else:
                    scale = 2. / (2. * word_mask + 1e-12) if not self.options.use_char else 4. / (4. * word_mask + 1e-12)
                    word_mask *= scale
                    word_mask = dy.inputTensor(word_mask, batched=True)
                    ret.append(word_mask)
            return ret

        self.generate_emb_mask = _emb_mask_generator

    def bilinear(self, M, W, H, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
        if bias_x:
            M = dy.concatenate([M, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
        if bias_y:
            H = dy.concatenate([H, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

        nx, ny = input_size + bias_x, input_size + bias_y
        lin = W * M
        if num_outputs > 1:
            lin = dy.reshape(lin, (ny, num_outputs * seq_len), batch_size=batch_size)
        blin =  dy.transpose(H) * lin
        if num_outputs > 1:
            blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
        return blin

    def __evaluate(self, H, M):
        M2 = dy.concatenate([M, dy.inputTensor(np.ones((1, M.dim()[0][1]), dtype=np.float32))])
        return  dy.transpose(H)*(self.w_arc.expr()*M2)

    def __evaluateLabel(self, i, j, HL, ML):
        H2 = dy.concatenate([HL, dy.inputTensor(np.ones((1, HL.dim()[0][1]), dtype=np.float32))])
        M2 = dy.concatenate([ML, dy.inputTensor(np.ones((1, ML.dim()[0][1]), dtype=np.float32))])
        h, m =  dy.transpose(H2),  dy.transpose(M2)
        return dy.reshape( dy.transpose(h[i]) * self.u_label.expr(), (len(self.irels), self.options.label_mlp+1)) * m[j]


    def load(self, filename):
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
            inputs = [dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
        return inputs


    def rnn_mlp(self, sens, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words, pos_tags, chars, langs = sens[0], sens[1], sens[4], sens[5]
        all_inputs = [0] * len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [dy.lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(len(words[lang])):
                cnn_reps[i] = dy.pick_batch(crnns, [i * words[lang].shape[1] + j for j in range(words[lang].shape[1])], 1)
            wembed = [dy.lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [dy.lookup_batch(self.plookup, pos_tags[lang][i]) for i in
                        range(len(pos_tags[lang]))] if self.options.use_pos else None
            lang_embeds = [dy.lookup_batch(self.lang_lookup, [self.lang2id[lang]]*len(pos_tags[lang][i])) for i in range(len(pos_tags[lang]))]

            if not train:
                inputs = [dy.concatenate([w, pos]) for w, pos in
                          zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                          zip(wembed, posembed, emb_masks)] if self.options.use_pos \
                    else [dy.cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            inputs = [dy.concatenate([inp, lembed]) for inp, lembed in zip(inputs, lang_embeds)]
            all_inputs[l] = inputs

        lstm_input = [dy.concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in
                      range(len(all_inputs[0]))]
        d = self.options.dropout
        h_out = self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)

        h =  dy.dropout_dim(dy.concatenate_cols(h_out), 1, d) if train else dy.concatenate_cols(h_out)
        H = self.activation(dy.affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(dy.affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(dy.affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(dy.affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

        if train:
            H, M, HL, ML =  dy.dropout_dim(H, 1, d),  dy.dropout_dim(M, 1, d),  dy.dropout_dim(HL, 1, d),  dy.dropout_dim(ML, 1, d)
        return H, M, HL, ML

    def shared_rnn_mlp(self, batch, train):
        '''
        Here, I assumed all sens have the same length.
        '''
        words, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = batch

        all_inputs = [0] * len(chars.keys())
        for l, lang in enumerate(chars.keys()):
            cembed = [dy.lookup_batch(self.clookup[lang], c) for c in chars[lang]]
            char_fwd = self.char_lstm[lang].builder_layers[0][0].initial_state().transduce(cembed)[-1]
            char_bckd = self.char_lstm[lang].builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
            crnns = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, chars[lang].shape[1]))
            cnn_reps = [list() for _ in range(len(words[lang]))]
            for i in range(words[lang].shape[0]):
                cnn_reps[i] = dy.pick_batch(crnns, char_batches[lang][i], 1)
            wembed = [dy.lookup_batch(self.elookup, words[lang][i]) + cnn_reps[i] for i in range(len(words[lang]))]
            posembed = [dy.lookup_batch(self.plookup, pos_tags[lang][i]) for i in
                        range(len(pos_tags[lang]))] if self.options.use_pos else None
            lang_embeds = [dy.lookup_batch(self.lang_lookup, [self.lang2id[lang]] * len(pos_tags[lang][i])) for i in
                           range(len(pos_tags[lang]))]

            if not train:
                inputs = [dy.concatenate([w, pos]) for w, pos in
                          zip(wembed, posembed)] if self.options.use_pos else wembed
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            else:
                emb_masks = self.generate_emb_mask(words[lang].shape[0], words[lang].shape[1])
                inputs = [dy.concatenate([dy.cmult(w, wm), dy.cmult(pos, posm)]) for w, pos, (wm, posm) in
                          zip(wembed, posembed, emb_masks)] if self.options.use_pos \
                    else [dy.cmult(w, wm) for w, wm in zip(wembed, emb_masks)]
                inputs = [dy.tanh(self.proj_mat[lang].expr() * inp) for inp in inputs]
            inputs = [dy.concatenate([inp, lembed]) for inp, lembed in zip(inputs, lang_embeds)]
            all_inputs[l] = inputs

        lstm_input = [dy.concatenate_to_batch([all_inputs[j][i] for j in range(len(all_inputs))]) for i in range(len(all_inputs[0]))]
        d = self.options.dropout
        h_out = self.bi_rnn(lstm_input, lstm_input[0].dim()[1], d if train else 0, d if train else 0)
        h = dy.dropout_dim(dy.concatenate_cols(h_out), 1, d) if train else dy.concatenate_cols(h_out)
        H = self.activation(dy.affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(dy.affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(dy.affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(dy.affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

        if train:
            H, M, HL, ML = dy.dropout_dim(H, 1, d), dy.dropout_dim(M, 1, d), dy.dropout_dim(HL, 1, d), dy.dropout_dim(
                ML, 1, d)
        return H, M, HL, ML

    def build_graph(self, mini_batch, t=1, train=True):
        H, M, HL, ML = self.rnn_mlp(mini_batch, train)
        shape_0, shape_1 = mini_batch[-3], mini_batch[-2]
        arc_scores = self.bilinear(M, self.w_arc.expr(), H, self.options.arc_mlp, shape_0, shape_1,1, True, False)
        rel_scores = self.bilinear(ML, self.u_label.expr(), HL, self.options.label_mlp, shape_0, shape_1, len(self.irels), True, True)
        flat_scores = dy.reshape(arc_scores, (shape_0,), shape_0* shape_1)
        flat_rel_scores = dy.reshape(rel_scores, (shape_0, len(self.irels)), shape_0* shape_1)
        masks = np.reshape(mini_batch[-1], (-1,), 'F')
        mask_1D_tensor = dy.inputTensor(masks, batched=True)
        n_tokens = np.sum(masks)
        if train:
            heads = np.reshape(mini_batch[2], (-1,), 'F')
            partial_rel_scores =  dy.pick_batch(flat_rel_scores, heads)
            gold_relations = np.reshape(mini_batch[3], (-1,), 'F')
            arc_losses =  dy.pickneglogsoftmax_batch(flat_scores, heads)
            arc_loss = dy.sum_batches(arc_losses*mask_1D_tensor)/n_tokens
            rel_losses =  dy.pickneglogsoftmax_batch(partial_rel_scores, gold_relations)
            rel_loss = dy.sum_batches(rel_losses*mask_1D_tensor)/n_tokens
            err = 0.5 * (arc_loss + rel_loss)
            err.scalar_value()
            loss = err.value()
            err.backward()
            self.trainer.update()
            dy.renew_cg()
            return t + 1, loss
        else:
            arc_probs = np.transpose(np.reshape(dy.softmax(flat_scores).npvalue(), (shape_0,  shape_0,  shape_1), 'F'))
            rel_probs = np.transpose(np.reshape(dy.softmax( dy.transpose(flat_rel_scores)).npvalue(),
                                                (len(self.irels), shape_0, shape_0, shape_1), 'F'))
            outputs = []

            for msk, arc_prob, rel_prob in zip(np.transpose(mini_batch[-1]), arc_probs, rel_probs):
                # parse sentences one by one
                msk[0] = 1.
                sent_len = int(np.sum(msk))
                arc_pred = decoder.arc_argmax(arc_prob, sent_len, msk)
                rel_prob = rel_prob[np.arange(len(arc_pred)), arc_pred]
                rel_pred = decoder.rel_argmax(rel_prob, sent_len, self.PAD_REL, self.root_id)
                outputs.append((arc_pred[1:sent_len], rel_pred[1:sent_len]))
            dy.renew_cg()
            return outputs

    def train_shared_rnn(self, mini_batch, train_both=True):
        pwords, pos_tags, chars, langs, signs, positions, batch_num, char_batches, masks = mini_batch
        # Getting the last hidden layer from BiLSTM.
        H, M, HL, ML = self.shared_rnn_mlp(mini_batch, True)
        dim_0, dim_1, dim_2 = H.dim()[0][1], H.dim()[0][0], H.dim()[1]
        ldim_0, ldim_1, ldim_2 = HL.dim()[0][1], HL.dim()[0][0], HL.dim()[1]
        H_i = [dy.transpose(dy.reshape(dy.pick(H, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        M_i = [dy.transpose(dy.reshape(dy.pick(M, i, 1), (dim_1, dim_2))) for i in range(dim_0)]
        HL_i = [dy.transpose(dy.reshape(dy.pick(HL, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        ML_i = [dy.transpose(dy.reshape(dy.pick(ML, i, 1), (ldim_1, ldim_2))) for i in range(ldim_0)]
        # Calculating the kq values for NCE.
        loss_values = []
        last_pos = H.dim()[0][1] - 1

        for b in batch_num:
            for i in range(len(batch_num[b])):
                lang1 = langs[b][i]
                pos1 = positions[b][i]
                b1 = batch_num[b][i]
                HVec1 = H_i[pos1][b1]
                MVec1 = M_i[pos1][b1]
                HLVec1 = HL_i[pos1][b1]
                MLVec1 = ML_i[pos1][b1]

                lm_out = dy.affine_transform([self.lm_b.expr(), self.lm_w.expr(), HVec1])
                loss_values.append(dy.pickneglogsoftmax(lm_out, signs[b][i]))

                lm_out = dy.affine_transform([self.lm_b.expr(), self.lm_w.expr(), MVec1])
                loss_values.append(dy.pickneglogsoftmax(lm_out, signs[b][i]))

                lm_out = dy.affine_transform([self.lmL_b.expr(), self.lmL_w.expr(), HLVec1])
                loss_values.append(dy.pickneglogsoftmax(lm_out, signs[b][i]))

                lm_out = dy.affine_transform([self.lmL_b.expr(), self.lmL_w.expr(), MLVec1])
                loss_values.append(dy.pickneglogsoftmax(lm_out, signs[b][i]))

                for j in range(i + 1, len(batch_num[b])):
                    lang2 = langs[b][j]
                    pos2 = positions[b][j]
                    b2 = batch_num[b][j]
                    if lang1 != lang2:
                        HVec2 = H_i[pos2][b2]
                        MVec2 = M_i[pos2][b2]
                        HLVec2 = HL_i[pos2][b2]
                        MLVec2 = ML_i[pos2][b2]

                        if signs[b][i] == signs[b][j] == 1:
                            ps_loss = -dy.sqrt(dy.squared_distance(HVec1, HVec2))
                            term = -dy.log(dy.logistic(ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(MVec1, MVec2))
                            term = -dy.log(dy.logistic(ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(HLVec1, HLVec2))
                            term = -dy.log(dy.logistic(ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(MLVec1, MLVec2))
                            term = -dy.log(dy.logistic(ps_loss))
                            loss_values.append(term)

                            # alignment-based negative position.
                            for _ in range(5):
                                s_neg_position, t_neg_position = random.randint(0, last_pos), random.randint(0, last_pos)
                                if s_neg_position != pos1:
                                    s_vec = H_i[s_neg_position][b1]
                                    d_s = dy.sqrt(dy.squared_distance(s_vec, HVec2))
                                    term = -dy.log(dy.logistic(-d_s))
                                    loss_values.append(term)

                                    s_vec = M_i[s_neg_position][b1]
                                    d_s = dy.sqrt(dy.squared_distance(s_vec, MVec2))
                                    term = -dy.log(dy.logistic(-d_s))
                                    loss_values.append(term)

                                    s_vec = HL_i[s_neg_position][b1]
                                    d_s = dy.sqrt(dy.squared_distance(s_vec, HLVec2))
                                    term = -dy.log(dy.logistic(-d_s))
                                    loss_values.append(term)

                                    s_vec = ML_i[s_neg_position][b1]
                                    d_s = dy.sqrt(dy.squared_distance(s_vec, MLVec2))
                                    term = -dy.log(dy.logistic(-d_s))
                                    loss_values.append(term)
                                if t_neg_position != pos2:
                                    t_vec = H_i[t_neg_position][b2]
                                    d_t = dy.sqrt(dy.squared_distance(HVec1, t_vec))
                                    term = -dy.log(dy.logistic(-d_t))
                                    loss_values.append(term)

                                    t_vec = M_i[t_neg_position][b2]
                                    d_t = dy.sqrt(dy.squared_distance(MVec1, t_vec))
                                    term = -dy.log(dy.logistic(-d_t))
                                    loss_values.append(term)

                                    t_vec = HL_i[t_neg_position][b2]
                                    d_t = dy.sqrt(dy.squared_distance(HLVec1, t_vec))
                                    term = -dy.log(dy.logistic(-d_t))
                                    loss_values.append(term)

                                    t_vec = ML_i[t_neg_position][b2]
                                    d_t = dy.sqrt(dy.squared_distance(MLVec1, t_vec))
                                    term = -dy.log(dy.logistic(-d_t))
                                    loss_values.append(term)

                        elif signs[b][i] == 1 or signs[b][j] == 1:
                            ps_loss = -dy.sqrt(dy.squared_distance(HVec1, HVec2))
                            term = -dy.log(dy.logistic(-ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(MVec1, MVec2))
                            term = -dy.log(dy.logistic(-ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(HLVec1, HLVec2))
                            term = -dy.log(dy.logistic(-ps_loss))
                            loss_values.append(term)

                            ps_loss = -dy.sqrt(dy.squared_distance(MLVec1, MLVec2))
                            term = -dy.log(dy.logistic(-ps_loss))
                            loss_values.append(term)

        err_value = 0
        if len(loss_values) > 0:
            err = dy.esum(loss_values) / len(loss_values)
            err.forward()
            err_value = err.value()
            err.backward()
            self.trainer.update()
        dy.renew_cg()
        return err_value

    def save(self, path):
        with open(path, 'w') as paramsfp:
            deep_lstm_params = []
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                d_par = dict()
                for j in range(len(params)):
                    d_par[j] = params[j].expr().npvalue()
                deep_lstm_params.append(d_par)

            char_lstm_params = dict()
            for lang in self.char_lstm.keys():
                char_lstm_params[lang] = []
                for i in range(len(self.char_lstm[lang].builder_layers)):
                    builder = self.char_lstm[lang].builder_layers[i]
                    params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                    d_par = dict()
                    for j in range(len(params)):
                        d_par[j] = params[j].expr().npvalue()
                    char_lstm_params[lang].append(d_par)

            proj_mat_params = dict()
            for lang in self.proj_mat.keys():
                proj_mat_params[lang] = self.proj_mat[lang].expr().npvalue()

            clookup_params = dict()
            for lang in self.clookup.keys():
                clookup_params[lang] = self.clookup[lang].expr().npvalue()

            plookup_params = self.plookup.expr().npvalue()
            lang_lookup_params = self.lang_lookup.expr().npvalue()

            arc_mlp_head_params = self.arc_mlp_head.expr().npvalue()
            arc_mlp_head_b_params = self.arc_mlp_head_b.expr().npvalue()
            label_mlp_head_params = self.label_mlp_head.expr().npvalue()
            label_mlp_head_b_params = self.label_mlp_head_b.expr().npvalue()
            arc_mlp_dep_params = self.arc_mlp_dep.expr().npvalue()
            arc_mlp_dep_b_params = self.arc_mlp_dep_b.expr().npvalue()
            label_mlp_dep_params = self.label_mlp_dep.expr().npvalue()
            label_mlp_dep_b_params = self.label_mlp_dep_b.expr().npvalue()
            w_arc_params = self.w_arc.expr().npvalue()
            u_label_params = self.u_label.expr().npvalue()
            pickle.dump((self.lang2id, deep_lstm_params, char_lstm_params, clookup_params,
                         proj_mat_params, plookup_params, lang_lookup_params, arc_mlp_head_params,
                         arc_mlp_head_b_params, label_mlp_head_params, label_mlp_head_b_params, arc_mlp_dep_params,
                         arc_mlp_dep_params, arc_mlp_dep_b_params, arc_mlp_dep_b_params, label_mlp_dep_params,
                         label_mlp_dep_b_params, w_arc_params, u_label_params), paramsfp)

    # def load(self, path):
    #     with open(path, 'r') as paramsfp:
    #         return pickle.load(paramsfp)