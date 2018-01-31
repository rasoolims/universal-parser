import dynet as dy
from utils import read_conll, write_conll, ConllEntry
from collections import defaultdict
from operator import itemgetter
import time, random, decoder, gzip
import numpy as np
import codecs, os, sys
from linalg import *

class MSTParserLSTM:
    def __init__(self, pos, rels, options, chars, lang2id, deep_lstm_params, char_lstm_params, clookup_params,
                 proj_mat_params, plookup_params, lang_lookup_params, net_options):
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
        edim = net_options.we
        words = defaultdict(set)
        if options.conll_train:
            with open(options.conll_train, 'r') as conllFP:
                for sentence in read_conll(conllFP):
                    for node in sentence:
                        if isinstance(node, ConllEntry):
                            words[node.lang_id].add(node.form)
        if options.conll_dev:
            with open(options.conll_dev, 'r') as conllFP:
                for sentence in read_conll(conllFP):
                    for node in sentence:
                        if isinstance(node, ConllEntry):
                            words[node.lang_id].add(node.form)
        if options.conll_test:
            with open(options.conll_test, 'r') as conllFP:
                for sentence in read_conll(conllFP):
                    for node in sentence:
                        if isinstance(node, ConllEntry):
                            words[node.lang_id].add(node.form)

        if not options.no_init:
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, net_options.pe), init = dy.NumpyInitializer(plookup_params))
        else:
            self.plookup = self.model.add_lookup_parameters((len(pos) + 2, net_options.pe))

        if not options.tune_net:
            self.plookup.set_updated(False)
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

            if not options.no_init:
                self.clookup[lang] = self.model.add_lookup_parameters((len(chars[lang]) + 2, net_options.ce), init=dy.NumpyInitializer(clookup_params[lang]))
            else:
                self.clookup[lang] = self.model.add_lookup_parameters((len(chars[lang]) + 2, net_options.ce))

            if not options.tune_net: self.clookup[lang].set_updated(False)

            self.char_lstm[lang] = dy.BiRNNBuilder(1, net_options.ce, edim, self.model, dy.VanillaLSTMBuilder)
            if not options.no_init:
                for i in range(len(self.char_lstm[lang].builder_layers)):
                    builder = self.char_lstm[lang].builder_layers[i]
                    params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                    for j in range(len(params)):
                        params[j].set_value(char_lstm_params[lang][i][j])
                        if not options.tune_net: params[j].set_updated(False)

            if not options.no_init:
                self.proj_mat[lang] = self.model.add_parameters((edim + net_options.pe, edim + net_options.pe), init=dy.NumpyInitializer(proj_mat_params[lang]))
            else:
                self.proj_mat[lang] = self.model.add_parameters((edim + net_options.pe, edim + net_options.pe))

            if not options.tune_net: self.proj_mat[lang].set_updated(False)

        self.elookup = self.model.add_lookup_parameters((word_index, edim))
        self.num_all_words = word_index
        self.elookup.set_updated(False)
        self.elookup.init_row(0, [0] * edim)
        self.elookup.init_row(1, [0] * edim)
        for lang in self.evocab.keys():
            for word in external_embedding[lang].keys():
                self.elookup.init_row(self.evocab[lang][word], external_embedding[lang][word])

        if not options.no_init:
            self.lang_lookup = self.model.add_lookup_parameters((len(lang2id), net_options.le), init=dy.NumpyInitializer(lang_lookup_params))
        else:
            self.lang_lookup = self.model.add_lookup_parameters((len(lang2id), net_options.le))
        self.lang2id = lang2id
        self.lang_lookup_syntax = self.model.add_lookup_parameters((len(lang2id), net_options.le))

        input_dim = edim + net_options.pe if self.options.use_pos else edim

        self.deep_lstms = dy.BiRNNBuilder(net_options.layer, input_dim +  net_options.le, net_options.rnn * 2, self.model, dy.VanillaLSTMBuilder)
        if not options.no_init:
            for i in range(len(self.deep_lstms.builder_layers)):
                builder = self.deep_lstms.builder_layers[i]
                params = builder[0].get_parameters()[0] + builder[1].get_parameters()[0]
                for j in range(len(params)):
                    params[j].set_value(deep_lstm_params[i][j])
                    if not options.tune_net: params[j].set_updated(False)

        w_mlp_arc = orthonormal_initializer(options.arc_mlp, net_options.le + options.rnn * 2)
        w_mlp_label = orthonormal_initializer(options.label_mlp, net_options.le + options.rnn * 2)
        self.arc_mlp_head = self.model.add_parameters((options.arc_mlp, net_options.le + options.rnn * 2), init= dy.NumpyInitializer(w_mlp_arc))
        self.arc_mlp_head_b = self.model.add_parameters((options.arc_mlp,), init = dy.ConstInitializer(0))
        self.label_mlp_head = self.model.add_parameters((options.label_mlp, net_options.le + options.rnn * 2), init= dy.NumpyInitializer(w_mlp_label))
        self.label_mlp_head_b = self.model.add_parameters((options.label_mlp,), init = dy.ConstInitializer(0))
        self.arc_mlp_dep = self.model.add_parameters((options.arc_mlp, net_options.le +options.rnn * 2), init= dy.NumpyInitializer(w_mlp_arc))
        self.arc_mlp_dep_b = self.model.add_parameters((options.arc_mlp,), init = dy.ConstInitializer(0))
        self.label_mlp_dep = self.model.add_parameters((options.label_mlp, net_options.le + options.rnn * 2), init= dy.NumpyInitializer(w_mlp_label))
        self.label_mlp_dep_b = self.model.add_parameters((options.label_mlp,), init = dy.ConstInitializer(0))
        self.w_arc = self.model.add_parameters((options.arc_mlp, options.arc_mlp+1), init = dy.ConstInitializer(0))
        self.u_label = self.model.add_parameters((len(self.irels) * (options.label_mlp+1), options.label_mlp+1), init = dy.ConstInitializer(0))

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

    def save(self, filename):
        self.model.save(filename)

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
            inputs = [ dy.concatenate([f, b]) for f, b in zip(fs, reversed(bs))]
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
            syntax_lang_embeds = [dy.lookup_batch(self.lang_lookup_syntax, [self.lang2id[lang]]*len(pos_tags[lang][i])) for i in range(len(pos_tags[lang]))]

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
        print '*', len(h_out), h_out[0].dim(), len(syntax_lang_embeds), syntax_lang_embeds[0].dim()
        h_out = [dy.concatenate([syntax_lang_embeds[i], h_out[i]]) for i in range(len(h_out))]

        h =  dy.dropout_dim(dy.concatenate_cols(h_out), 1, d) if train else dy.concatenate_cols(h_out)
        H = self.activation(dy.affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
        M = self.activation(dy.affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
        HL = self.activation(dy.affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
        ML = self.activation(dy.affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))
        print h.dim(), H.dim(), M.dim(), HL.dim(), ML.dim()
        if train:
            H, M, HL, ML =  dy.dropout_dim(H, 1, d),  dy.dropout_dim(M, 1, d),  dy.dropout_dim(HL, 1, d),  dy.dropout_dim(ML, 1, d)
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
            # ratio = min(0.9999, float(t) / (9 + t))
            # self.moving_avg(ratio, 1 - ratio)
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