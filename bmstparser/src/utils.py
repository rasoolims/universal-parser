from collections import Counter, defaultdict
import re, codecs, sys, random
import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')


class ConllEntry:
    def __init__(self, id, form, lemma, pos, fpos, lang_id='_', head=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form.lower()  # assuming everything is lowercased.
        self.norm = normalize(form)
        self.pos = pos.upper()
        self.fpos = fpos.upper()
        self.head = head
        self.relation = relation

        self.lemma = lemma
        self.lang_id = lang_id
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.fpos, self.lang_id,
                  str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def vocab(conll_path):
    relCount = Counter()
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])

    return list(relCount.keys())


def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-FPOS', '_', 0, 'root', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens) > 1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5].strip(),
                                         int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
    if len(tokens) > 1:
        yield tokens


def write_conll(fn, conll_gen):
    with codecs.open(fn, 'w', encoding='utf-8') as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + u'\n')
            fh.write('\n')


def eval(gold, predicted):
    correct_deps, correct_l, all_deps = 0, 0, 0
    r2 = open(predicted, 'r')
    for l1 in open(gold, 'r'):
        s1 = l1.strip().split('\t')
        s2 = r2.readline().strip().split('\t')
        if len(s1) > 6:
            if not is_punc(s2[3]):
                all_deps += 1
                if s1[6] == s2[6]:
                    correct_deps += 1
                    if s1[7] == s2[7]:
                        correct_l += 1
    return 100 * float(correct_deps) / all_deps, 100 * float(correct_l) / all_deps


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<num>' if numberRegex.match(word) else ('<url>' if urlRegex.match(word) else word.lower())


def normalize_sent(sent):
    words, tags = get_words_tags(sent)
    return ' '.join(['*root*_ROOT-POS']+[normalize(words[i])+'_'+tags[i] for i in range(len(words))])

def get_words_tags(sent):
    words, tags = [], []
    for sen_t in sent.strip().split():
        r = sen_t.rfind('_')
        words.append(sen_t[:r])
        tags.append(sen_t[r + 1:])
    return words, tags


def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, cur_len, cur_c_len, batch_len = defaultdict(list), 0, 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d) <= 100) or not is_train:
                batch[d[1].lang_id].append(d)
                cur_c_len = max(cur_c_len, max([len(w.norm) for w in d]))
                cur_len = max(cur_len, len(d))
                batch_len += 1

            if cur_len * batch_len >= model.options.batch:
                mini_batches.append(get_minibatch(batch, cur_c_len, cur_len, model))
                batch, cur_len, cur_c_len, batch_len = defaultdict(list), 0, 0, 0

    if len(batch) > 0:
        mini_batches.append(get_minibatch(batch, cur_c_len, cur_len, model))
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def get_minibatch(batch, max_c_len, cur_len, model):
    all_batches, langs = [], []
    for lang_id in batch.keys():
        all_batches += batch[lang_id]
        langs += [lang_id] * len(batch[lang_id])
    chars, words, pwords, pos =dict(), dict(), dict(), dict()
    for lang_id in batch.keys():
        chars_ = [list() for _ in range(max_c_len)]
        for c_pos in range(max_c_len):
            ch = [model.PAD] * (len(batch[lang_id]) * cur_len)
            offset = 0
            for w_pos in range(cur_len):
                for sen_position in range(len(batch[lang_id])):
                    if w_pos < len(batch[lang_id][sen_position]) and c_pos < len(batch[lang_id][sen_position][w_pos].norm):
                        ch[offset] = model.chars[lang_id].get(batch[lang_id][sen_position][w_pos].norm[c_pos], 0)
                    offset += 1
            chars_[c_pos] = np.array(ch)
        chars[lang_id] = np.array(chars_)
        words[lang_id] = np.array([np.array(
            [model.vocab[langs[i]].get(batch[lang_id][i][j].form, 0) if j < len(batch[lang_id][i]) else model.PAD
             for i in range(len(batch[lang_id]))]) for j in range(cur_len)])
        pwords[lang_id] = np.array([np.array(
            [model.evocab[langs[i]].get(batch[lang_id][i][j].form, 0) if j < len(batch[lang_id][i]) else model.PAD
             for i in range(len(batch[lang_id]))]) for j in range(cur_len)])
        pos[lang_id] = np.array([np.array(
            [model.pos.get(batch[lang_id][i][j].pos, 0) if j < len(batch[lang_id][i]) else model.PAD for i in
             range(len(batch[lang_id]))]) for j in range(cur_len)])
    masks = np.array([np.array([1 if 0 < j < len(all_batches[i]) else 0 for i in range(len(all_batches))])
                      for j in range(cur_len)])
    heads = np.array([np.array(
        [all_batches[i][j].head if 0 < j < len(all_batches[i]) and all_batches[i][j].head >= 0 else 0 for i in
         range(len(all_batches))]) for j in range(cur_len)])
    relations = np.array([np.array(
        [model.rels.get(all_batches[i][j].relation, 0) if j < len(all_batches[i]) else model.PAD_REL for i in
         range(len(all_batches))]) for j in range(cur_len)])

    mini_batch = (words, pwords, pos, heads, relations, chars, langs, cur_len, len(all_batches), masks)
    return mini_batch


def is_punc(pos):
    return pos == '.' or pos == 'PUNC' or pos == 'PUNCT' or \
           pos == "#" or pos == "''" or pos == "(" or \
           pos == "[" or pos == "]" or pos == "{" or pos == "}" or \
           pos == "\"" or pos == "," or pos == "." or pos == ":" or \
           pos == "``" or pos == "-LRB-" or pos == "-RRB-" or pos == "-LSB-" or \
           pos == "-RSB-" or pos == "-LCB-" or pos == "-RCB-" or pos == '"' or pos == ')'
