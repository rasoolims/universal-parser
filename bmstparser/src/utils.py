from collections import Counter
import re, codecs,sys, random
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

class ConllEntry:
    def __init__(self, id, form, lemma, pos, fpos, feats=None, head=None, relation=None, deps=None, misc=None):
        self.id = id
        self.form = form.lower() # assuming everything is lowercased.
        self.norm = normalize(form)
        self.fpos = fpos.upper()
        self.pos = pos.upper()
        self.head = head
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.pos, self.fpos, self.feats, str(self.pred_parent_id) if self.pred_parent_id is not None else None, self.pred_relation, self.deps, self.misc]
        return '\t'.join(['_' if v is None else v for v in values])

def vocab(conll_path, min_count=2):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()
    langs = set()
    chars = set()
    with open(conll_path, 'r') as conllFP:
        for sentence in read_conll(conllFP):
            wordsCount.update([node.norm for node in sentence if isinstance(node, ConllEntry)])
            posCount.update([node.pos for node in sentence if isinstance(node, ConllEntry)])
            relCount.update([node.relation for node in sentence if isinstance(node, ConllEntry)])
            for node in sentence:
                for c in list(node.norm):
                    chars.add(c.lower())
            langs.add(sentence[1].feats)

    words = set()
    for w in wordsCount.keys():
        if wordsCount[w]>=min_count:
            words.add(w)
    return ({w: i for i, w in enumerate(words)}, list(posCount.keys()), list(relCount.keys()), list(chars), list(sorted(langs)))

def read_conll(fh):
    root = ConllEntry(0, '*root*', '*root*', 'ROOT-POS', 'ROOT-FPOS', '_', 0, 'root', '_', '_')
    tokens = [root]
    for line in fh:
        tok = line.strip().split('\t')
        if not tok or line.strip() == '':
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            if line[0] == '#' or '-' in tok[0] or '.' in tok[0]:
                tokens.append(line.strip())
            else:
                tokens.append(ConllEntry(int(tok[0]), tok[1], tok[2], tok[3], tok[4], tok[5], int(tok[6]) if tok[6] != '_' else -1, tok[7], tok[8], tok[9]))
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
                    if s1[7]==s2[7]:
                        correct_l+=1
    return 100 * float(correct_deps) / all_deps, 100 * float(correct_l) / all_deps

numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")
urlRegex = re.compile("((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

def normalize(word):
    return '<num>' if numberRegex.match(word) else ('<url>' if urlRegex.match(word) else word.lower())

def get_batches(buckets, model, is_train):
    d_copy = [buckets[i][:] for i in range(len(buckets))]
    if is_train:
        for dc in d_copy:
            random.shuffle(dc)
    mini_batches = []
    batch, cur_len, cur_c_len = [], 0, 0
    for dc in d_copy:
        for d in dc:
            if (is_train and len(d)<=100) or not is_train:
                batch.append(d)
                cur_c_len = max(cur_c_len, max([len(w.norm) for w in d]))
                cur_len = max(cur_len, len(d))

            if cur_len * len(batch) >= model.options.batch:
                add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train)
                batch, cur_len, cur_c_len = [], 0, 0

    if len(batch)>0:
        add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train)
        batch, cur_len = [], 0
    if is_train:
        random.shuffle(mini_batches)
    return mini_batches


def add_to_minibatch(batch, cur_c_len, cur_len, mini_batches, model, is_train):
    words = np.array([np.array(
        [model.vocab.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pwords = np.array([np.array(
        [model.evocab.get(batch[i][j].norm, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    pos = np.array([np.array(
        [model.pos.get(batch[i][j].pos, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    langs = np.array([np.array(
        [model.langs.get(batch[i][j].feats, 0) if j < len(batch[i]) else model.PAD for i in
         range(len(batch))]) for j in range(cur_len)])
    heads = np.array(
        [np.array([batch[i][j].head if 0 < j < len(batch[i]) and batch[i][j].head>=0 else 0 for i in range(len(batch))]) for j
         in range(cur_len)])
    relations = np.array([np.array(
        [model.rels.get(batch[i][j].relation, 0) if j < len(batch[i]) else model.PAD_REL for i in
         range(len(batch))]) for j in range(cur_len)])
    chars = [list() for _ in range(cur_c_len)]
    for c_pos in range(cur_c_len):
        ch = [model.PAD] * (len(batch) * cur_len)
        offset = 0
        for w_pos in range(cur_len):
            for sen_position in range(len(batch)):
                if w_pos < len(batch[sen_position]) and c_pos < len(batch[sen_position][w_pos].norm):
                    ch[offset] = model.chars.get(batch[sen_position][w_pos].norm[c_pos], 0)
                offset += 1
        chars[c_pos] = np.array(ch)
    chars = np.array(chars)

    clangs = [list() for _ in range(cur_c_len)]
    for c_pos in range(cur_c_len):
        ch = [model.PAD] * (len(batch) * cur_len)
        offset = 0
        for w_pos in range(cur_len):
            for sen_position in range(len(batch)):
                if w_pos < len(batch[sen_position]) and c_pos < len(batch[sen_position][w_pos].norm):
                    ch[offset] = model.langs.get(batch[sen_position][1].feats, 0)
                offset += 1
        clangs[c_pos] = np.array(ch)
    clangs = np.array(clangs)

    masks = np.array([np.array([1 if 0 < j < len(batch[i]) and (batch[i][j].head>=0 or not is_train) else 0 for i in range(len(batch))]) for j in
                      range(cur_len)])
    mini_batches.append((words, pwords, pos, heads, relations, chars,langs,clangs, masks))


def is_punc(pos):
	return  pos=='.' or pos=='PUNC' or pos =='PUNCT' or \
        pos=="#" or pos=="''" or pos=="(" or \
		pos=="[" or pos=="]" or pos=="{" or pos=="}" or \
		pos=="\"" or pos=="," or pos=="." or pos==":" or \
		pos=="``" or pos=="-LRB-" or pos=="-RRB-" or pos=="-LSB-" or \
		pos=="-RSB-" or pos=="-LCB-" or pos=="-RCB-" or pos=='"' or pos==')'

