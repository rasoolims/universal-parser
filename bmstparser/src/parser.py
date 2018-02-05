from optparse import OptionParser
import pickle, utils, mstlstm, sys, os.path, time
from data_loader import Data
from collections import defaultdict
from utils import *

def test(parser, buckets, test_file, output_file):
    results = list()
    for mini_batch in utils.get_batches(buckets, parser, False):
        outputs = parser.build_graph(mini_batch, 1, False)
        for output in outputs:
            results.append(output)

    arcs = reduce(lambda x, y: x + y, [list(result[0]) for result in results])
    rels = reduce(lambda x, y: x + y, [list(result[1]) for result in results])
    idx = 0
    with open(test_file) as f:
        with open(output_file, 'w') as fo:
            for line in f.readlines():
                info = line.strip().split('\t')
                if info and line.strip() != '':
                    assert len(info) == 10, 'Illegal line: %s' % line
                    info[6] = str(arcs[idx])
                    info[7] = parser.irels[rels[idx]]
                    fo.write('\t'.join(info) + '\n')
                    idx += 1
                else:
                    fo.write('\n')
    return utils.eval(test_file, output_file)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default=None)
    parser.add_option("--par", dest="par", help="Paralell data directory", metavar="FILE", default=None)
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default=None)
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default=None)
    parser.add_option("--output", dest="conll_output",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=5000)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--ce", type="int", dest="ce", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--t", type="int", dest="t", default=50000)
    parser.add_option("--arc_mlp", type="int", dest="arc_mlp", default=400)
    parser.add_option("--label_mlp", type="int", dest="label_mlp", default=100)
    parser.add_option("--lr", type="float", dest="lr", default=0.002)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.9)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--net", metavar="FILE", dest="netfile", default=None)
    parser.add_option("--activation", type="string", dest="activation", default="leaky")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--rnn", type="int", dest="rnn", help='dimension of rnn in each direction', default=200)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--eval_non_avg", action="store_true", dest="eval_non_avg", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--no_char", action="store_false", dest="use_char", default=True)
    parser.add_option("--no_pos", action="store_false", dest="use_pos", default=True)
    parser.add_option("--stop", type="int", dest="stop", default=50)
    parser.add_option("--tune_net", action="store_true", dest="tune_net", default=False)
    parser.add_option("--no_init", action="store_true", dest="no_init", default=False)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)
    parser.add_option("--neg_num", type="int", dest="neg_num", help="number of negative example per language", default=5)
    parser.add_option("--par_batch", type="int", dest="par_batch", default=20)
    parser.add_option("--le", type="int", dest="le", help="language embedding", default=25)
    parser.add_option("--lm_iter", type="int", dest="lm_iter", help="number of pretraining iterations for LM", default=200)
    parser.add_option("--joint", action="store_true", dest="joint", default=False)


    (options, args) = parser.parse_args()
    print options
    universal_tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN',
                      'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

    print 'Using external embedding:', options.external_embedding
    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            words, chars, rels, stored_opt = pickle.load(paramsfp)
        words = defaultdict(set)
        with open(options.conll_test, 'r') as conllFP:
            for sentence in read_conll(conllFP):
                for node in sentence:
                    if isinstance(node, ConllEntry):
                        words[node.lang_id].add(node.form)

        stored_opt.external_embedding = options.external_embedding
        print stored_opt
        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(universal_tags, rels, stored_opt, words, chars, options.model)
        ts = time.time()
        print 'loading buckets'
        test_buckets = [list()]
        test_data = list(utils.read_conll(open(options.conll_test, 'r')))
        for d in test_data:
            test_buckets[0].append(d)
        print 'parsing'
        test(parser, test_buckets, options.conll_test, options.conll_output)
        te = time.time()
        print 'Finished predicting test.', te-ts, 'seconds.'
    else:
        words = defaultdict(set)
        if options.joint:
            print 'reading shared model'
            par_data = Data(options.par, universal_tags)
            if par_data:
                for lang in par_data.neg_examples.keys():
                    for word in par_data.neg_examples[lang]:
                        words[lang].add(word)

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

        chars = dict()
        for lang in words.keys():
            ch = set()
            for word in words[lang]:
                for c in word:
                    ch.add(c)
            chars[lang] = sorted(list(ch))

        print 'Preparing vocab'
        rels = utils.vocab(options.conll_train)
        if not os.path.isdir(options.output): os.mkdir(options.output)
        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((words, chars, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(universal_tags, rels, options, words, chars)
        best_acc = -float('inf')
        t, epoch = 0,1
        train_data = list(utils.read_conll(open(options.conll_train, 'r')))
        max_len = max([len(d) for d in train_data])
        min_len = min([len(d) for d in train_data])
        buckets = [list() for i in range(min_len, max_len)]
        for d in train_data:
            buckets[len(d)-min_len-1].append(d)
        buckets = [x for x in buckets if x != []]
        if options.conll_dev:
            dev_buckets = [list()]
            dev_data = list(utils.read_conll(open(options.conll_dev, 'r')))
            for d in dev_data:
                dev_buckets[0].append(d)
        best_las = 0
        no_improvement = 0
        errors = []
        while t<=options.t:
            print 'Starting epoch', epoch, 'time:', time.ctime()
            mini_batches = utils.get_batches(buckets, parser, True)
            start, closs = time.time(), 0
            for i, minibatch in enumerate(mini_batches):
                t, loss = parser.build_graph(minibatch, t, True)
                if options.joint:
                    mb = par_data.get_next_batch(parser, options.par_batch, options.neg_num)
                    errors.append(parser.train_shared_rnn(mb, True)) #todo
                if len(errors) >= 100:
                    print '%, loss', sum(errors) / len(errors)
                    errors = []

                if parser.options.anneal:
                    decay_steps = min(1.0, float(t) / 50000)
                    lr = parser.options.lr * 0.75 ** decay_steps
                    parser.trainer.learning_rate = lr
                closs += loss
                if t%10==0:
                    sys.stdout.write('overall progress:' + str(round(100 * float(t) / options.t, 2)) + '% current progress:' + str(round(100 * float(i + 1) / len(mini_batches), 2)) + '% loss=' + str(closs / 10) + ' time: ' + str(time.time() - start) + '\n')
                    if t%100==0 and options.conll_dev:
                        uas, las = test(parser, dev_buckets, options.conll_dev, options.output + '/dev.out')
                        print 'dev non-avg acc', las, uas
                        if las > best_las:
                            best_las = las
                            print 'saving non-avg with', best_las, uas
                            parser.save(options.output + '/model')
                            no_improvement = 0
                        else:
                            no_improvement += 1
                    start, closs = time.time(), 0

            if no_improvement>options.stop:
                print 'No improvements after',no_improvement, 'steps -> terminating.'
                sys.exit(0)
            print 'current learning rate', parser.trainer.learning_rate, 't:', t
            epoch+=1

        if not options.conll_dev:
            print 'saving non-avg without tuning'
            parser.save(options.output + '/model')
