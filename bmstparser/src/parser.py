from optparse import OptionParser
import pickle, utils, mstlstm, sys, os.path, time
from functools import reduce


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
    parser.add_option("--epoch", type="int", dest="epoch", default=1000)
    parser.add_option("--arc_mlp", type="int", dest="arc_mlp", default=400)
    parser.add_option("--label_mlp", type="int", dest="label_mlp", default=100)
    parser.add_option("--lr", type="float", dest="lr", default=0.002)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.9)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="leaky")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--rnn", type="int", dest="rnn", help='dimension of rnn in each direction', default=200)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--eval_non_avg", action="store_true", dest="eval_non_avg", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--no_char", action="store_false", dest="use_char", default=True)
    parser.add_option("--no_pos", action="store_false", dest="use_pos", default=True)
    parser.add_option("--stop", type="int", dest="stop", default=50)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)
    parser.add_option("--dynet-gpus", action="store_true", dest="dynet-gpus", default=False, help='Use GPU instead of cpu.')

    (options, args) = parser.parse_args()
    print(options)
    print('Using external embedding:', options.external_embedding)
    if options.predictFlag:
        with open(options.params, 'rb') as paramsfp:
            w2i, pos, rels, chars, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        print(stored_opt)
        print('Initializing lstm mstparser:')
        parser = mstlstm.MSTParserLSTM(pos, rels, w2i, chars, stored_opt)
        parser.Load(options.model)
        ts = time.time()
        print('loading buckets')
        test_buckets = [list()]
        test_data = list(utils.read_conll(open(options.conll_test, 'r')))
        for d in test_data:
            test_buckets[0].append(d)
        print('parsing')
        test(parser, test_buckets, options.conll_test, options.conll_output)
        te = time.time()
        print('Finished predicting test.', te-ts, 'seconds.')
    else:
        print('Preparing vocab')
        w2i, pos, rels, chars = utils.vocab(options.conll_train)
        if not os.path.isdir(options.output): os.mkdir(options.output)
        with open(os.path.join(options.output, options.params), 'wb') as paramsfp:
            pickle.dump((w2i, pos, rels, chars, options), paramsfp)
        print('Finished collecting vocab')

        print('Initializing lstm mstparser:')
        parser = mstlstm.MSTParserLSTM(pos, rels, w2i, chars, options)
        best_acc = -float('inf')
        t, epoch = 0,1
        train_data = list(utils.read_conll(open(options.conll_train, 'r')))
        max_len = max([len(d) for d in train_data])
        min_len = min([len(d) for d in train_data])
        buckets = [list() for i in range(min_len, max_len+1)]
        for d in train_data:
            buckets[min(0,len(d)-min_len-1)].append(d)
        buckets = [x for x in buckets if x != []]
        dev_buckets = [list()]
        if options.conll_dev:
            dev_data = list(utils.read_conll(open(options.conll_dev, 'r')))
            for d in dev_data:
                dev_buckets[0].append(d)
        best_las = 0
        no_improvement = 0
        while t<=options.t and epoch <= options.epoch:
            print('Starting epoch', epoch, 'time:', time.ctime())
            mini_batches = utils.get_batches(buckets, parser, True)
            start, closs = time.time(), 0
            for i, minibatch in enumerate(mini_batches):
                t, loss = parser.build_graph(minibatch, t, True)
                if parser.options.anneal:
                    decay_steps = min(1.0, float(t) / 50000)
                    lr = parser.options.lr * 0.75 ** decay_steps
                    parser.trainer.learning_rate = lr
                closs += loss
                if t%10==0:
                    sys.stdout.write('overall progress:' + str(round(100 * float(t) / options.t, 2)) + '% current progress:' + str(round(100 * float(i + 1) / len(mini_batches), 2)) + '% loss=' + str(closs / 10) + ' time: ' + str(time.time() - start) + '\n')
                    if t%100==0 and options.conll_dev:
                        if options.eval_non_avg:
                            uas, las = test(parser, dev_buckets, options.conll_dev, options.output + '/dev.out')
                            print('dev non-avg acc', las, uas)
                            if las > best_las:
                                best_las = las
                                print('saving non-avg with', best_las, uas)
                                parser.Save(options.output + '/model')
                                no_improvement = 0
                            else:
                                no_improvement += 1
                        avg_model = mstlstm.MSTParserLSTM(pos, rels, w2i, chars, options, parser)
                        uas, las = test(avg_model, dev_buckets, options.conll_dev, options.output+'/dev.out')
                        print('dev avg acc', las, uas)
                        if las > best_las:
                            best_las = las
                            print('saving avg with', best_las, uas)
                            avg_model.Save(options.output + '/model')
                            no_improvement = 0
                        else:
                            no_improvement += 1
                    start, closs = time.time(), 0

            if no_improvement>options.stop:
                print('No improvements after',no_improvement, 'steps -> terminating.')
                sys.exit(0)
            print('current learning rate', parser.trainer.learning_rate, 't:', t)
            epoch+=1

        if not options.conll_dev:
            avg_model = mstlstm.MSTParserLSTM(pos, rels, w2i, chars, options, parser)
            print('Saving default model without dev-tuning')
            avg_model.Save(options.output + '/model')


