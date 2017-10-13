from optparse import OptionParser
import pickle, utils, mstlstm, os, os.path, time


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/en-universal-train.conll.ptb")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/en-universal-dev.conll.ptb")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/en-universal-test.conll.ptb")
    parser.add_option("--output", dest="conll_output",  metavar="FILE", default=None)
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="parser.model")
    parser.add_option("--we", type="int", dest="we", default=100)
    parser.add_option("--batch", type="int", dest="batch", default=5000)
    parser.add_option("--pe", type="int", dest="pe", default=100)
    parser.add_option("--re", type="int", dest="re", default=25)
    parser.add_option("--t", type="int", dest="t", default=50000)
    parser.add_option("--arc_mlp", type="int", dest="arc_mlp", default=500)
    parser.add_option("--label_mlp", type="int", dest="label_mlp", default=100)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=200)
    parser.add_option("--lr", type="float", dest="lr", default=0.002)
    parser.add_option("--beta1", type="float", dest="beta1", default=0.9)
    parser.add_option("--beta2", type="float", dest="beta2", default=0.9)
    parser.add_option("--dropout", type="float", dest="dropout", default=0.33)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="leaky")
    parser.add_option("--layer", type="int", dest="layer", default=3)
    parser.add_option("--rnn", type="int", dest="rnn", help='dimension of rnn in each direction', default=400)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--no_anneal", action="store_false", dest="anneal", default=True)
    parser.add_option("--dynet-mem", type="int", dest="mem", default=0)
    parser.add_option("--dynet-autobatch", type="int", dest="dynet-autobatch", default=0)

    (options, args) = parser.parse_args()
    print options
    print 'Using external embedding:', options.external_embedding
    if options.predictFlag:
        with open(options.params, 'r') as paramsfp:
            w2i, pos, rels, stored_opt = pickle.load(paramsfp)
        stored_opt.external_embedding = options.external_embedding
        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(pos, rels, w2i, stored_opt)
        parser.Load(options.model)
        ts = time.time()
        test_res = list(parser.Predict(options.conll_test, False, True))
        te = time.time()
        print 'Finished predicting test.', te-ts, 'seconds.'
        utils.write_conll(options.conll_output, test_res)
    else:
        print 'Preparing vocab'
        w2i, pos, rels = utils.vocab(options.conll_train)

        with open(os.path.join(options.output, options.params), 'w') as paramsfp:
            pickle.dump((w2i, pos, rels, options), paramsfp)
        print 'Finished collecting vocab'

        print 'Initializing lstm mstparser:'
        parser = mstlstm.MSTParserLSTM(pos, rels, w2i, options)
        best_acc = -float('inf')
        t, epoch = 0,1
        while t<=options.t:
            print 'Starting epoch', epoch
            t, epoch = parser.Train(options.conll_train, t), epoch+1
            devpath = os.path.join(options.output, 'dev_epoch_out')
            # utils.write_conll(devpath, parser.Predict(options.conll_dev, True, False))
            # uas,las1 = utils.eval(options.conll_dev, devpath)
            # print 'greedy UAS/LAS', uas, las1

            utils.write_conll(devpath, parser.Predict(options.conll_dev, False, False))
            uas, las2 = utils.eval(options.conll_dev, devpath)
            print 'eisner UAS/LAS',  uas, las2

            # utils.write_conll(devpath, parser.Predict(options.conll_dev, False, True))
            # uas, las3 = utils.eval(options.conll_dev, devpath)
            # print 'tarjan UAS/LAS', uas, las3
            # las = max(las1, max(las2, las3))
            las = las2 #todo
            if las > best_acc:
                print 'saving model', las
                best_acc = las
                parser.Save(os.path.join(options.output, os.path.basename(options.model)))

            avg_model = mstlstm.MSTParserLSTM(pos, rels, w2i, options, parser)
            utils.write_conll(devpath, avg_model.Predict(options.conll_dev, False, False))
            uas, las = utils.eval(options.conll_dev, devpath)
            print 'eisner avg UAS/LAS', uas, las

            if las > best_acc:
                print 'saving avg model', las
                best_acc = las
                avg_model.Save(os.path.join(options.output, os.path.basename(options.model)))