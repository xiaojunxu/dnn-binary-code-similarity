import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from exp_utils import *
import os
import argparse
import cPickle as pickle

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='0,1,2,3,4,5,6,7', help='visible gpu device')
parser.add_argument('--fea_dim', type=int, default=7, help='feature dimension')
parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension')
parser.add_argument('--embed_depth', type=int, default=2, help='embedding network depth')
parser.add_argument('--output_dim', type=int, default=64, help='output layer dimension')
parser.add_argument('--iter_level', type=int, default=5, help='iteration times')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--load_path', type=str, default=None, help='path for model loading, "#LATEST#" for restoring from the latest checkpoint')
parser.add_argument('--save_path', type=str, default='./saved_model/graphnn-model', help='path for model saving')
parser.add_argument('--log_path', type=str, default=None, help='path for training log')




if __name__ == '__main__':
    args = parser.parse_args()
    args.dtype = tf.float32
    print("=================================")
    print(args)
    print("=================================")

    os.environ["CUDA_VISIBLE_DEVICES"]=args.device
    Dtype = args.dtype
    NODE_FEATURE_DIM = args.fea_dim
    EMBED_DIM = args.embed_dim
    EMBED_DEPTH = args.embed_depth
    OUTPUT_DIM = args.output_dim
    ITERATION_LEVEL = args.iter_level
    LEARNING_RATE = args.lr
    MAX_EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    LOAD_PATH = args.load_path
    SAVE_PATH = args.save_path
    LOG_PATH = args.log_path

    SHOW_FREQ = 1
    TEST_FREQ = 1
    SAVE_FREQ = 5
    DATA_FILE_NAME = './acfgSSL_{}/'.format(NODE_FEATURE_DIM)
    SOFTWARE=('openssl-1.0.1f-', 'openssl-1.0.1u-')
    OPTIMIZATION=('-O0', '-O1','-O2','-O3')
    COMPILER=('armeb-linux', 'i586-linux', 'mips-linux')
    VERSION=('v54',)

    FUNC_NAME_DICT = {}

    ##  Processing input begin  ##
    F_NAME = get_f_name(DATA_FILE_NAME, SOFTWARE, COMPILER, OPTIMIZATION, VERSION)
    FUNC_NUM, FUNC_NAME_DICT = get_f_dict(F_NAME)
     

    Gs, classes = read_graph(F_NAME, FUNC_NAME_DICT, NODE_FEATURE_DIM)
    print "{} graphs, {} functions".format(len(Gs), len(classes))


    #Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = partition_data(Gs, classes, [0.8,0.1,0.1], np.load("class_perm.npy"))
    Gs_train, classes_train, Gs_dev, classes_dev, Gs_test, classes_test = partition_data(Gs, classes, [0.05,0.05,0.9], np.load("class_perm.npy"))

    print "Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train))
    print "Dev: {} graphs, {} functions".format(len(Gs_dev), len(classes_dev))
    print "Test2: {} graphs, {} functions".format(len(Gs_test), len(classes_test))

    #Processing the pairs for validation and testing
    try:
        valid_ids = pickle.load(open('valid.ids'))
        valid_epoch = generate_epoch_pair(Gs_dev, classes_dev, BATCH_SIZE, load_id=valid_ids)
    except:
        valid_epoch, valid_ids = generate_epoch_pair(Gs_dev, classes_dev, BATCH_SIZE, output_id=True)
        pickle.dump(valid_ids, open('valid.ids', 'w'))
    try:
        test_ids = pickle.load(open('test.ids'))
        test_epoch = generate_epoch_pair(Gs_test, classes_test, BATCH_SIZE, load_id=test_ids)
    except:
        test_epoch, test_ids = generate_epoch_pair(Gs_test, classes_test, BATCH_SIZE, output_id=True)
        pickle.dump(test_ids, open('test.ids', 'w'))


    ##  Processing input end  ##

    gnn = graphnn(
            N_x = NODE_FEATURE_DIM,
            Dtype = Dtype, 
            N_embed = EMBED_DIM,
            depth_embed = EMBED_DEPTH,
            N_o = OUTPUT_DIM,
            ITERATION_LEVEL = ITERATION_LEVEL,
            lr = LEARNING_RATE
        )

    gnn.init(LOAD_PATH, LOG_PATH)



    #Training:

    l = get_loss_epoch(gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say( "Testing model: validation loss = {0} @ {1}".format(l, datetime.now()) )
    auc0, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
    gnn.say( "Testing model: validation auc = {0} @ {1}".format(auc0, datetime.now()) )


    best_auc = 0
    for i in range(1, MAX_EPOCH+1):
        train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)

        if (i % SHOW_FREQ == 0):
            l = get_loss_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
            gnn.say( "EPOCH {3}/{0}, loss = {1} @ {2}".format(MAX_EPOCH, l, datetime.now(), i) )

        if (i % TEST_FREQ == 0):
            gnn.say( "Testing model: validation loss = {0} @ {1}".format(l, datetime.now()) )
            auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_dev, classes_dev, BATCH_SIZE, load_data=valid_epoch)
            gnn.say( "Testing model: validation auc = {0} @ {1}".format(auc, datetime.now()) )

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
            

    # Test
    test_auc, fpr, tpr, thres = get_auc_epoch(gnn, Gs_test, classes_test, BATCH_SIZE, load_data=test_epoch)
    gnn.say( "Final auc on testing set: {}".format(test_auc) )
