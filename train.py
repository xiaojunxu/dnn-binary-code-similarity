import tensorflow as tf
print tf.__version__
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from graphnnSiamese import graphnn
from exp_utils import *
import os
import argparse

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
parser.add_argument('--add_data_time', type=int, default=0, help='how many added data will be included into training')




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
    ADD_TIME = args.add_data_time

    SHOW_FREQ = 1
    SAVE_FREQ = MAX_EPOCH
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

    Gs_train, classes_train = Gs, classes

    print "Train: {} graphs, {} functions".format(len(Gs_train), len(classes_train))


    #Deal with the added data for retraining
    add_pairs = []
    for i in range(1, ADD_TIME+1):
        new_pairs = read_added_pair('./added_data/added_data{}'.format(i), NODE_FEATURE_DIM)
        add_pairs += new_pairs

    print "Added data: {} pairs".format(len(add_pairs))

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

    for i in range(1, MAX_EPOCH+1):
        train_epoch(gnn, Gs_train, classes_train, BATCH_SIZE, add_pairs)

        if (i % SHOW_FREQ == 0):
            l = get_loss_epoch(gnn, Gs_train, classes_train, BATCH_SIZE)
            gnn.say( "EPOCH {3}/{0}, loss = {1} @ {2}".format(MAX_EPOCH, l, datetime.now(), i) )

        if (i % SAVE_FREQ == 0):
            path = gnn.save(SAVE_PATH, i)
            gnn.say("Model saved in {}".format(path))
            

    gnn.say( "Training done." )
