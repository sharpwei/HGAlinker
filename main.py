from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
import time
import os

import tensorflow as tf
import numpy as np
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics

from datautil import hanloaddata, savelabelresult, drawroc
from mymodel.deep.optimizer import Optimizer
from mymodel.deep.model import HanModel
from mymodel.deep.minibatch import HANEdgeMinibatchIterator
from mymodel.utility import rank_metrics, preprocessing



method = "hanlink"
print(method)
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

# Train on GPU
# os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

np.random.seed(0)


# Functions

def get_accuracy_scores(edges_pos, edges_neg):
    feed_dict.update({placeholders['dropout']: 0})
    rec = sess.run(opt.predictions, feed_dict=feed_dict)

    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    # Predict on test set of edges
    preds = []
    actual = []
    predicted = []
    edge_ind = 0
    for u, v in edges_pos:
        score = sigmoid(rec[u, v])
        preds.append(score)
        assert adj_mats[u,v] == 1, 'Problem 1'

        actual.append(edge_ind)
        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_neg = []
    for u, v in edges_neg:
        score = sigmoid(rec[u, v])
        preds_neg.append(score)
        assert adj_mats[u,v] == 0, 'Problem 0'

        predicted.append((score, edge_ind))
        edge_ind += 1

    preds_all = np.hstack([preds, preds_neg])
    preds_all = np.nan_to_num(preds_all)
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    predicted = list(zip(*sorted(predicted, reverse=True, key=itemgetter(0))))[1]

    roc_sc = metrics.roc_auc_score(labels_all, preds_all)
    aupr_sc = metrics.average_precision_score(labels_all, preds_all)
    apk_sc = rank_metrics.apk(actual, predicted, k=50)

    # drawroc(labels_all, preds_all)

    return roc_sc, aupr_sc, apk_sc,labels_all, preds_all


def construct_placeholders():
    placeholders = {
        'batch': tf.placeholder(tf.int32,shape=(FLAGS.batch_size,2), name='batch'),
        'degrees': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'attn_drop': tf.placeholder(dtype=tf.float32, shape=()),
        'ffd_drop': tf.placeholder(dtype=tf.float32, shape=())
    }

    placeholders.update({
        'adj_mats_%d' %i: tf.placeholder(tf.float32,shape=(1,adj_mats.shape[0],adj_mats.shape[1]))
        for i in range(len(adj_mats_list))})
    placeholders.update({
        'feat_%d' %i: tf.placeholder(tf.float32, shape=(1,nodenum,nodenum))
        for i in range(len(fea_list))
    })

    return placeholders

# Load and preprocess data (This is a dummy toy example!)
val_test_size = 0.05
nodenum = 6311
drudis, upusps, ususus = hanloaddata()

adj = drudis

degrees = np.array(adj.sum(axis=0)).squeeze()

adj_mats = adj.astype(np.float32)

# drudis, upusps, ususus
upusps = upusps.astype(np.float32)
ususus = ususus.astype(np.float32)

adj_mats_list = [adj_mats, upusps, ususus]


feat = np.eye(nodenum, dtype=np.float32)
nonzero_feat,num_feat = feat.shape
fea_list = [feat, feat, feat]

edge_type2dim = adj.shape
edge_type2decoder = 'innerproduct'

# Settings and placeholders
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('neg_sample_size', 1, 'Negative sample size.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 10, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0.1, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('max_margin', 0.1, 'Max margin parameter in hinge loss')
flags.DEFINE_integer('batch_size', 5000, 'minibatch size.')
flags.DEFINE_boolean('bias', True, 'Bias term.')
# Important -- Do not evaluate/print validation performance every iteration as it can take
# substantial amount of time
PRINT_PROGRESS_EVERY = 150

print("Defining placeholders")

placeholders = construct_placeholders()


# Create minibatch iterator, model and optimizer

print("Create minibatch iterator")
minibatch = HANEdgeMinibatchIterator(
    adj_mats=adj_mats_list,
    feat=fea_list,
    batch_size=FLAGS.batch_size,
    val_test_size=val_test_size
)

print("Create model")
model = HanModel(
    placeholders=placeholders,
    num_feat=num_feat,
    nonzero_feat=nonzero_feat,
    decoders=edge_type2decoder,
    matenum = 3
)


print("Create optimizer")
with tf.name_scope('optimizer'):
    opt = Optimizer(
        embeddings=model.embeddings,
        latent_inters=model.latent_inters,
        latent_varies=model.latent_varies,
        degrees=degrees,
        edge_type2dim=edge_type2dim,
        placeholders=placeholders,
        batch_size=FLAGS.batch_size,
        margin=FLAGS.max_margin
    )

print("Initialize session")
sess = tf.Session()
sess.run(tf.global_variables_initializer())
feed_dict = {}

# Train model
print("Train model")
for epoch in range(FLAGS.epochs):

    minibatch.shuffle()
    itr = 0
    # print("Construct feed dictionary")
    while not minibatch.end():
        # print("feed"+itr)
        # Construct feed dictionary
        feed_dict = minibatch.next_minibatch_feed_dict(placeholders=placeholders)
        feed_dict = minibatch.update_feed_dict(
            feed_dict=feed_dict,
            dropout=FLAGS.dropout,
            placeholders=placeholders)

        t = time.time()

        # Training step: run single weight update
        outs = sess.run([opt.opt_op, opt.cost,opt.embeddings], feed_dict=feed_dict)
        train_cost = outs[1]
        embeddings = outs[2]
        # batch_edge_type = outs[2]

        if itr % PRINT_PROGRESS_EVERY == 0:
            val_auc, val_auprc, val_apk,_,_ = get_accuracy_scores(
                minibatch.val_edges, minibatch.val_edges_false)

            print("Epoch:", "%04d" % (epoch + 1), "Iter:", "%04d" % (itr + 1),
                  "train_loss=", "{:.5f}".format(train_cost),
                  "val_roc=", "{:.5f}".format(val_auc), "val_auprc=", "{:.5f}".format(val_auprc),
                  "val_apk=", "{:.5f}".format(val_apk), "time=", "{:.5f}".format(time.time() - t))
        itr += 1

print("Optimization finished!")

roc_score, auprc_score, apk_score,labels_all, preds_all = get_accuracy_scores(
    minibatch.test_edges, minibatch.test_edges_false)

print("Test AUROC score", "{:.5f}".format(roc_score))
print("Test AUPRC score", "{:.5f}".format(auprc_score))
print("Test AP@k score", "{:.5f}".format(apk_score))
print()
# method = "hanembdim16"
drawroc(labels_all, preds_all, method)
savelabelresult(labels_all, preds_all, embeddings, method)

