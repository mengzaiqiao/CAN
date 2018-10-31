from __future__ import division
from __future__ import print_function
import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import  OptimizerCAN
from input_data import load_AN
from model import CAN
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges, mask_test_feas
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# Settings
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_string('dataset', 'facebook', 'Dataset string.')

dataset_str = FLAGS.dataset
weight_decay = FLAGS.weight_decay
# Load data
adj, features = load_AN(dataset_str)
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false = mask_test_feas(features)

adj = adj_train
features_orig = features
features = sp.lil_matrix(features)

link_predic_result_file = "result/AGAE_{}.res".format(dataset_str)
embedding_node_mean_result_file = "result/AGAE_{}_n_mu.emb".format(dataset_str)
embedding_attr_mean_result_file = "result/AGAE_{}_a_mu.emb".format(dataset_str)
embedding_node_var_result_file = "result/AGAE_{}_n_sig.emb".format(dataset_str)
embedding_attr_var_result_file = "result/AGAE_{}_a_sig.emb".format(dataset_str)

adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'features_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]
features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = CAN(placeholders, num_features, num_nodes, features_nonzero)
pos_weight_u = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm_u = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
pos_weight_a = float(features[2][0] * features[2][1] - len(features[1])) / len(features[1])
norm_a = features[2][0] * features[2][1] / float((features[2][0] * features[2][1] - len(features[1])) * 2)
# Optimizer
with tf.name_scope('optimizer'):
    opt = OptimizerCAN(preds=model.reconstructions,
                       labels=(tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'], validate_indices=False), [-1]),
                                tf.reshape(tf.sparse_tensor_to_dense(placeholders['features_orig'], validate_indices=False), [-1])),
                       model=model,
                       num_nodes=num_nodes,
                       num_features=num_features,
                       pos_weight_u=pos_weight_u,
                       norm_u=norm_u,
                       pos_weight_a=pos_weight_a,
                       norm_a=norm_a)
# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = sess.run(model.reconstructions[0], feed_dict=feed_dict).reshape([num_nodes, num_nodes])
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_a(feas_pos, feas_neg):

    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    fea_rec = sess.run(model.reconstructions[1], feed_dict=feed_dict).reshape([num_nodes, num_features])
    preds = []
    pos = []
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)
features_label = sparse_to_tuple(features_orig)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, features_label, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.log_lik, opt.kl], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    log_lik = outs[3]
    kl = outs[4]
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    roc_curr_a, ap_curr_a = get_roc_score_a(val_feas, val_feas_false)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1),
          "train_loss=", "{:.5f}".format(avg_cost),
          "log_lik=", "{:.5f}".format(log_lik),
          "KL=", "{:.5f}".format(kl),
          "train_acc=", "{:.5f}".format(avg_accuracy),
          "val_edge_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_edge_ap=", "{:.5f}".format(ap_curr),
          "val_attr_roc=", "{:.5f}".format(roc_curr_a),
          "val_attr_ap=", "{:.5f}".format(ap_curr_a),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
    
roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
roc_score_a, ap_score_a = get_roc_score_a(test_feas, test_feas_false)

z_u_mean = sess.run(model.z_u_mean, feed_dict=feed_dict)
z_a_mean = sess.run(model.z_a_mean, feed_dict=feed_dict)
np.save(embedding_node_mean_result_file, z_u_mean)
np.save(embedding_attr_mean_result_file, z_a_mean)
z_u_log_std = sess.run(model.z_u_log_std, feed_dict=feed_dict)
z_a_log_std = sess.run(model.z_a_log_std, feed_dict=feed_dict)
np.save(embedding_node_var_result_file, z_u_log_std)
np.save(embedding_attr_var_result_file, z_a_log_std)    
print('Test edge ROC score: ' + str(roc_score))
print('Test edge AP score: ' + str(ap_score))
print('Test attr ROC score: ' + str(roc_score_a))
print('Test attr AP score: ' + str(ap_score_a))
