from __future__ import division
from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import *


def multiclass_node_classification_eval(X, y, ratio=0.5, rnd=2018):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=rnd)
    clf = SVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    micro_f1 = f1_score(y_test, y_pred, average="micro")

    return macro_f1, micro_f1


def node_classification_F1(Embeddings, y, ratio):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        macro_f1, micro_f1 = multiclass_node_classification_eval(
            Embeddings, y, ratio, rnd)
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
    macro_f1_avg /= 10
    micro_f1_avg /= 10
    print ("Macro_f1: " + str(macro_f1_avg))
    print ("Micro_f1: " + str(micro_f1_avg))

    
def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    i = 0
    for line in lines:
        l = line.strip("\n\r")
        y[i] = int(l)
        i += 1
    return y


datasets = ['cora' ]#'cora', 'citeseer', 'pubmed', 'pubmed','BlogCatalog']
for datasetname in datasets:
    for ratio in [0.2]:
        print('dataset:', datasetname, ',ratio:', ratio)
        embedding_node_result_file = "result/AGAE_{}_n_mu.emb.npy".format(datasetname)
        label_file = "data/" + datasetname + ".label"
        y = read_label(label_file)
        Embeddings = np.load(embedding_node_result_file)
        node_classification_F1(Embeddings, y, ratio)