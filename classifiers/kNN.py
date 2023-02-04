#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : kNN implementation
#=======================================================

import numpy as np

def calc_euclidean_dist(x_train, x_test):
    x_train_sqr=np.square(x_train)
    x_test_sqr=np.square(x_test)
    sumdis_test = np.sum(x_test_sqr, axis=1)
    sumdis_train = np.sum(x_train_sqr, axis=1)
    dot_product=np.dot(x_test, np.transpose(x_train))
    dists=np.sqrt(sumdis_train.reshape(1,-1)-2*dot_product+sumdis_test.reshape(-1,1))

    return dists

def predict_labels(dists, y_train, k):
    pred = np.zeros(dists.shape[0])
    for i in range(dists.shape[0]):
        sorted_indices = np.argsort(dists[i])
        closest_y = y_train[sorted_indices[:k]]
        most_common = np.bincount(closest_y)
        pred[i] = np.argmax(most_common)

    return pred