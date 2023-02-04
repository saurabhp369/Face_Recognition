#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : data utils for task 2
#=======================================================
import numpy as np

def new_data(data):
    m = 2
    idx = []
    for i in range(600):
        if i != m:
            idx.append(i)
        else:
            m = m+3
    return idx

def make_labels():
    labels = []
    for i in range(400):
        if i%2 == 0:
            labels.append(1)
        else:
            labels.append(2)

    return np.array(labels)

def make_labels_SVM():
    labels = []
    for i in range(400):
        if i%2 == 0:
            labels.append(1)
        else:
            labels.append(-1)

    return np.array(labels)