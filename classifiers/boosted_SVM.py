#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : Boosted SVM implementation
#=======================================================

import numpy as np
import random
from classifiers.kernel_SVM import *


def linear_SVM(x_train, y_train):
    P = cal_P_matrix(x_train,y_train,0, 3)
    Q = -np.ones(x_train.shape[0])
    G = -1*np.eye(x_train.shape[0])
    H = np.zeros((x_train.shape[0],1))
    # solve the quadratic programming problem
    mu = cvxopt_solve_qp(P,Q,G,H)
    non_zero = np.nonzero(mu)
    f_train = 0
    # calculate theta_0
    for k in range(x_train.shape[0]):
        f_train += mu[k]*y_train[k]*kernel(x_train[non_zero[0][0]], x_train[k],0,3)
    theta_0 = y_train[non_zero[0][0]] - f_train

    # calculate theta
    theta = np.zeros((x_train.shape[1],1))
    for i in range(x_train.shape[0]):
        theta += (mu[i]*y_train[i]*x_train[i]).reshape(504,1)

    return theta_0, theta

def boosted_SVM(x_train,x_test, y_train, K):
    w = np.ones((x_train.shape[0],1))
    F = np.zeros((x_test.shape[0], 1))
    for i in range(K):
        idx = random.sample(range(0, 300), 50)
        x = x_train[idx]
        y = y_train[idx]
        # find the linear SVM
        theta_0, theta = linear_SVM(x, y)
        # calculate phi
        phi = np.sign(theta_0 + np.matmul(x_train, theta))
        # calculate P
        P = w/w.sum(axis=0)
        # calculate epsilon
        epsilon = np.matmul(P.T, (y_train.reshape(300,1)!=phi))
        if epsilon >= 0.5:
            continue
        # find a 
        a = 0.5*np.log((1-epsilon)/epsilon)
        for i in range(x_train.shape[0]):
            w[i] = w[i]*np.exp(-a*y_train[i]*phi[i])
        F += a*np.sign(theta_0 + np.matmul(x_test, theta))
    
    return np.sign(F)