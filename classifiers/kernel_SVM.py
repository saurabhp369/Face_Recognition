#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : Kernel SVM implementation
#=======================================================

import numpy as np
import math
import cvxopt

# find kernel
def kernel(x,y,z, choice):
    # sigma = 2
    # r = 2
    if choice == 1:
        k = np.exp((-1/z**2)*np.linalg.norm(x-y))
    elif choice == 2:
        k = math.pow(np.matmul(x,y.T)+1,z)
    else:
        k = np.matmul(x,y.T)

    return k

def cal_P_matrix(x, y_train, z, choice):
    p = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            p[i][j] = y_train[i]*y_train[j]*kernel(x[i], x[j], z, choice)
        
    return p

def cvxopt_solve_qp(P, q, G, h, A=None, b=None):
    P = .5 * (P + P.T)  # make sure P is symmetric
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    if A is not None:
        args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*args)
    if 'optimal' not in sol['status']:
        return None
    return np.array(sol['x']).reshape((P.shape[1],1))

def kernel_SVM(x_train,y_train,x_test, z, choice):
    P = cal_P_matrix(x_train,y_train, z, choice)
    Q = -np.ones(x_train.shape[0])
    G = -1*np.eye(x_train.shape[0])
    H = np.zeros((x_train.shape[0],1))
    mu = cvxopt_solve_qp(P,Q,G,H)
    non_zero = np.nonzero(mu)
    # print(non_zero)

    f_train = 0
    for k in range(x_train.shape[0]):
        f_train += mu[k]*y_train[k]*kernel(x_train[non_zero[0][0]], x_train[k],z, choice)
    theta_0 = y_train[non_zero[0][0]] - f_train

    f = np.zeros((x_test.shape[0],1))
    for i in range(x_test.shape[0]):
        for j in range(x_train.shape[0]):

            f[i] += mu[j]*y_train[j]*kernel(x_train[j], x_test[i], z, choice)

    y_pred = np.sign(theta_0*np.ones((x_test.shape[0], 1)) + f)
    return y_pred

    
