#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : MDA implementation
#=======================================================

from classifiers.bayes import *

def MDA(x,y, n_classes, mode = 'train'):
    prior = 1/n_classes
    features = x.shape[1]
    # calculate mu and sigma for each class
    mu,sigma,_,_ = calc_ML_estimates(x,y, n_classes, mode)
    mu = np.array(mu)
    # calculate the anchor mean
    anchor_mean = np.sum(prior*mu,axis = 0).reshape((1,features))
    sigma_b = np.zeros((features,features))
    sigma_w = np.zeros((features,features))
    # calculate sigma_b and sigma_W
    for i in range(n_classes):
        sigma_b += prior*np.matmul((mu[i]-anchor_mean).T, mu[i]- anchor_mean)
        sigma_w += prior*sigma[i]
    
    sigma_w += 0.0001*np.eye(features)
    eig_val, eig_vec = np.linalg.eig(np.matmul(np.linalg.inv(sigma_w), sigma_b))
    idx = np.argsort(np.real(eig_val))[::-1]
    sorted_eigenvectors = eig_vec[:,idx]
    non_zero = np.count_nonzero(np.real(eig_val)>1e-10)
    A = sorted_eigenvectors[:,0:non_zero]
    theta = (1/features)*A
    z = (np.matmul(theta.T, x.T)).T
    x_mda = np.matmul(z, theta.T)

    return np.real(x_mda)


