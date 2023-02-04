#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : Gaussian Bayes Classifier implementation
#=======================================================

import numpy as np
# from scipy.stats import multivariate_normal as mvn

# calculating the ML estimate
def calc_ML_estimates(x,y,n_classes, mode = 'train'):
    mu = []
    sigma = []
    sigma_inv = []
    sigma_det = []
    threshold = 0.0000001
    for i in set(y):
        if mode == 'train':
            idx = np.where(y == i)
            mu_i = np.mean(x[idx], axis=0)
            sigma_i = np.cov(x[idx], rowvar=False)*(1/2)
        else:
            idx = np.where(y == i)
            if len(idx) == 1:
                mu_i = x[idx[0][0]]
                sigma_i = np.cov(x[[idx[0][0],idx[0][0]]], rowvar=False)
            else:
                mu_i = np.mean(x[idx], axis=0)
                sigma_i = np.cov(x[idx], rowvar=False)
        mu.append(mu_i)
        if np.linalg.det(sigma_i) < 0.00001:
            w, v = np.linalg.eig(sigma_i)
            sigma_det.append(np.product(np.real(w[w > threshold])))
            sigma_i = sigma_i + 0.0001*np.eye(len(mu_i))
            sigma.append(sigma_i)
            sigma_inv.append(np.linalg.inv(sigma_i))
        else:
            sigma.append(sigma_i)
            sigma_det.append(np.linalg.det(sigma_i))
            sigma_inv.append(np.linalg.inv(sigma_i))

    return mu, sigma, sigma_det, sigma_inv

def find_label(test_sample, mu, sigma, sigma_det, sigma_inv, n):
    likelihood = np.zeros(n)
    for i in range(n):
        likelihood[i] = -0.5*(np.log(sigma_det[i]) + np.dot(np.dot(test_sample-mu[i], sigma_inv[i]), test_sample-mu[i]) + (len(mu[i])*np.log(2*np.pi)))
        # likelihood[i] = mvn.logpdf(test_sample, mean=mu[i], cov=sigma[i])
    return np.argmax(likelihood) + 1

def bayes_classifier(x_train,y_train, x_test, n_classes):
    mu, sigma, sigma_det, sigma_inv = calc_ML_estimates(x_train, y_train, n_classes)
    pred = []
    for i in range(x_test.shape[0]):
        l = find_label(x_test[i], mu, sigma, sigma_det, sigma_inv, n_classes)
        pred.append(l)
    return pred
    
