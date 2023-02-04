#=======================================================
# Author : Saurabh Palande
# Package : CMSC828C Project 1
# Module : PCA implementation
#=======================================================

import numpy as np

def PCA(x):
    normalized_face = x - np.mean(x, axis= 0)
    cov = np.cov(normalized_face , rowvar = False)
    eigen_values, eigen_vectors = np.linalg.eig(cov)
    n = 100
    # sort the eigenvalues in decreasing order 
    idx = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[idx]
    #sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,idx]
    n_eigenvectors = sorted_eigenvectors[:,0:n]
    x_reduced = np.real(np.matmul(n_eigenvectors.transpose(),normalized_face.transpose()).transpose())
    # reproject the reduced image
    x_new = np.real(np.matmul(n_eigenvectors, x_reduced.transpose()).transpose())
    x_new = x_new + np.mean(x, axis= 0)


    return np.real(x_new)
