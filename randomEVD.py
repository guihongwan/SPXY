"""
Implementation of the method proposed in the paper:
'A Fast Algorithm for Simultaneous Sparse Approximation'
by Guihong Wan, Haim Schweitzer.
Published at PAKDD 2021.
Copyright (C) 2021
Guihong Wan
The University of Texas at Dallas.
"""

import numpy as np

def getRandomMatrix(m, k, mean=0, std=1, seed=-1):
    '''
    Generate randomly (Gaussian) a matrix; 
    the shape is m by k.
    '''
    if seed > 0:
        np.random.seed(seed)
    return np.random.normal(mean, std, m*k).reshape((m,k))

def getOrthogonalMatrix(m, k, mean=0, std=1, seed = -1):
    '''
    Generate randomly (Gaussian) a matrix; the shape is m by k.
    And then QR s.t. the columns of produced matrix are orthogonal:
    Q.T@Q=I
    '''
    H = getRandomMatrix(m, k, mean, std, seed)
    Q, R = np.linalg.qr(H)
    return Q

def EVDnotPSD(B):
    '''
    The full EVD of B.
    @input: B
    B may be not PSD.
    '''
    e, V = np.linalg.eig(B)
    
    sorted_idxes = np.argsort(-e)
    e = e[sorted_idxes]
    V = V[:, sorted_idxes]

    return (e.real, V.real)

def randomEVD(X, r, iteration=5):
    '''
    X is a column matrix with size mxn.
    output: return the first r eigenvectors of B = XX^T
    '''
    m, n = X.shape
    p = 10 # oversampling parameter
    
    # stage 1
    Q = getOrthogonalMatrix(m, r+p)
    for j in range(iteration):
        tmp = X.T@Q
        Y = X@tmp
        Q, R = np.linalg.qr(Y)
    
    # stage 2
    tmp = X.T @ Q
    B = tmp.T @ tmp
    
    E, V = EVDnotPSD(B) # B may be not PSD
    V = Q @ V
    V = V[:,:r].reshape((m,r))

    return V







