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

def error(Y, S, X=np.empty((0,0))):
  '''
     If X is not given, X = Y.
  '''
  if X.shape[0] > 0:
     XS = X[:,S]
  else:
     XS = Y[:,S]
  Q,_ = np.linalg.qr(XS)
  W = Q.T@Y
  total = np.linalg.norm(Y)**2
  explained = np.linalg.norm(W)**2
  error = total - explained

  return error, (error/total)*100, explained, (explained/total)*100,total

def PCAerror(X, k):
  '''
  X: column matrix
  '''
  [U,S,V]=np.linalg.svd(X,full_matrices=False)
  U=U[:,:k]
  W = U.T@X
  total = np.linalg.norm(X)**2
  explained = np.linalg.norm(W)**2
  error = total - explained
  return error, (total/total)*100, explained, (explained/total)*100
