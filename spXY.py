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
from randomEVD import randomEVD
from scipy.linalg import qr_delete

def SPXY(Y, K, X=np.empty((0,0)), DEBUG=False):
  '''
  X: source matrix.
  Y: target matrix.
  K: number of columns to be selected from X to approximate Y.
  If X is not given, then X and Y are same.
  '''

  mX, nX = X.shape
  mY, nY = Y.shape
  if mX > 0 and mY != mX:
    print("error! mX must be same as mY.")
    return []
  if K <= 0:
    return []
  
  # ===STAGE 1: Selection===
  Yr = Y
  Xr = Y
  if mX > 0:
    Xr = X

  inds=np.zeros(K,)
  inds=inds.astype(int)
  for k in range(0,K):
    # step1: first eigenvector of the reduced Y.
    u = randomEVD(Yr, 1) # m x 1
    
    # step2: correlation of x_i and u
    norms=np.linalg.norm(Xr,axis=0)
    norms[norms==0] = 1e-20 # to avoid from division of 0
    Xr_n=Xr/norms

    Xr_n=np.transpose(Xr_n) # n x m
    scores=np.abs(Xr_n@u).reshape((1,-1))[0]
    ind=np.argsort(scores)[-1]
    inds[k]=ind

    # step3: compute reduced matrices 
    q = Xr_n[ind,:].reshape((-1,1))
    tmp = q.T@Yr
    Yr=Yr-q@tmp
    if mX > 0:
      tmp = q.T@Xr
      Xr=Xr-q@tmp
    else:
      Xr = Yr

  # === STAGE 2: Improvement ===
  minerr = np.linalg.norm(Yr)**2
  mininds = np.copy(inds)
  print("initial error:", minerr, mininds)
  
  errors = [minerr]
  COUNT = 0 # for early stop
  olderr = minerr
  for iter in range(0,30):
    k = iter%K #+2 #skip the first two assuming that they are usually good selection
    if k > (K-1):
      continue

    ind2=np.delete(inds,k)
    if mX > 0:
      S=X[:,ind2]
    else:
      S=Y[:,ind2]
    Q,_ = np.linalg.qr(S)
    
    tmp = Q.T@Y
    Yr=Y-Q@tmp
    if mX > 0:
      tmp = Q.T@X
      Xr=X-Q@tmp
    else:
      Xr = Yr
    
    u = randomEVD(Yr, 1) # m x 1
    
    norms=np.linalg.norm(Xr,axis=0)
    norms[norms==0] = 1e-20
    Xr_n=Xr/norms

    tmp=np.transpose(Xr_n)
    scores=np.abs(tmp@u).reshape((1,-1))[0]
    ind=np.argsort(scores)[-1]
    
    
    # new error
    err = np.linalg.norm(Yr)**2
    explained_new = Xr_n[:,ind].reshape((-1,1)).T@Yr
    err -= np.linalg.norm(explained_new)**2
    print('new error:', err, (err < minerr and inds[k] != ind))
    
    if err < minerr and inds[k] != ind:
      print('--', iter, inds[k], '-->', ind)
      inds[k]=ind
      minerr = err
      mininds = np.copy(inds)
    
    if DEBUG:
        errors.append(minerr)
    
    # early stop
    if (olderr-err) <= 0.00001:
      COUNT += 1
    else:
        COUNT = 0
    if COUNT > K:
      break

    olderr = err
  if DEBUG:
      print("errors=",errors)
  
  return mininds
