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
import argparse
import time

from spXY import SPXY
import tools


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Approximate Y using selection from X')
    parser.add_argument('-Y', type = str, required = True, help = 'input csv file of Y, column matrix.')
    parser.add_argument('-X', type = str, default = '', help = 'input csv file of X, column matrix. If not given, X = Y.')
    parser.add_argument('-k', type = int, required = True, help = '#columns to be selected from X.')

    args = parser.parse_args()

    Y_file = args.Y
    X_file = args.X
    k = args.k
    
    # read data
    Y = np.loadtxt(Y_file, delimiter=',')
    if len(X_file) == 0:
        X = Y
    else:
        X = np.loadtxt(X_file, delimiter=',')
    print(X.shape, Y.shape)
    
    tic = time.time()
    selected_idx = SPXY(Y, k, X)
    toc = time.time()

    # print results:
    print("===================Results==========================")
    print('mX:nX=', X.shape)
    print('mY:nY=', Y.shape)
    print('selection:', selected_idx)
    if len(selected_idx) > 0:
        err, err_n, explained, explained_n,total = tools.error(Y, selected_idx, X)
        _, _, explainedPCA, _ = tools.PCAerror(Y, k)
        
        # print('explained:', explained, "(%10.4e)"% err, '|| percentage gain:', explained_n, " rank k explained:", explainedPCA/total)
        print('error:', err, '|| percentage error:', err_n)
        print('fractional bound:', (1-explained/explainedPCA)*100)
        print('running time    :', toc-tic,'s.')
    else:
        print('something wrong???')
    
    