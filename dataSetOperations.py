# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 11:38:12 2016

@author: btek
"""
import numpy as np
from sklearn.cross_validation import train_test_split

import listOps

def divideSetsRandomByRatio(X, Y, ratio, balanceOnLabel=None,
                           randomize=True, randState=0):
    """
    This function divides data X and label Y into two sets by ratio
        Parameters
        ----------
        X: float
            (NxM) feature data matrix

        Y: float
            (N,) array of labels

        ratio: data will be divided with this ratio.

        balanceOnLabel: if this has a value, ratio will be
        applied to this label and same number of samples will be chosen
        from the other label, so that resulting second dataset will have
        equal number of labels. Assuming label is binary

        randomixze: set True for random picking of samples.
    """

    if  randomize and not balanceOnLabel:
        # print X.shape[0], Y.shape[0]
        # trnX, trnY = X.copy(), Y.copy()
        trnX, tstX, trnY, tstY = train_test_split(X, Y, test_size=ratio,
                                                  random_state=randState)
    else:


        numAll = len(Y)

        indices = np.array(range(numAll))


        indiceSet1 = indices[Y == balanceOnLabel]
        indiceSet2 = indices[Y != balanceOnLabel]
        if randomize:
            indiceSet1 = np.random.permutation(indiceSet1)
            indiceSet2 = np.random.permutation(indiceSet2)

            # take number of samples in the balance priority label
        numBalanceLabel = len(indiceSet1)
        numOtherLabel = len(indiceSet2)
        if numBalanceLabel < numOtherLabel:
            numForTest = np.int(numBalanceLabel *  ratio)
        else:
            numForTest = np.int(numOtherLabel *  ratio)
            # this many samples will be in

        # print numAll, numBalanceLabel, numOtherLabel, numForTest

        indicesTst = np.concatenate((indiceSet1[0:numForTest],
                                         indiceSet2[0:numForTest]), axis=0)

        indicesTrn = np.concatenate((indiceSet1[numForTest:],
                                         indiceSet2[numForTest:]), axis=0)


        trnX = X[indicesTrn,:]
        tstX = X[indicesTst,:]
        trnY = Y[indicesTrn]
        tstY = Y[indicesTst]
    
    return trnX, tstX, trnY, tstY

def balanceLabels(X, Y, balanceOnLabel=None,
                       randomize=True, randState=0):
    """
    This function divides data X and label Y into two sets by ratio
        Parameters
        ----------
        X: float
            (NxM) feature data matrix
    
        Y: float
            (N,) array of labels
    
        ratio: data will be divided with this ratio.
    
        balanceOnLabel: if this has a value, ratio will be
        applied to this label and same number of samples will be chosen
        from the other label, so that resulting second dataset will have
        equal number of labels. Assuming label is binary
    
        randomixze: set True for random picking of samples.
    """
    
    
    numAll = len(Y)
    
    indices = np.array(range(numAll))
    # take the first label
    if balanceOnLabel is None:
        balanceOnLabel = Y[0]
    
    indiceSet1 = indices[Y == balanceOnLabel]
    indiceSet2 = indices[Y != balanceOnLabel]
    if randomize:
        indiceSet1 = np.random.permutation(indiceSet1)
        indiceSet2 = np.random.permutation(indiceSet2)
    
            # take number of samples in the balance priority label
    numBalanceLabel = len(indiceSet1)
    numOtherLabel = len(indiceSet2)
    if numBalanceLabel < numOtherLabel:
        numForRes = np.int(numBalanceLabel)
    else:
        numForRes = np.int(numOtherLabel)
            # this many samples will be in
    
        # print numAll, numBalanceLabel, numOtherLabel, numForTest
    
    indicesRes = np.concatenate((indiceSet1[0:numForRes],
                                         indiceSet2[0:numForRes]), axis=0)
    
    resX = X[indicesRes,:]
    resY = Y[indicesRes]
    
    return resX, resY

