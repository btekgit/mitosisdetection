# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:58:52 2016

@author: btek
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
#import sklearn.datasets
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_fscore_support
from sampleFactory import normRGB, denormRGB, plotTrainSetInSubPlots, reshapeData
from sparse_filtering import SparseFiltering

def report_accuracy(pred, labels, name=None):
    
    ypred = pred
    nhit = sum(ypred == labels)
    ntpos = sum((ypred==1) & (labels==1))
    ntneg = sum((ypred==0) & (labels==0))
    npos = sum((labels==1))
    nneg =sum((labels==0))
    print name
    print "Tpos:", ntpos," / ", npos
    print "Tneg:", ntneg," / ", nneg
    print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
    print "prec:", precision_recall_fscore_support(labels, ypred)[0]
    print "rec:", precision_recall_fscore_support(labels, ypred)[1]
    print "f-score:", precision_recall_fscore_support(labels, ypred)[2]

ROOTFOLDER = u'/home/btek/data/mitosisData/'
INPUTFILE = ROOTFOLDER+'AugmentedSample_v1_X.npy'
#INPUTFILE = ROOTFOLDER+'OriginalSampleX.npy'
#TARGETCLASSFILE = ROOTFOLDER+'AugmentedSampleAll_v3_Y.npy'
TARGETCLASSFILE = ROOTFOLDER+'AugmentedSample_v1_Y.npy'
WIDTH = 50
HEIGHT = 50
  
input_shape = WIDTH*HEIGHT*3
np.random.seed(999)
#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')

#plotTrainSetInSubPlots(XXin[-40:, :], WIDTH, HEIGHT, 3, 101)
XXin = reshapeData(XXin, (WIDTH, HEIGHT, 3), (12, 12, 3))
#plotTrainSetInSubPlots(XXin[-40:, :], 12, 12, 3, 101)

print XXin.shape, "max: ", np.max(XXin), "min: ", np.min(XXin)

y= (np.load(TARGETCLASSFILE)).astype('int32')

train_lim = XXin.shape[0]/2

trainix = range(0, train_lim, 1)
testix = range(train_lim, XXin.shape[0], 1)

XXtrainIn = XXin[trainix,]
XXtest = XXin[testix,]
ytrain = np.ravel(y[trainix])
ytest = np.ravel(y[testix])

del XXin

# create weight to create a bias    
WEIGHTBALANCE = 0.8
wtrain = dict({0:1.0-WEIGHTBALANCE, 1:WEIGHTBALANCE})
#wtrain = dict({0:1.0,1:1.0})


clf = RandomForestClassifier(n_estimators=51, class_weight=wtrain, max_depth=15)
clf = clf.fit(XXtrainIn, ytrain)
#clf_transformed = clf.apply(XXtrainIn)
pred = clf.predict(XXtrainIn)
#save the classifier
_ = joblib.dump(clf, 'random_forest_trained.pkl', compress =9)

labels = ytrain.squeeze()
report_accuracy(pred,labels, ' RF Raw Train')

pred = clf.predict(XXtest)
labels = ytest.squeeze()
report_accuracy(pred,labels, ' RF Raw Test')

#SparseFiltering
n_features = 64   # How many features are learned

estimator = SparseFiltering(n_features=n_features,
                            maxfun=200,  # The maximal number of evaluations of the objective function
                            iprint=-1)  # after how many function evaluations is information printed
                                        # by L-BFGS. -1 for no information
print("sparse yapıldı")
#model2 = RandomForestClassifier(n_estimators=20, max_depth=100, criterion='entropy', max_features='sqrt',max_leaf_nodes=30,oob_score=False)

train_features = estimator.fit_transform(XXtrainIn)
rfr = RandomForestClassifier(n_estimators=51, class_weight=wtrain, max_depth=15)
rfr.fit(train_features,ytrain.ravel())
pred = rfr.predict(train_features)
report_accuracy(pred,ytrain.ravel(), 'Sparse RF Train')

test_features = estimator.transform(XXtest)
pred = rfr.predict(test_features)
report_accuracy(pred,ytest.ravel(), 'Sparse RF Test')


from sklearn.svm import SVC
clf = SVC(class_weight=wtrain)
clf.fit(train_features, ytrain)
pred = clf.predict(train_features)
report_accuracy(pred,ytrain.ravel(), 'Sparse SVM Train')
pred = clf.predict(test_features)
report_accuracy(pred,ytest.ravel(), 'Sparse SVM Test')