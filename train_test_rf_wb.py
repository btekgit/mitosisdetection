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
from sklearn.metrics import f1_score

from sampleFactory import normRGB, denormRGB
from dataSetOperations import divideSetsRandomByRatio
import dataSetOperations

ROOTFOLDER = '/home/btek/data/mitosisData/'
INPUTFILE = ROOTFOLDER+'OriginalSampleX.npy'
TARGETFILE = ROOTFOLDER+'OriginalSampleY.npy'
#INPUTFILE = ROOTFOLDER+'AugmentedSample_v1_X.npy'
#TARGETFILE = ROOTFOLDER+'AugmentedSample_v1_Y.npy'

WIDTH = 50
HEIGHT = 50
debug = False
 

narg = len(sys.argv)
   
input_shape = WIDTH*HEIGHT*3

#gfilter = makeGaussian(50,)

#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')
print XXin.shape, "max: ", np.max(XXin), "min: ", np.min(XXin)

Yin= (np.load(TARGETFILE)).astype('int32')
print Yin.shape, "pos:",sum(Yin[:]), "neg", len(Yin[:])-sum(Yin[:])

trainix = range(0, XXin.shape[0]/2, 1)
testix = range(XXin.shape[0]/2, XXin.shape[0], 1)
XXtrainIn = XXin[trainix,]
ytrain = Yin[trainix]
XXtest = XXin[testix,]
ytest = Yin[testix]

#XXin, Yin = dataSetOperations.balanceLabels(XXin, Yin,balanceOnLabel=1)
import sampleFactory
#sampleFactory.plotTrainSetInSubPlots(XXin[-40:,:],50,50,3,101)


# XXtrainIn, XXtest, ytrain, ytest = divideSetsRandomByRatio(XXin,Yin,0.5)


# create weight to create a bias    
WEIGHTBALANCE = 1.0-(sum(Yin[:])/float(len(Yin)))
wtrain = dict({0:1.0-WEIGHTBALANCE, 1:WEIGHTBALANCE})
print wtrain
# wtrain = dict({0:1.0,1:1.0})

clf = RandomForestClassifier(n_estimators=21, class_weight = wtrain,
max_depth=10)
clf = clf.fit(XXtrainIn, ytrain)

pred = clf.predict(XXtrainIn)

#save the classifier
_ = joblib.dump(clf, 'random_forest_trained_wb.pkl', compress =9)

labels = ytrain.squeeze()
ypred = pred#np.argmax(pred, axis = 1)
nhit = sum(ypred==labels)
ntpos = sum((ypred==1) & (labels==1))
ntneg = sum((ypred==0) & (labels==0))
npos = sum((labels==1))
nneg = sum((labels==0))

print "\n Random Forest train results"
print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
print "Tneg:",ntneg," / ", nneg, "TD:", ntneg/float(nneg)
print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
print "F-sc: ", f1_score(labels, ypred)

pred = clf.predict(XXtest)
labels = ytest.squeeze()
ypred = pred#np.argmax(pred, axis = 1)
nhit = sum(ypred==labels)
ntpos = sum((ypred==1) & (labels==1))
ntneg = sum((ypred==0) & (labels==0))
npos = sum((labels==1))
nneg =sum((labels==0))
print "\n Random Forest Testing results"
print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
print "F-sc: ", f1_score(labels, ypred)

#for k in range(0, XXin.shape[0], 1000):
#    sampleFactory.plotTrainSetInSubPlots(XXin[k:k+20,:],50,50,3,101)
#    print Yin[k:k+20]

augment = True
if augment == True:
    XXtrainIn3D =  np.reshape(XXtrainIn,(-1,50,50,3))
    #XXtrainInAug = XXtrainInAug[0:10,:]
    #XXaug  = XXtrainInAug
    
    XXaug, aug_ix = sampleFactory.generateSamples(XXtrainIn3D,10)
    
    XXtrainAug = np.reshape(XXaug, (-1, 50*50*3))
    ytrainAug = ytrain[aug_ix]
    clf_aug = RandomForestClassifier(n_estimators=51, class_weight = wtrain,
    max_depth=10)
    clf_aug = clf_aug.fit(XXtrainAug, ytrainAug)

    pred = clf_aug.predict(XXtrainAug)
    labels = ytrainAug.squeeze()
    ypred = pred
    nhit = sum(ypred==labels)
    ntpos = sum((ypred==1) & (labels==1))
    ntneg = sum((ypred==0) & (labels==0))
    npos = sum((labels==1))
    nneg = sum((labels==0))

    print "\n Random Forest Augmented train results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Tneg:",ntneg," / ", nneg, "TD:", ntneg/float(nneg)
    print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
    print "F-sc: ", f1_score(labels, ypred)
    
    pred = clf_aug.predict(XXtest)
    labels = ytest.squeeze()
    ypred = pred
    nhit = sum(ypred==labels)
    ntpos = sum((ypred==1) & (labels==1))
    ntneg = sum((ypred==0) & (labels==0))
    npos = sum((labels==1))
    nneg =sum((labels==0))
    print "\n Random Forest Augmented training testing results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
    print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
    print "F-sc: ", f1_score(labels, ypred)
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(100, 100), random_state=1, verbose=False,learning_rate='adaptive')
    
    clf.fit(XXtrainIn, ytrain)
    
    pred = clf.predict(XXtest)
    labels = ytest.squeeze()
    ypred = pred
    nhit = sum(ypred==labels)
    ntpos = sum((ypred==1) & (labels==1))
    ntneg = sum((ypred==0) & (labels==0))
    npos = sum((labels==1))
    nneg =sum((labels==0))
    print "\n Neural Network Original training testing results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
    print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
    print "F-sc: ", f1_score(labels, ypred)
    
    clf.fit(XXtrainAug, ytrainAug)
    
    pred = clf.predict(XXtest)
    labels = ytest.squeeze()
    ypred = pred
    nhit = sum(ypred==labels)
    ntpos = sum((ypred==1) & (labels==1))
    ntneg = sum((ypred==0) & (labels==0))
    npos = sum((labels==1))
    nneg =sum((labels==0))
    print "\n Neural Network Augmented training testing results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
    print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
    print "F-sc: ", f1_score(labels, ypred)
    
