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

plotTrainSetInSubPlots(XXin[-40:, :], WIDTH, HEIGHT, 3, 101)
XXin = reshapeData(XXin, (WIDTH, HEIGHT, 3), (12, 12, 3))
plotTrainSetInSubPlots(XXin[-40:, :], 12, 12, 3, 101)

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
WEIGHTBALANCE = 0.5
wtrain = dict({0:1.0-WEIGHTBALANCE, 1:WEIGHTBALANCE})
#wtrain = dict({0:1.0,1:1.0})


clf = RandomForestClassifier(n_estimators=105, class_weight=wtrain, max_depth=5)
clf = clf.fit(XXtrainIn, ytrain)
#clf_transformed = clf.apply(XXtrainIn)
pred = clf.predict(XXtrainIn)
#save the classifier
_ = joblib.dump(clf, 'random_forest_trained.pkl', compress =9)

labels = ytrain.squeeze()
ypred = pred#np.argmax(pred, axis = 1)
nhit = sum(ypred==labels)
ntpos = sum((ypred==1) & (labels==1))
ntneg = sum((ypred==0) & (labels==0))
npos = sum((labels==1))
nneg = sum((labels==0))

print "\n Random Forest train results"
print "Tpos:", ntpos," / ", npos, "TD:", ntpos/float(npos)
print "Tneg:", ntneg," / ", nneg, "TD:", ntneg/float(nneg)
print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
print "Score prec,rec,f-score,supprt:", precision_recall_fscore_support(labels, ypred) 

pred = clf.predict(XXtest)
labels = ytest.squeeze()
ypred = pred#np.argmax(pred, axis = 1)
nhit = sum(ypred==labels)
ntpos = sum((ypred==1) & (labels==1))
ntneg = sum((ypred==0) & (labels==0))
npos = sum((labels==1))
nneg =sum((labels==0))
print "\n Random Forest Testing results"
print "Tpos:", ntpos," / ", npos, "TD:", ntpos/float(npos)
print "Tneg:", ntneg," / ", nneg, "TN:", ntneg/float(nneg)
print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)
print "prec,rec,f-score,supprt:", precision_recall_fscore_support(labels, ypred)[0]
print "rec,f-score,supprt:", precision_recall_fscore_support(labels, ypred)[1]
print "f-score,supprt:", precision_recall_fscore_support(labels, ypred)[2]



#if __name__ == "__main__":
    #main()



#from sklearn.ensemble import RandomTreesEmbedding
#from sklearn.linear_model import LogisticRegression
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.pipeline import make_pipeline
#from sklearn import cross_validation
#from sklearn import metrics
#
#rt = RandomTreesEmbedding(max_depth=3, n_estimators=5,random_state=0)
#rt_lm = RandomForestClassifier(n_estimators=20)
#pipeline = make_pipeline(rt, rt_lm)
#pipeline.fit(XXtrainIn, YYtrain)
#y_pred_rt = pipeline.predict(XXtrainIn)
##fpr_rt_lm, tpr_rt_lm, _ = metrics.roc_curve(ytest, y_pred_rt)
    ##
#p = ytrain.squeeze()
#yout = np.argmax(y_pred_rt, axis = 1)
#nhit = sum(yout==p)
#ntpos = sum((yout==1) & (p==1))
#npos = sum((p==1))
###
#print "\n Embedded Forest train results"
#print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
#print "Acc: ", nhit/(float)(len(p)), "total", len(p)

#predx = xlf.predict(XXtest)
#p = ytest.squeeze()
#yout = predx #np.argmax(predx, axis = 1)
#nhit = sum(yout==p)
#ntpos = sum((yout==1) & (p==1))
#ntneg = sum((yout==0) & (p==0))
#npos = sum((p==1))
#nneg =sum((p==0))
#print "\n Extra Forest Testing results"
#print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
#print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
#print "Acc: ", nhit/(float)(len(p)), "total", len(p)
