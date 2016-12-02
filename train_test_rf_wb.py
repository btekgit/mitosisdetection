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
from sampleFactory import normRGB, denormRGB

INPUTFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_X_wb.npy'
#INPUTTARGETFILE = u'/home/btek/Dropbox/code/matlabcode/mitosis_code/amida_elm/AugmentedSampleAll_v2_orig_X.pkl'
TARGETCLASSFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_Y_wb.npy'
MODELFILE = u'elm_model_all'
WIDTH = 50
HEIGHT = 50
debug = False

def makeGaussian(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)



  

narg = len(sys.argv)

    
input_shape = WIDTH*HEIGHT*3


gfilter = makeGaussian(50,)
#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')
#XXout = (np.load(INPUTTARGETFILE)).astype('float32')
##c = np.float32(1/255.0*2.0)
#XXin *=c 
#XXout*=c 
#XXin  -=1.0 
#XXin = normRGB(XXin) 
import sampleFactory
sampleFactory.plotTrainSetInSubPlots(XXin[-40:,:],50,50,3,101)
#input("here")
#XXout = normRGB(XXout) 
#XXin = (XXin/255.0)*2.0 - 1.0
#XXout = (XXout/255.0)*2.0 - 1.0
print XXin.shape, "max: ", np.max(XXin), "min: ", np.min(XXin)

y= (np.load(TARGETCLASSFILE)).astype('int32')
print y
#YY = np.zeros([y.shape[0],2])    
# dont know how to do this in python
#for ix in range(0, y.shape[0]):
#    YY[ix,y[ix]]= 1
#XX = np.load(inputfile)

trainix = range(XXin.shape[0]/3, XXin.shape[0], 1)
testix = range(0, XXin.shape[0]/3, 1)

XXtrainIn = XXin[trainix,]
XXtest = XXin[testix,]

ytrain = np.ravel(y[trainix])    
ytest = y[testix]

# create weight to create a bias    
WEIGHTBALANCE = 0.5
wtrain = dict({0:1.0-WEIGHTBALANCE, 1:WEIGHTBALANCE})
#wtrain = dict({0:1.0,1:1.0})
#wtrain = np.ones(ytrain.shape)
#wtrain[ytrain==1] = wtrain[ytrain==1]*WEIGHTBALANCE
#wtrain =  wtrain / sum(wtrain)

clf = RandomForestClassifier(n_estimators=10, class_weight = wtrain)
clf = clf.fit(XXtrainIn, ytrain)
#clf_transformed = clf.apply(XXtrainIn)
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


nn = 70000
kk = 10000
for k in range(nn, nn+kk, 100):
    sampleFactory.plotTrainSetInSubPlots(XXin[k:k+20,:],50,50,3,101)
    print y[k:k+20]


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
