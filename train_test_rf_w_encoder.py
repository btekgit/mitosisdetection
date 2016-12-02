# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:58:52 2016

@author: btek
"""
import numpy as np
import matplotlib.pyplot as plt

import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sampleFactory import normRGB, denormRGB
sys.path.append('./objdetector/')
import EncoderNetwork as EN
import sampleFactory

INPUTFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_X.npy'
#INPUTTARGETFILE = u'/home/btek/Dropbox/code/matlabcode/mitosis_code/amida_elm/AugmentedSampleAll_v2_orig_X.pkl'
TARGETCLASSFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_Y.npy'
MODELFILE = u'elm_model_all'

WIDTH = 50
HEIGHT = 50
debug = False
USE_ENCODER = False

#INPUTFILE = '/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/OriginalSampleX.npy'
#TARGETCLASSFILE = '/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/OriginalSampleY.npy'

narg = len(sys.argv)
    
input_shape = WIDTH*HEIGHT*3

#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')


sampleFactory.plotTrainSetInSubPlots(XXin[-40:,:],WIDTH,HEIGHT,3,101)
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

if USE_ENCODER:
    encode_size = 100
    ed = EN.EncoderNetwork((1,3,WIDTH,HEIGHT), encode_size=100, reg_weight=0.01, use_st=True)
    ed.load_params('./objdetector/model_20.npz')
    
    XXtrainIn = np.reshape(XXtrainIn, (-1, WIDTH, HEIGHT, 3))
    XXtest = np.reshape(XXtest, (-1, WIDTH, HEIGHT, 3)) 
    XXtrainIn = np.transpose(XXtrainIn, (0,3,2,1))
    XXtest = np.transpose(XXtest,(0,3,2,1))
    
    

    Xtr_enc = ed.get_encoded_feas(XXtrainIn, batch=1000)
    Xtst_enc = ed.get_encoded_feas(XXtest, batch=1000)
else:
    Xtr_enc = XXtrainIn
    Xtst_enc= XXtest 
        


clf = RandomForestClassifier(n_estimators=5, class_weight = wtrain)
clf = clf.fit(Xtr_enc, ytrain)
#clf_transformed = clf.apply(XXtrainIn)
pred = clf.predict(Xtr_enc)
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
print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
print "Tneg:",ntneg," / ", nneg, "TD:", ntneg/float(nneg)
print "Acc: ", nhit/(float)(len(ypred)), "total", len(ypred)

pred = clf.predict(Xtst_enc)
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


