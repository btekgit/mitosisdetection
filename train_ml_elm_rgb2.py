# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 11:58:52 2016

@author: btek
"""

from hpelm import ELM
#INPUTFILE = u'amida_samples_org.npy'
#INPUTFILE = u'AugmentedSampleAll_aug_X.npy'
#INPUTTARGETFILE = u'AugmentedSampleAll_orig_X.npy'
#TARGETCLASSFILE = u'AugmentedSampleAll_all_y.npy'
INPUTFILE = u'./mitosisData/AugmentedSampleAll_aug_X.npy'
INPUTTARGETFILE = u'./mitosisData/AugmentedSampleAll_orig_X.npy'
TARGETCLASSFILE = u'./mitosisData/AugmentedSampleAll_all_y.npy'
MODELFILE = u'elm_model_all'
WIDTH = 50
HEIGHT = 50
debug = False
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.ensemble import RandomForestClassifier
#from random import randrange

import sys

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
ninputsig = 500


# normalize a matrix of rgb values to zero mean variance
def normRGB(X, mx = 255.0, mn =0):
    
    c = np.float32(1/mx*2.0)
    #print c
    X *= c
    X -= 1.0
    return X
    
    
def mapZeroMeanOneVariance(X):
    mn = np.mean(X,0)
    std = np.std(X,0)
    # python does columnwise subtraction very interestingly!!
    X = (X-mn)/std
    return X
    
    
def mapMinMax(X):
    mx = float(np.max(X))
    mn = float(np.min(X))
    # python does columnwise subtraction very interestingly!!
    X = (X-mn)/(mx-mn)
    return X#mapZeroMeanOneVariance(X)
    
    
    
    
    
narg = len(sys.argv)

    
input_shape = WIDTH*HEIGHT*3


gfilter = makeGaussian(50,)
#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')
XXout = (np.load(INPUTTARGETFILE)).astype('float32')
c = np.float32(1/255.0*2.0)
#XXin *=c 
#XXout*=c 
#XXin  -=1.0 
XXin = normRGB(XXin) 
XXout = normRGB(XXout) 
#XXin = (XXin/255.0)*2.0 - 1.0
#XXout = (XXout/255.0)*2.0 - 1.0
print XXin.shape, "max: ", np.max(XXin), "min: ", np.min(XXin)

y= (np.load(TARGETCLASSFILE)).astype('int32')
YY = np.zeros([y.shape[0],2])
# dont know how to do this in python
for ix in range(0, y.shape[0]):
    YY[ix,y[ix]]= 1
#XX = np.load(inputfile)

trainix = range(0,XXin.shape[0]/2,1)
testix = range(XXin.shape[0]/2,XXin.shape[0],1)
XXtrainIn = XXin[trainix,]
XXtrainOut = XXout[trainix,]
YYtrain = YY[trainix,]
ytrain = y[trainix]

XXtest = XXin[testix,]
YYtest = YY[testix,]
ytest = y[testix]



ninputsig = 900
ninputlin = 3
#num_neurons = max(np.sqrt(n_example), 10)

n_example = XXtrainIn.shape[0]

#id_input = np.random.permutation(n_example)
#id_input =  [i for i in range(n_example) if y[i] ==1]
#id_output = id_input[::-1]
#xinput = XXtrain[id_input,]
#xoutput = XXtrain[id_input,]
#print xinput.shape
## INPUT LAYER
singleLayer = True
if singleLayer:
    t = 5
    mn_error = np.zeros([t,1])
    for i in range(0,t):
        elmSingle = ELM(input_shape, YY.shape[1])
        #elmSingle.add_neurons(ninputsig/2, "sigm")
        elmSingle.add_neurons(ninputsig, "sigm")
        #elmSingle.add_neurons(100, "rbf_l1")
        #elmSingle.add_neurons(ninputsig/10, "lin")
        elmSingle.train(XXtrainIn, YYtrain,"c", norm = 1e-5) 
        print "\n Trained input elm",elmSingle
        youtput = elmSingle.predict(XXtest)

        p = ytest.squeeze()
        yout = np.argmax(youtput, axis = 1)
        nhit = sum(yout==p)
        ntpos = sum((yout==1) & (p==1))
        ntneg = sum((yout==0) & (p==0))
        npos = sum((p==1))
        nneg =sum((p==0))
        print "\n Testing results"
        print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
        print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
        print "Acc: ", nhit/(float)(len(p)), "total", len(p)
        mn_error[i] = nhit/(float)(len(p))
    print mn_error
    print "mean error", np.mean(mn_error)

else:
    elmInput = ELM(input_shape, input_shape)
    elmInput.add_neurons(ninputsig, "tanh")
    #elmInput.add_neurons(500, "lin")
    elmInput.train(XXtrainIn, XXtrainOut,"r") 
    elmInputProjection= elmInput.project(XXtrainIn) 
    print "\n Trained input elm ", elmInput
    print "Projection Max :", np.max(elmInputProjection), "Min :", np.min(elmInputProjection)
    
    # do norm before continuing.
    elmInputProjectionNormed = mapMinMax(elmInputProjection)
    print "Normed Projection Max :", np.max(elmInputProjectionNormed), "Min :", np.min(elmInputProjectionNormed)

    ## HIDDEN LAYER    
    elmHidden1 = ELM(elmInputProjectionNormed.shape[1], elmInputProjectionNormed.shape[1])
    elmHidden1.add_neurons(ninputsig, "tanh")
    elmHidden1.add_neurons(ninputlin, "lin")
    elmHidden1.train(elmInputProjectionNormed, elmInputProjectionNormed,"r")
    elmHiddenProjection= elmHidden1.project(elmInputProjectionNormed)
    
    print " Hidden Projection Max :", np.max(elmHiddenProjection), "Min : ", np.min(elmHiddenProjection)
    # do norm before continuing.
    elmHiddenProjectionNormed = mapMinMax(elmHiddenProjection)
    print "\n Trained hidden elm",elmInput
    print " Normed Projection Max :", np.max(elmHiddenProjectionNormed), "Min : ", np.min(elmHiddenProjectionNormed)

    # OUTPUT LAYER 
    elmOutput = ELM(elmHiddenProjectionNormed.shape[1], YY.shape[1])
    elmOutput.add_neurons(ninputsig, "tanh")
    elmOutput.add_neurons(ninputlin, "lin")
    
    # train and get result
    elmOutput.train(elmHiddenProjectionNormed, YYtrain,"c")
    # now prediction for trainining
    youtput = elmOutput.predict(elmHiddenProjectionNormed)
    print "\n Trained output elm",elmOutput
    print " Output Projection Max :",np.max(youtput), "Min :", np.min(youtput)
    
    # training results
    mse_error = elmOutput.error(youtput, YYtrain)
    print "Training mse:", mse_error
    p = ytrain.squeeze()
    yout = np.argmax(youtput, axis = 1)
    nhit = sum(yout==p)
    ntpos = sum((yout==1) & (p==1))
    npos = sum((p==1))
    print "\n Training results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Acc: ", nhit/(float)(len(p)), "total", len(p)
    
    
    # now prediction for testing set
    prjinput = elmInput.project(XXtest)
    prjinput_normed = mapMinMax(prjinput)
    prjhidden = elmHidden1.project(prjinput_normed)
    prjhidden_normed = mapMinMax(prjhidden)
    youtput = elmOutput.predict(prjhidden_normed)
    
    
    # training results
    mse_error = elmOutput.error(youtput, YYtest)
    p = ytest.squeeze()
    yout = np.argmax(youtput, axis = 1)
    nhit = sum(yout==p)
    ntpos = sum((yout==1) & (p==1))
    ntneg = sum((yout==0) & (p==0))
    npos = sum((p==1))
    nneg =sum((p==0))
    print "\n Testing results"
    print "Tpos:",ntpos," / ", npos, "TD:", ntpos/float(npos)
    print "Tneg:",ntneg," / ", nneg, "TN:", ntneg/float(nneg)
    print "Acc: ", nhit/(float)(len(p)), "total", len(p)
    
    
    
    print '\ saving layers'
    elmInput.save("elm_input_0")
    elmHidden1.save("elm_hidden_0")
    elmOutput.save("elm_output_0")






