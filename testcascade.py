# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:31:56 2016

@author: btek
"""
from sklearn import ensemble
from sklearn.externals import joblib
import cascade
import sampleFactory
import numpy as np
import time

INPUTFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_X.npy'
#INPUTTARGETFILE = u'/home/btek/Dropbox/code/matlabcode/mitosis_code/amida_elm/AugmentedSampleAll_v2_orig_X.pkl'
TARGETCLASSFILE = u'/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/AugmentedSampleAll_v3_Y.npy'
MODELFILE = u'elm_model_all'
WIDTH = 50
HEIGHT = 50
debug = False
QUICKTEST = False

if QUICKTEST:
    X = joblib.load('bootstrap_collected_sampleX.pkl')
    Y = joblib.load('bootstrap_collected_sampleY.pkl')
else:
    X = (np.load(INPUTFILE)).astype('float32')

    X = sampleFactory.normRGB(X)
    Y = (np.load(TARGETCLASSFILE)).astype('int32').ravel() 
    N = 30000
    M = 2000
    X1 = X[0:N,:]
    Y1 = Y[0:N]
    X2 = X[N+M:,:]
    Y2 = Y[N+M:]
clf = ensemble.RandomForestClassifier()
a = cascade.CascadedBooster(base_clf=clf, max_layers=1, priors=[0.90, 0.01])
a.set_params(nAdaptiveNumberEstimators = True)
a.fit(X1, Y1)

t1 = time.clock()
prob = a.predict_proba(X)
tp = ((Y == 1) & (prob[:,1]>=0.5)).sum() / float((Y==1).sum())
fp = 1 - ((Y == 0) & (prob[:,1]<0.5)).sum() / float((Y==0).sum())
pr = ((Y == 1) & (prob[:,1]>=0.5)).sum() / float((prob[:,1]>=0.5).sum())
print "Tp: ", tp, "Fp: ", fp, "Pr: ", pr
t2 = time.clock()
print "Elapsed Time: ", t2-t1

prob = a.predict_proba(X2)
tp = ((Y2 == 1) & (prob[:,1]>=0.5)).sum() / float((Y2==1).sum())
fp = 1 - ((Y2 == 0) & (prob[:,1]<0.5)).sum() / float((Y2==0).sum())
pr = ((Y2 == 1) & (prob[:,1]>=0.5)).sum() / float((prob[:,1]>=0.5).sum())
print "Tp: ", tp, "Fp: ", fp, "Pr: ", pr
# test on whole

#ypred = a.predict(X1)
#tp = ((Y1 == 1) & ypred).sum() / float((Y1==1).sum())
#fp = 1- ((Y1 == 0) & (ypred==False)).sum() / float((Y1==0).sum())
#print "Tp: ", tp, "Fp: ", fp
a.max_layers = 100
clf = a
_ = joblib.dump(clf, 'random_forest_cascade_trained.pkl', compress =9)


# here I test adding a new layer with M samples. 

a.set_params(nAdaptiveNumberEstimators = True, warm_start = True, n_estimators = 3)
a.fit(X[N:N+M,:], Y[N:N+M])
t1 = time.clock()
prob = a.predict_proba(X2)
tp = ((Y2 == 1) & (prob[:,1]>=0.5)).sum() / float((Y2==1).sum())
fp = 1 - ((Y2 == 0) & (prob[:,1]<0.5)).sum() / float((Y2==0).sum())
pr = ((Y2 == 1) & (prob[:,1]>=0.5)).sum() / float((prob[:,1]>=0.5).sum())
print "Tp: ", tp, "Fp: ", fp, "Pr: ", pr
t2 = time.clock()
print "Elapsed Time: ", t2-t1


#a.add_cascade_layer(X[N:N+M,], Y[N:N+M])
#
#t1 = time.clock()
#pred, prob = a.predict_proba(X1)
#tp = ((Y1 == 1) & (pred>=0)).sum() / float((Y1==1).sum())
#fp = 1 - ((Y1 == 0) & (pred<0)).sum() / float((Y1==0).sum())
#print "Tp: ", tp, "Fp: ", fp
#t2 = time.clock()
#print "Elapsed Time: ", t2-t1

