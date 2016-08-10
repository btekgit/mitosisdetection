# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 16:31:56 2016

@author: btek
"""
from sklearn import ensemble
from sklearn.externals import joblib
import cascade


X = joblib.load('bootstrap_collected_sampleX.pkl')

Y = joblib.load('bootstrap_collected_sampleY.pkl')
clf = ensemble.RandomForestClassifier()
a = cascade.CascadedBooster(base_clf=clf, max_layers=4)
a.fit(X,Y)
a.add_cascade_layer(X, Y)
