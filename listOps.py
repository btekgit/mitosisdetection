# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 20:52:12 2016

@author: btek
"""
import numpy as np


def listDiff(la, lb):
    d = list(x for x in la if x not in lb)
    return d



def randomlySelectN(la, N):
# this method selects N samples random from a list
    nn = len(la)
    lb = []
    if nn>N:
        #print "length of falsepos", len(falsepos)
        #print "randomly selecting", MAXNEGATIVESAMPLESTOTRAIN
        rindex = range(nn)
        lb = np.random.permutation(la)
        lb = lb[0:N]
        return list(lb)
        #print "length of falsepos", len(falsepos)    
    else:
        return lb