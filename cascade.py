# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:38:55 2016

@author: btek

Based on code
by Chris Beaumont https:
//github.com/ChrisBeaumont/brut/blob/master/bubbly/cascade.py
"""
from math import ceil
from warnings import warn
from functools import wraps

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone
from dataSetOperations import divideSetsRandomByRatio
ADAPTIVENESTIMATORS = True
MIN_SAMPLES = 20


def _recall_bias(df, frac):
    """Choose a bias for a decision function, such that
    at least `frac` fraction of positive examples satisfy
    clf.get_clf_binary_decision_func(X) >= bias
    Parameters
    ----------
    df: array-like
        Decision function evalued for positive examples
    frac : float
        Minimum fraction of positive examples to pass through
    Returns
    -------
    float
    """
    df.sort()

    tiny = 1e-7
    if frac <= 0:
        return df[-1] + tiny
    if frac >= 1:
        return df[0]

    ind = int((1 - frac) * df.size)
    return df[ind]


#def _fpos_bias(df, fpos, tneg):
#    """Choose a bias for a decision function, such that
#    at most `frac` fraction of negative examples satisfy
#    clf.get_clf_binary_decision_func(X) >= bias
#    Parameters
#    ----------
#    df: array-like
#        Decision function evalued for negative examples
#    fpos : float
#        Maximum false positive fraction
#    tneg : int
#        Total number of negative examples (may be larger than df,
#        if prev  ind = int((1 - frac) * df.size)
#    return df[ind]ious negatives have been removed in previous versions
#        of the cascade)
#    ReturnsBased on code
#    -------
#    float
#    """
#    df.sort()
#
#    tiny = 1e-7
#    fpos = fpos * tneg / df.size
#
#    if fpos >= 1:
#        return df[0] - tiny
#
#    ind = max(min(int((1 - fpos) * df.size), df.size - 1), 0)
#
#    result = df[ind] + tiny
#    assert (df >= result).mean() <= fpos
#
#    return result

def _fpos_bias(df, fpos, tneg):
    """Choose a bias for a decision function, such that
    at most `frac` fraction of negative examples satisfy
    clf.get_clf_binary_decision_func(X) >= bias
    Parameters
    ----------
    df: array-like
        Decision function evalued for negative examples
    fpos : float
        Maximum false positive fraction
    tneg : int
        Total number of negative examples (may be larger than df,
        if prev  ind = int((1 - frac) * df.size)
    return df[ind]ious negatives have been removed in previous versions
        of the cascade)
    ReturnsBased on code
    -------
    float
    """
    df.sort()

    tiny = 1e-7
    #fpos = fpos * tneg / df.size

    if fpos >= 1:
        return df[0] - tiny

    ind = max(min(int((1 - fpos) * df.size), df.size - 1), 0)

    result = df[ind] + tiny
    assert (df >= result).mean() <= fpos

    return result


def _set_bias(clf, X, Y, recall, fpos, tneg):
    """Choose a bias for a classifier such that the classification
    rule
    clf.get_clf_binary_decision_func(X) - bias >= 0
    has a recall of at least `recall`, and (if possible) a false positive rate
    of at most `fpos`
    Paramters
    ---------
    clf : Classifier
        classifier to use
    X : array-like [M-examples x N-dimension]
        feature vectors
    Y : array [M-exmaples]
        Binary classification
    recall : float
        Minimum fractional recall
    fpos : float
        Desired Maximum fractional false positive rate
    tneg : int
        Total number of negative examples (including previously-filtered
        examples)
    """
    df = get_clf_binary_decision_func(clf, X, 1)
    r = _recall_bias(df[Y == 1], recall)
    f = _fpos_bias(df[Y == 0], fpos, tneg)
    return min(r, f)


def needs_fit(func):

    @wraps(func)
    def result(self, *args, **kwargs):
        if not hasattr(self, 'estimators_'):
            raise ValueError("Estimator not fitted, call `fit` "
                             "before making predictions")
        return func(self, *args, **kwargs)

    return result


class CascadedBooster(BaseEstimator, ClassifierMixin):
    def __init__(self, targetTP=0.99, targetFP=1e-3,
                 max_layers=1, verbose=1,
                 base_clf=None, weak_learner_params=None,
                 priors=[0.5,0.5]):
        """ A cascaded boosting classifier
        Based on the Viola and Jones (2001) Cascade
        A cascaded classifier aggregates multiple classifiers in
        sequence. Objects are classified as +1 only if every
        individual classifier classifies them as +1.
        Parameters
        ----------
        targetTP : float
            The minimum recall of training data at each layer
            of the cascade
        targetFP : float
            The desired false positive rate of the entire cascade
        mas_layers : int
            The maximum number of cascade layers to build during `fit`
        verbose : int
            Set verbose > 0 to print informational messages
        base_clf : Classifier instance (optional)
            The classifier object to clone and train at each layer of the
            cascade. Defaults to sklearn.ensemble.GradientBoostingClassifier()
        """

        weak_learner_params = weak_learner_params or {}
        if base_clf is None:
            base_clf = GradientBoostingClassifier(**weak_learner_params)

        self.base_clf = base_clf


        self.targetTP = targetTP
        self.targetFP = targetFP

        self.max_layers = max_layers

        self.converged = False

        self.verbose = verbose
        self.estimators_ = []
        self.bias_ = []
        self.priors = priors
        self.warm_start = False # used for incremental learning


        # these are to hold each layer validated rates
        self.layerValidatedTP = []
        self.layerValidatedFP = []
        self.stage_pos_pred = []
        self.stage_neg_pred = []

        self.cascadeTP = []
        self.cascadeFP = []
        self.featureLength = -1



    @needs_fit
    def staged_predict(self, X):
        """Predict classification of `X` for each iteration"""
        result = np.ones(X.shape[0], dtype=np.int)
        for b, c in zip(self.bias_, self.estimators_):
            result &= (get_clf_binary_decision_func(c, X , 1) >= b)
            yield result.copy()

    @needs_fit
    def staged_decision_function(self, X):
        """Compute decision function of `X` for each iteration"""
        result = np.zeros(X.shape[0])
        for c, b in zip(self.estimators_, self.bias_):
            good = result >= 0
            result[good] = get_clf_binary_decision_func(c, X[good], 1) - b
            yield result.copy()

    @needs_fit
    def decision_function(self, X):
        """Compute the decision function of `X`"""
        result = np.zeros(X.shape[0])
        for c, b in zip(self.estimators_, self.bias_):
            good = result >= 0
            result[good] = get_clf_binary_decision_func(c, X[good], 1) - b
        return result

    @needs_fit
    def predict_proba(self, X, P1=None, P2=None):
        """compute the prediction probability for x in two classes
        """
        if P1 is not None and P2 is not None and\
        (P1 !=self.priors[0] or P2 !=self.priors[1]):
            re_calculate_stage_prediction_values(P1, P2)
        else:
            P1 = self.priors[0]
            P2 = self.priors[1]
        #print "incoming Data shape", X.shape
        assert(self.featureLength == X.shape[1])

        #if(X.shape[0] == 1):
        #   print "it is better if you send lots of X

        prob = np.zeros([X.shape[0],2])
        result = np.zeros(X.shape[0])
        stage = 0
        ep = 1e-10
        for c, b in zip(self.estimators_, self.bias_):
            good = np.flatnonzero(result >= 0)
            #good = [i for result[i]>=0]
            if len(good)==0:
                break
            #print good
            good = np.reshape(good,[len(good),])
#            subset = X[good,:]
#            if (subset.shape[0] == 0):
#                print "good:", good.shape
#                print "X[good,:] ", X[good,:].shape
#                subset = np.reshape(subset, [1, self.featureLength])
#                print "subset shape", subset.shape
            decision_func = get_clf_binary_decision_func(c, X[good,:], 1)
            stage_score = decision_func - b
            result[good] = stage_score
            #stage_good = stage_score>=0
            #if stage == 0:
            #    prob[good[stage_good], 1] = (decision_func[stage_good]) * self.stage_pos_pred[stage]
            #else:
            #    prob[good[stage_good], 1] = prob[good[stage_good], 1]  + (decision_func[stage_good]) * self.stage_pos_pred[stage]
            # stage_pos = stage_score >= 0
            # stage_neg = stage_score < 0

#            for g in good:
#                if result[g]>=0:
#                    # prob[g, 1] = prob[g, 1] * (decision_func[j]) * self.stage_pos_pred[stage] * P1
#                    
#                    prob[g, 1] = prob[g, 1] +(decision_func[j]) * self.stage_pos_pred[stage] * P1
#                    # prob[g, 1] = prob[g, 1] * self.stage_pos_pred[stage] * P1
#                else:
#                    prob[g, 0] = prob[g, 0] +(1-decision_func[j]) * self.stage_neg_pred[stage] * P2
#                    # prob[g, 0] = prob[g, 0] * self.stage_neg_pred[stage] * P2
#                j = j + 1
            
            stage = stage + 1
            # print prob
        # normalizing so positive prob is in 0.5-1.0
        prob[:,1] = result * 0.5 + 0.5
        prob[:,0] = 1 - prob[:,1]
        return result, prob

    @needs_fit
    def re_calculate_stage_prediction_values(self, P1, P2):
        """ Assumming prior probabilities for two classes P1,P2
        different than self.priors
        """
        ar_tp = np.array(self.layerValidatedTP)
        ar_fp = np.array(self.layerValidatedFP)
        ln_stages = len(ar_tp)
        self.stage_pos_pred = np.zeros(ln_stages)
        self.stage_neg_pred = np.zeros(ln_stages)
        for i in range(0, ln_stages):
            self.stage_pos_pred[i] = P1*ar_tp[i] / (ar_tp[i] * P1 + ar_fp[i] * P2)
            self.stage_neg_pred[i] = P2*(1-ar_fp[i]) / ((1-ar_tp[i]) * P1 + (1-ar_fp[i]) * P2)




    def fit(self, poolX, poolY, valX=None, valY=None, val_ratio=0.25):
        """ Fit the cascaded boosting model """
        if self.warm_start:
            return self.add_cascade_layer(poolX, poolY, valX=None, valY=None, val_ratio=0.1)
        # poolXbkup = poolX.copy()
        # poolYbkup = poolY.copy()
            # note that this happens every turn
        if valX is None:
            noextravalidation = True

        F = [1.0]
        # self.bias_ = []
        # self.estimators_ = []

        # record feature length
        self.featureLength = poolX.shape[1]

        for i in range(self.max_layers):
            #assert self._check_invariants(X, Y, tpos, tneg, i)
            if noextravalidation:
                #val_ratio = 0.50 #  portion to take for validation
                # ratio will be used on this label will be,
                # samples from the other label will be taken equal.
                balance_label = None
                trnX, valX, trnY, valY = divideSetsRandomByRatio(poolX, poolY,
                                                             val_ratio,
                                                             balance_label)
            else:
                # there is val from outside so just copy pool to trn
                trnX, trnY = poolX.copy(), poolY.copy()

            nTrain = trnY.shape[0]
            nVal = valY.shape[0]

            nPos = trnY.sum()
            nNeg = nTrain - nPos

            nValPos = valY.sum()
            nValNeg = nVal - nValPos
            print "Training set: +", nPos, " -",nNeg
            print "Validation set: +", nValPos, " -",nValNeg

            if nNeg < MIN_SAMPLES or nPos < MIN_SAMPLES:
                print "stoping training due to num samples"
                self.converged = False
                break

            if F[-1] < self.targetFP:
                self.converged = True
                break

            if np.unique(trnY).size == 1:  # down to a single class
                self.converged = True
                break

            F.append(F[-1])
            clf = clone(self.base_clf)
            if ADAPTIVENESTIMATORS:
                nest = np.int(np.log(nPos+nNeg))
                print "Training n ", nest, " estimators "
                clf.set_params(n_estimators=nest)

            clf.fit(trnX, trnY)

            bias = _set_bias(clf, valX, valY, self.targetTP,
                             self.targetFP, nValNeg)
            self.estimators_.append(clf)
            self.bias_.append(bias)
            print "Decision th: ", bias
            
            print self

            Yp = get_clf_binary_decision_func(clf, valX, 1) >= bias
            pr = 1.0 * ((valY == 0) & Yp).sum() / nValNeg
            rc = 1.0 * ((valY == 1) & Yp).sum() / nValPos
            if self.verbose > 0:
                print ("Cascade round %i. False pos rate: %1.2f. "
                       "Recall: %1.2f" % (i + 1, pr, rc))
            # record everything
            self.layerValidatedTP.append(rc)
            self.layerValidatedFP.append(pr)
            # calculate prediction values
            P1 = self.priors[0]
            P2 = self.priors[1]
            ep = 1e-10
            self.stage_pos_pred.append(P1*rc / (rc * P1 + pr * P2 + ep))
            self.stage_neg_pred.append(P2*(1-pr) / ((1-rc) * P1 + (1-pr) * P2)+ep)
            # test whole levels
            Yp = self.predict(poolX)
            F[-1] = 1.0 * ((poolY == 0) & Yp).sum() / (nNeg + nValNeg)
            RC = 1.0 * ((poolY == 1) & Yp).sum() / (nPos + nValPos)

            self.cascadeTP.append(RC)
            self.cascadeFP.append(F[-1])

            if self.verbose > 0:
                print "TPtrack:", self.cascadeTP
                print "FPtrack:", self.cascadeFP

            if Yp.all():
                warn("Could not filter any more examples. "
                     "False positive rate: %1.2f. Recall: %1.2f" % (F[-1], RC))
                self.converged = False
                break

            print self
            # take only positive samples....
            poolX = poolX[Yp]
            poolY = poolY[Yp]
        else:
            warn("Could not reach target false positive rate enough after "
                 "%i layers. False positive rate: %1.2f. Recall: %1.2f" %
                 (self.max_layers, F[-1], RC))
            self.converged = False

        return self

    def __str__ (self):
        sent = ""
        sent = sent +("Cascade of " + str(len(self.estimators_))+" classifiers \n")
#        k = 1
#        for e, tp, fp in zip(self.estimators_, self.cascadeTP,
#                     self.cascadeFP):
#               (sent+ "Classifier " +str(k) + " Est: "+
#               str((e.n_estimators))+
#               "TP: "+str(tp) + "FP: "+ str(fp))
#               k = k+1
        return sent

     
    def add_layer(self, X, Y, valX, valY, fitParam=None):

        nTrain = trnY.shape[0]
        nVal = valY.shape[0]
        nPos = trnY.sum()
        nNeg = nTrain - nPos
        
        print "Validation set: +", nValPos, " -",nValNeg       
        clf = clone(self.base_clf)
        if ADAPTIVENESTIMATORS:
            nest = np.int(np.log(nPos+nNeg))
            print "Training n ", nest, " estimators "
            clf.set_params(n_estimators=nest)
        
        clf.fit(trnX, trnY)

        bias = _set_bias(clf, valX, valY, self.targetTP,
                             self.targetFP, nTotNeg)
        print bias
        self.estimators_.append(clf)
        self.bias_.append(bias)
        print self

        
        
    def add_cascade_layer(self, X, Y, valX=None, valY=None, val_ratio=0.2, fitParam=None):
        """Add another layer to the cascade.
        The new layer will achieve a recall of at least `targetTP`
        on the input data. It will not apply any false positive criteria
        """
        fitParam = fitParam or {}
        poolX, poolY = X.copy(), Y.copy()
        if(self.featureLength != poolX.shape[1]):
            print poolX.shape

        if valX is not None:
            valX = valX
            valY = valY
        else:
            #val_ratio = val_ratio
            # ratio will be used on this label will be,
            # samples from the other label will be taken equal.
            balance_label = 1
        trnX, valX, trnY, valY = divideSetsRandomByRatio(poolX, poolY,
                                                         val_ratio,
                                                         balance_label)

        nTrain = trnY.shape[0]
        nVal = valY.shape[0]

        nPos = trnY.sum()
        nNeg = nTrain - nPos

        nValPos = valY.sum()
        nValNeg = nVal - nValPos

        print "Training set: +", nPos, " -",nNeg
        print "Validation set: +", nValPos, " -", nValNeg


        clf = clone(self.base_clf)

        if ADAPTIVENESTIMATORS:
            nest = np.int(np.ceil((np.log(nPos+nNeg)/2)))
            if nest<1: 
                nest = 0
                print "Number of examples too low, training cancelled"
                return
            print "Training n ", nest, " estimators "
            clf.set_params(n_estimators=nest)
        
        clf.fit(trnX, trnY)
        bias = _set_bias(clf, valX, valY, self.targetTP,
                             self.targetFP, nValNeg)
        print "Decision th: ", bias

        self.estimators_.append(clf)
        self.bias_.append(bias)
        print self

        Yp = get_clf_binary_decision_func(clf, valX, 1) >= bias
        pr = 1.0 * ((valY == 0) & Yp).sum() / nValNeg
        rc = 1.0 * ((valY == 1) & Yp).sum() / nValPos
        if self.verbose > 0:
            print ("Added Layer False pos rate: %1.2f. "
                       "Recall: %1.2f" % (pr, rc))
            # record everything
        self.layerValidatedTP.append(rc)
        self.layerValidatedFP.append(pr)
        # calculate prediction values
        P1 = self.priors[0]
        P2 = self.priors[1]
        ep = 1e-10
        self.stage_pos_pred.append(P1* rc / (rc * P1 + pr * P2 + ep))
        self.stage_neg_pred.append(P2* (1-pr) / ((1-rc) * P1 + (1-pr) * P2)+ep)

        return self


    def pop_cascade_layer(self):
        """Remove one layer from the cascade"""
        if not hasattr(self, 'estimators_') or len(self.estimators_) == 0:
            raise IndexError("No cascade layers to remove")
        self.estimators_.pop()
        self.bias_.pop()
        self.converged = False

    @needs_fit
    def predict(self, X):
        """Predict class for X"""
        return self.decision_function(X).ravel() >= 0


def get_clf_binary_decision_func(clf, X, label=None):
    """ not all classifiers have a decision function..
    this one returns, probability of predicting label
    """
    if label is None:
        return (clf.predict_proba(X))
    else:
        return (clf.predict_proba(X)[:,label]).ravel()


