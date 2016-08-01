# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:16:52 2016

@author: btek
"""
from sklearn.externals import joblib
from scipy.misc import imresize, imread
import numpy as np
from os import listdir
from os.path import isfile, join, splitext, dirname
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
import sampleFactory


# constants
ROOTFOLDER = '/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/'
IMGLIST = 'mitosisData/amida_train/listjpgs.txt'
CSVLIST = 'mitosisData/amida_train/listcsvs.txt'
CLASSIFIERFILE_INIT = 'random_forest_trained.pkl'
CLASSIFIERFILE_CONT = 'random_forest_trained_e2.pkl'
WORKMODE = 'TESTANDCOLLECTSAMPLES'
WINDOWSIZE = 50
# roughly 5mm for amida??, not sure about this.
DISTANCETHRESHOLD = 20
DEBUGPLOT = False
AUGMENTPOSITIVESAMPLES = True
SAMPLE_MULTIPLIER = 20  # number of sample augmentation for each sample
NSAMPLETRAININGINTERVAL = 10
TRAININGMODE = 0 


# load my classifier
def loadClassifier(filename):
    clf = joblib.load(filename)
    print(clf)

    return clf


# train the classifier with new DAta. warm start, increase estimators by one.
# this function actually adds newEstimators trees to the forest.
def trainIncrementally(classifier, X, y, newEstimators=1):
    classifier.set_params(warm_start=True,
                          n_estimators=classifier.n_estimators+newEstimators, 
                          oob_score=True)
    classifier.fit(X, y)
    # return the classifier itself and the oob error
    return classifier, 1 - classifier.oob_score


# load image
def loadImage(filename):
    img = imread(filename)
    img = np.asarray(img, dtype='float32')
    return img


# load ground truth
def loadCSV(filename):
    lines = open(filename).readlines()
    gtlist = np.zeros([len(lines), 2])
    k = 0
    for strline in lines:
        x, y = strline.rstrip('\n').split(",")
        gtlist[k, :] = map(int, [x, y])
        k += 1

    return gtlist


# load image file list
def loadFileList(filename):
    lines = open(filename).readlines()
    rootfolder = dirname(filename)+'/'
    return lines, rootfolder


# create list of jpgs in a folder
def enumerateJPGFromFolder(mypath):
    onlyjpgs = [f for f in listdir(mypath) if (isfile(join(mypath, f)) &  splitext(f)[1]== '.jpg')]
    return onlyjpgs


# create multiscales
def prepareMultiScale(img, scales):
    scaled_images = []
    for s in scales:
        im_scaled = imresize(img, s)
        scaled_images.append(im_scaled)
    return scaled_images


# scan the image with stepSize separated windows 2 or 3d
def slidingWindow(classifier, img, windowSize, stepSize=1):
    ndim = np.ndim(img)
    if ndim == 4:
        print("not implemented yet")
        return None
    elif ndim == 3:
        return slidingWindow2D(classifier, img, windowSize, stepSize)
    else:
        return slidingWindow2D(classifier, img, windowSize, stepSize)


def slidingWindow2D(classifier, img, windowSize, stepSize=1):

    margin_row = windowSize/2
    margin_col = windowSize/2
    ndim = np.ndim(img)
    
    if ndim == 3:
        nrows, ncols, nchannel = np.shape(img)
    else:
        nrows, ncols = np.shape(img)

    outputBinary = np.zeros([nrows, ncols])
    outputProbability = np.zeros([nrows, ncols])
    for r in range(margin_row, nrows-margin_row, stepSize):
        for c in range(margin_col, ncols-margin_col, stepSize):
            windowimg = (img[r - windowSize/2:r + windowSize/2, c -
                         windowSize/2:c + windowSize/2, :])
            # print np.shape(windowimg)
            prob = classifier.predict_proba(windowimg.reshape([1, windowSize*windowSize*nchannel]))[0]
            #print prob
            outputProbability[r, c] = max(prob)
            outputBinary[r, c] = np.argmax(prob)
            #print r

    return outputBinary, outputProbability


#####################################
def distancetoSet(mCord, setCoord):
    mx = mCord[0]
    my = mCord[1]
    # this thing is a list, gtlist convert it to array
    setCoord = np.asarray(setCoord)
    # print setCoord, np.shape(setCoord)
    setX =setCoord[:,0]
    setY =setCoord[:,1]
    difx = (mx-setX)*(mx-setX)
    dify = (my-setY)*(my-setY)
    dif_l2 = np.sqrt(difx+dify)
    return dif_l2


#####################################
def labelAndEvaluateOutput(binaryOutput, gtlist):
    """
        function groups detections and compares against gtlist.

        Keyword arguments:
        binaryOutput -- detection of the classifier, binary
        gtlist -- a list of ground truth locations in (x,y) format

        Returns
        tp --
        fp --
        fn --slidingWindow
    """
    labeledImg = label(binaryOutput, background =0)
    calcMeasures = regionprops(labeledImg, cache = True)
        
    listcentroids = np.array([prop.centroid for prop in calcMeasures], dtype='int')
    ndetection = len(listcentroids)
    print "number of detection", ndetection
    truePos = []
    trueHit = []
    falsePos = []
    falseNeg = []
    missedGTIndex = range(0, len(gtlist))
    for detectedCentroid in listcentroids:
        #print detectedCentroid
        distance2GTList = distancetoSet(detectedCentroid,gtlist)
        ##distanceBelowThreshold = distance2GTList < DISTANCETHRESHOLD
        u = np.argmin(distance2GTList)
        #print u
        if distance2GTList[u] < DISTANCETHRESHOLD:
            
            #print 'detected centroid',u
            #print 'missed ix',missedGTIndex
            if u in missedGTIndex:
                truePos.append(detectedCentroid)           
                missedGTIndex.remove(u)  ## we remove the detected from the missed index
            else:
                trueHit.append(detectedCentroid)
                print "gt {0} allready detected".format(u)
#            TODODODODODODODOODOD HERE 
            
        elif distance2GTList[u] > DISTANCETHRESHOLD*2:# put a large margin around the gt. 
            falsePos.append(detectedCentroid)
        
    falseNeg = [gtlist[i] for i in missedGTIndex ]
    return trueHit,truePos,falsePos,falseNeg
    
    
############################################################################
def overlayandPlotEvalution(img, tp = [], hit = [], falsepos = [], missed = [], rectW = WINDOWSIZE, rectH = WINDOWSIZE ):
    # all of them are supposed to be a list of arrays as coords x,y
    fig = plt.figure()
    ax = fig.gca()
    ax.imshow(np.uint8((img+1.0)*127.5))
    hW = rectW/2
    hH = rectH/2
    for pt in tp:
        minc = pt[1]-hW
        minr = pt[0]-hH
                
        rect = mpatches.Rectangle((minc, minr), rectH, rectW,
                              fill=False, edgecolor='green', linewidth=3)
        ax.add_patch(rect)
        
    for pt in hit:
        minc = pt[1] - hH
        minr = pt[0] - hW
                
        rect = mpatches.Rectangle((minc, minr), rectH, rectW,
                              fill=False, edgecolor='blue', linewidth=2, linestyle = 'dotted')
        ax.add_patch(rect)
    reduced_set = np.random.permutation(len(falsepos))
    
    nDrawFalse = 200
    if len(reduced_set)>nDrawFalse:
        reduced_set = reduced_set[0:nDrawFalse].squeeze().astype(int)
    # print reduced_set.shape
    false_reduced = [falsepos[i] for i in reduced_set]
    for pt in false_reduced:
        minc = pt[1]-2
        minr = pt[0]-2
                
        rect = mpatches.Rectangle((minc, minr), 8, 8,
                              fill=False, edgecolor='cyan', linewidth=1)
        ax.add_patch(rect)
        
    for pt in missed:
        minc = pt[1]-hH
        minr = pt[0]-hW
                
        rect = mpatches.Rectangle((minc, minr), rectH, rectW,
                              fill=False, edgecolor='yellow', linewidth=3, linestyle = 'dashed')
        ax.add_patch(rect)
    ax.annotate('TP', xy=(2000,50), color='green', weight='heavy', size='large' )
    ax.annotate('Hit', xy=(2000,150), color='blue', weight='heavy', size='large')
    ax.annotate('False', xy=(2000,250), color='cyan', weight='heavy', size='large')
    ax.annotate('Missed', xy=(2000,350), color='yellow', weight='heavy', size='large')    
    plt.show()

#############################################################################
def collectSamples(img, posList = [], negList = [], rectW = WINDOWSIZE, 
                   rectH = WINDOWSIZE, multiScale=[1.0]):

    posWindows =[]
    negWindows=[]
    p = 0
    for m in multiScale:
        hW = np.int(rectW/2 * m)
        hH = np.int(rectH/2 * m)
    
        for pt in posList:
            # print pt
            minc = np.int(pt[1] - hH)
            minr = np.int(pt[0] - hW)
            wind = img[minr:minr+2*hH, minc:minc+2*hW, :]
            # check if crop was full
            if wind.shape != (rectH, rectW, 3):
                # print "Window resized", wind.shape
                wind2 = imresize(wind, [rectH, rectW])
#                plt.imshow(wind2)
#                plt.show()
                wind = wind2
#            if m != 1.0:
#                wind = imresize(wind, [rectH, rectW])
            
            posWindows.append(wind)
            p = p + 1
        
    

#        if DEBUGPLOT:
#
#            plt.figure(12)
#        # plt.gray()
#            plt.imshow(posWindows[p-1])
#            plt.show(block=False)
            # print minr, minr+rectH
            # print minc, minc+rectW

    print 'Collected', p, ' positive samples', 'in ', len(multiScale), ' scales'
    for pt in negList:
        hW = np.int(rectW/2)
        hH = np.int(rectH/2)
        minc = pt[1] - hH
        minr = pt[0] - hW
        wind = img[minr:minr+rectW, minc:minc+rectH, :]
        negWindows.append(wind)
        p = p+1

    print 'Collected', p, ' total samples'
    return posWindows, negWindows


#################################################################
# take the winodows, augment, create labels
def createTrainSet(posWinds, negWinds, augmentpos=False):
    if augmentpos:
        posWinds = augmentSamples(posWinds)
    #posWinds = np.asarray(posWinds)
    #negWinds = np.asarray(negWinds)
    npos = np.shape(posWinds)[0]
    nneg = np.shape(negWinds)[0]
    wind = posWinds[0]
    windlen = np.shape(wind)[0] * np.shape(wind)[1] * np.shape(wind)[2]
    #print len(posWinds)
    # print [npos, windlen]
    Xpos = np.zeros([npos, windlen])
    k = 0 
    for w in posWinds:
        # print w.shape
        Xpos[k,:] = w.reshape([1, windlen])
        k = k + 1
    Xneg = np.zeros([nneg, windlen])
    k = 0 
    for w in negWinds:
        Xneg[k,:] = w.reshape([1, windlen])
        k = k + 1
    #Xpos = np.asarray(np.reshape(posWinds, [npos, windlen]))
    #Xneg = np.asarray(np.reshape(negWinds, [nneg, windlen]))
    X = np.concatenate((Xpos, Xneg), axis=0)

    y = np.ones(npos)
    y = np.concatenate((y, np.zeros(nneg)))

    return X, y


#############################################################
def augmentSamples(sampleWindows):
    return sampleFactory.generateSamples(sampleWindows, SAMPLE_MULTIPLIER)
#############################################################

# main #####################################################
clf = loadClassifier(CLASSIFIERFILE_INIT)

# take the list of jpgs in the folder
listjpg, rootjpg = loadFileList(IMGLIST)
listcsv, rootcsv = loadFileList(CSVLIST)
# print listjpg, listcsv

# these lists accumulate samples collected from each image

# this counts newly collected samples from the last training
newSamples = 0
sampleCollectStepSizeList = [100, 50, 25, 20, 10, 5]
for stpSize in sampleCollectStepSizeList:
    k = 1
    allCollectedSampleSetX = []
    allCollectedSampleLabels = []
    listnTp = []
    listnFp = []
    listnGT = []
    newSamples = 0


    for f, g in zip(listjpg, listcsv):
        # load the ground truth marks
        gtforfile = loadCSV(rootcsv+g.rstrip('\n'))
        gtforfile = list(gtforfile)
        print f, g
        print "Wındow step:", stpSize
        imInput = loadImage(rootjpg + f.rstrip('\n'))
        # normalize image to [0-1.0]
        imInput = sampleFactory.normRGB(imInput)
    
        # classify each window with stepSize
        imBinary, imProb = slidingWindow(clf, imInput, WINDOWSIZE, stepSize=stpSize)
    
        # label detections as true false pos
        hit, truepos, falsepos, falseneg = labelAndEvaluateOutput(
                                                                  imBinary,
                                                                  gtforfile)
        print "tp {0}, fp {1}, fn {2}, gt {3}, hit-{4}".format(len(truepos), len(falsepos), len(falseneg), len(gtforfile), len(hit))
        listnTp.append(len(truepos))
        listnFp.append(len(falsepos))
        listnGT.append(len(gtforfile))
    
        overlayandPlotEvalution(imInput, truepos, hit, falsepos, falseneg)
    
        # merge detections with the gt
        posList = gtforfile + truepos
        newPos, newNeg = collectSamples(imInput, posList, falsepos,
                                        multiScale=[1.0, 1.1, 1.2])
    
        trainsetx, trainsety = createTrainSet(newPos, newNeg, AUGMENTPOSITIVESAMPLES)
    
    #   record new sample length
        newSamples = newSamples + len(trainsety)
    
        if k == 1:
            allCollectedSampleSetX = np.copy(trainsetx)
            allCollectedSampleLabels = np.copy(trainsety)
        else:
            allCollectedSampleSetX = np.concatenate((allCollectedSampleSetX, trainsetx), axis=0)
            allCollectedSampleLabels = np.concatenate((allCollectedSampleLabels, trainsety), axis=0)
    
    #   record new samples to the file
        _ = joblib.dump(allCollectedSampleSetX,'bootstrap_collected_sampleX.pkl', compress=9)
        _ = joblib.dump(allCollectedSampleLabels, 'bootstrap_collected_sampleY.pkl', compress=9)
    
        if TRAININGMODE == 0:
            # we do not train
            # training will be done after the loop
            print "Samples are stored"
    
        elif TRAININGMODE == 1:
            print "Training X: ", np.shape(trainsetx)
            print "Training y: ", np.shape(trainsety)
            clf, oob_error = trainIncrementally(clf, trainsetx, trainsety)
    
        elif TRAININGMODE == 2 and newSamples > NSAMPLETRAININGINTERVAL:
    
            print "Training X: ", np.shape(allCollectedSampleSetX)
            print "Training y: ", np.shape(allCollectedSampleLabels)
            clf, oob_error = trainIncrementally(clf,
                                                allCollectedSampleSetX,
                                                allCollectedSampleLabels)
    
    #       reset sample counter
            newSamples = 0
        # now use new samples to train our classifier
    
        # leave time to recognize keyboard interrupt
        sleep(1)
    
        if k == 5:
            # calculate the performance
            TPrate = np.sum(np.array(listnTp)) / np.sum(np.array(listnGT))
            FPpIMG = np.sum(np.array(listnFp)) / k
            break
        k = k+1


    if TRAININGMODE == 0:
        # training will be performed after loop ends
        print "Training X: ", np.shape(allCollectedSampleSetX)
        print "npos samples ", np.sum(allCollectedSampleLabels==1)
        print "nneg samples ", np.sum(allCollectedSampleLabels==0)
                                
    
        clf, oob_error = trainIncrementally(clf,
                                            allCollectedSampleSetX,
                                            allCollectedSampleLabels,
                                            newEstimators=3)
        print "Trained Error {0}".format(oob_error)
    
    # save the training
    _ = joblib.dump(clf, CLASSIFIERFILE_CONT, compress=9)
    print "trained", clf
    sleep(10)
