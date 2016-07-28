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
CLASSIFIERFILE = 'random_forest_trained.pkl'
WORKMODE = 'TESTANDCOLLECTSAMPLES'
WINDOWSIZE = 50
# roughly 5mm for amida??, not sure about this.
DISTANCETHRESHOLD = 20
DEBUGPLOT = False
AUGMENTPOSITIVESAMPLES = True
SAMPLE_MULTIPLIER = 20  # number of sample augmentation for each sample


# load my classifier
def loadClassifier(filename):
    clf = joblib.load(filename)
    print(clf)

    return clf


# train the classifier with new DAta. warm start, increase estimators by one.
# this function actually adds one single tree to the forest.
def trainIncrementally(classifier, X, y):
    classifier.set_params(warm_start=True,
                          n_estimators=classifier.n_estimators+1, 
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
    print reduced_set.shape
    false_reduced = [falsepos[i] for i in reduced_set]
    for pt in false_reduced:
        minc = pt[1]-2
        minr = pt[0]-2
                
        rect = mpatches.Rectangle((minc, minr), 4, 4,
                              fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        
    for pt in missed:
        minc = pt[1]-hH
        minr = pt[0]-hW
                
        rect = mpatches.Rectangle((minc, minr), rectH, rectW,
                              fill=False, edgecolor='yellow', linewidth=3, linestyle = 'dashed')
        ax.add_patch(rect)
    ax.annotate('TP', xy=(2000,50), color='green', weight='heavy', size='large' )
    ax.annotate('Hit', xy=(2000,150), color='blue', weight='heavy', size='large')
    ax.annotate('False', xy=(2000,250), color='red', weight='heavy', size='large')
    ax.annotate('Missed', xy=(2000,350), color='yellow', weight='heavy', size='large')    
    plt.show()

#############################################################################
def collectSamples(img, posList = [], negList = [], rectW = WINDOWSIZE, rectH = WINDOWSIZE):

    posWindows =[]
    negWindows=[]
    hW = np.int(rectW/2)
    hH = np.int(rectH/2)
    p = 0
    for pt in posList:
        print pt
        minc = np.int(pt[1] - hH)
        minr = np.int(pt[0] - hW)
        wind = img[minr:minr+rectH, minc:minc+rectW, :]
        # check if crop was full
        if wind.shape != (rectH, rectW, 3):
            print "Window resized", wind.shape
            wind2 = imresize(wind, [rectH, rectW])
            plt.imshow(wind2)
            plt.show()
            wind = wind2
            
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

    print 'Collected', p, ' positive samples'
    for pt in negList:
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
    # print len(posWinds)
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


#############################################################################
def augmentSamples(sampleWindows):
    return sampleFactory.generateSamples(sampleWindows, SAMPLE_MULTIPLIER)
#############################################################################




# main #############################################################################
clf = loadClassifier(CLASSIFIERFILE)

# take the list of jpgs in the folder
listjpg, rootjpg = loadFileList(IMGLIST)
listcsv, rootcsv = loadFileList(CSVLIST)
# print listjpg, listcsv
k = 1
for f, g in zip(listjpg, listcsv):
    
    gtforfile = loadCSV(rootcsv+g.rstrip('\n'))
    gtforfile = list(gtforfile)
    print f, g
    imInput = loadImage(rootjpg + f.rstrip('\n'))
    imInput = sampleFactory.normRGB(imInput)

    # classify each window with stepSize
    imBinary, imProb = slidingWindow(clf, imInput, WINDOWSIZE, stepSize=10)

    # label detections as true false pos
    hit, truepos, falsepos, falseneg = labelAndEvaluateOutput(imBinary, gtforfile)
    print "tp {0}, fp {1}, fn {2}, gt {3}, hit-{4}".format(len(truepos), len(falsepos), len(falseneg), len(gtforfile), len(hit))
#    if DEBUGPLOT:
#        plt.figure(1)
#        plt.gray()
#        plt.imshow(imOutput)
#        plt.show(block=True)
#        input()

    # plot truepos and some random falsepositives
    overlayandPlotEvalution(imInput, truepos, hit, falsepos, falseneg)
    
    # merge detections with the gt
    posList = gtforfile + truepos
    newPos, newNeg = collectSamples(imInput, posList, falsepos)

    trainsetx, trainsety = createTrainSet(newPos, newNeg, AUGMENTPOSITIVESAMPLES)

    # now use new samples to train our classifier
    clf, oob_error = trainIncrementally(clf, trainsetx, trainsety)
    print "Training Error {0}".format(oob_error)
    # newPos = augmentSamples(newPos)


#    if k%2==0:
#        s= raw_input('continue? y or n')
#    else:
#        s = ''
#    if((s == 'n') | (s =='0')):
#        break
    #w =raw_input("PRESS ENTER TO CONTINUE.")
    
    sleep(10)
    k = k+1
    if k == 3:
       
        break
