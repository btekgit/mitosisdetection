# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:16:52 2016

@author: btek
"""
from sklearn.externals import joblib
from scipy.misc import imread
from skimage.transform import resize
import numpy as np
from os import listdir
from os.path import isfile, join, splitext, dirname
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import sleep
import sampleFactory
from listOps import listDiff


# constants
ROOTFOLDER = '/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/'
IMGLIST = 'mitosisData/amida_train/listjpgs.txt'
CSVLIST = 'mitosisData/amida_train/listcsvs.txt'
#CLASSIFIERFILE_INIT = 'random_forest_trained_e3.pkl'
#CLASSIFIERFILE_INIT = 'random_forest_cascade_trained.pkl'
#CLASSIFIERFILE_CONT = 'random_forest_casceade_trained_e2.pkl'
#CLASSIFIERFILE_CONT = 'random_forest_trained_e4.pkl'
WORKMODE = 'TESTANDCOLLECTSAMPLES'
CASCADEMODE = True
WINDOWSIZE = 50
# roughly 5mm for amida??, not sure about this.
DISTANCETHRESHOLD = 20
DEBUGPLOT = False
AUGMENTPOSITIVESAMPLES = True
SAMPLE_MULTIPLIER = 20  # number of sample augmentation for each sample
NSAMPLETRAININGINTERVAL = 10
TRAININGMODE = 0
STORESAMPLES = False
NUMBEROFTRAININGIMAGES = 40
MINIMUMNEGATIVESAMPLESTOTRAIN = 100

#SCANBYROWS = True # if False classifier is called for each window separately


# load my classifier
def loadClassifier(filename):
    clf = joblib.load(filename)
    print(clf)

    return clf


# train the classifier with new DAta. warm start, increase estimators by one.
# this function actually adds newEstimators trees to the forest.
def trainIncrementally(classifier, X, y, newEstimators=-1):
    if newEstimators<=0:
        nest = np.int(np.log(len(y))/2)
        newEstimators = max(nest, 1)
        print "Training n ", nest, " estimators "
        #clf.set_params(n_estimators=nest)
    
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
    lines = [rootfolder+f for f in lines]
    return lines


# separate Train test. Training set all have gt
def separateTrainTest(listcsv, listjpg):
    randIndex = np.random.permutation(len(listcsv))
    randNTrain = randIndex[0:NUMBEROFTRAININGIMAGES]
    traincsv = [listcsv[i] for i in randNTrain]
    # now select the images
    trainjpg = []
    jpgext = '.jpg\n'
    for csv in traincsv:
        csvimg, csvext = str.split(csv, '.')
        a = csvimg+jpgext
        # print a
        # print listjpg
        i = listjpg.index(a)
        trainjpg.append(listjpg[i])
    testcsv = listDiff(listcsv, traincsv)
    testjpg = listDiff(listjpg, trainjpg)
    return traincsv, trainjpg, testcsv, testjpg


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
        col_ix = []
        col_winds =[]        
        for c in range(margin_col, ncols-margin_col, stepSize):
            windowimg = (img[r - windowSize/2:r + windowSize/2, c -
                         windowSize/2:c + windowSize/2, :])
            window_reshape = windowimg.reshape([1, -1])
            col_ix.append(c)
            col_winds.append(window_reshape)
            
        col_winds = np.reshape(np.array(col_winds),[len(col_winds),-1])
        col_ix = np.array(col_ix)
        #print col_winds.shape
        #print col_ix.shape
        prob = classifier.predict_proba(col_winds) 
        outputProbability[r, col_ix] = prob[:, 1]
        outputBinary[r, col_ix] = np.argmax(prob, axis = 1)


#####################################
def distancetoSet(mCord, setCoord):
    # coordinates come in Y,X
    my = np.float32(mCord[0])
    mx = np.float32(mCord[1])
    # this thing is a list, gtlist convert it to array
    setCoord = np.asarray(setCoord)
    # print setCoord, np.shape(setCoord)
    setY = np.float32(setCoord[:,0])
    setX = np.float32(setCoord[:,1])

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
        fn --
    """
    labeledImg = label(binaryOutput, background =0)
    calcMeasures = regionprops(labeledImg, cache = True)

    listcentroids = np.array([prop.centroid for prop in calcMeasures], dtype='int')
    # centroids come Y, X
    ndetection = len(listcentroids)
    # print "number of detection", ndetection
    # print "list detections", listcentroids
    truePos = []
    trueHit = []
    falsePos = []
    falseNeg = []
    missedGTIndex = range(0, len(gtlist))

    for detectedCentroid in listcentroids:
        # print detectedCentroid
        distance2GTList = distancetoSet(detectedCentroid, gtlist)
        ##distanceBelowThreshold = distance2GTList < DISTANCETHRESHOLD
        u = np.argmin(distance2GTList)
        # print distance2GTList
        if distance2GTList[u] < DISTANCETHRESHOLD:

            #print 'detected centroid',u
            #print 'missed ix',missedGTIndex
            if u in missedGTIndex:
                truePos.append(detectedCentroid)
                missedGTIndex.remove(u)  ## we remove the detected from the missed index
            else:
                trueHit.append(detectedCentroid)
                print "gt {0} allready detected".format(u)
            # TODO HERE

        elif distance2GTList[u] > DISTANCETHRESHOLD * 2: # put a large margin around the gt.
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

        rect = mpatches.Rectangle((minc, minr), 6, 6,
                              fill=True, edgecolor='cyan', linewidth=1)
        ax.add_patch(rect)

    for pt in missed:
        minc = pt[1]-hH
        minr = pt[0]-hW

        rect = mpatches.Rectangle((minc, minr), rectH, rectW,
                              fill=False, edgecolor='yellow', linewidth=3, linestyle = 'dashed')
        ax.add_patch(rect)
    ax.annotate('TP', xy=(2000, 50), color='green', weight='heavy', size='large' )
    ax.annotate('Hit', xy=(2000, 150), color='blue', weight='heavy', size='large')
    ax.annotate('False', xy=(2000, 250), color='cyan', weight='heavy', size='large')
    ax.annotate('Missed', xy=(2000, 350), color='yellow', weight='heavy', size='large')
    plt.show()

#############################################################################
def collectSamples(img, posList = [], negList = [], rectW = WINDOWSIZE,
                   rectH = WINDOWSIZE, multiScale=[1.0]):

    posWindows = []
    negWindows = []
    p = 0
    for m in multiScale:
        hW = np.int(rectW/2 * m)
        hH = np.int(rectH/2 * m)

        for pt in posList:
            # print pt
            if ((pt[0] < 5 and pt[0] > img.shape[0]-5) or
                (pt[1] < 5 and pt[1] > img.shape[1]-5)):
                print "skipping point too close to border:", pt
                continue

            minc = np.int(pt[1] - hW)  # this is x
            minr = np.int(pt[0] - hH)  # this is y
            minr = np.clip(minr, 0, img.shape[0])  # make sure it is positive
            minc = np.clip(minc, 0, img.shape[1])  # make sure it is positive

            wind = img[minr:minr+2*hH, minc:minc+2*hW, :]
            # print "Mean wind:", np.mean(wind), np.max(wind)
            # check if crop was full
            if wind.shape != (rectH, rectW, 3):
                wind2 = resize(wind, [rectH, rectW])
#                plt.imshow(wind2)
#                plt.show()
                wind = wind2
                # print "Mean wind:", np.mean(wind), np.max(wind)

            posWindows.append(wind)
            p = p + 1


    print 'Collected', p, ' positive samples', 'in ', len(multiScale), ' scales'
    for pt in negList:
        hW = np.int(rectW/2)
        hH = np.int(rectH/2)
        minc = np.int(pt[1] - hW)  # this is x
        minr = np.int(pt[0] - hH)  # this is y
        minr = np.clip(minr, 0, img.shape[0])  # make sure it is positive
        minc = np.clip(minc, 0, img.shape[1])  # make sure it is positive
        wind = img[minr:minr+2*hH, minc:minc+2*hW, :]
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


def testImages(clf, testjpg, testcsv=None, stepSize=1, scales=[1.0],
               plotResults=True, saveOutput=False):
    
    listnTp = []
    listnFp = []
    listnGT = []

    for k in range(0, len(testjpg)):
        f = testjpg[k]
        print f
        print "Wındow step:", stepSize
        imInput = loadImage(f.rstrip('\n'))
        # normalize image to [0-1.0]
        imInput = sampleFactory.normRGB(imInput)
        
        imBinary, imProb = slidingWindow(clf,
                                         imInput, WINDOWSIZE, stepSize=stepSize)

        if testcsv:
            g = testcsv[k]
            gtforfile = loadCSV(g.rstrip('\n'))
            gtforfile = list(gtforfile)        
            hit, truepos, falsepos, falseneg =labelAndEvaluateOutput(imBinary,
                                                                 gtforfile)
            print "tp {0}, fp {1}, fn {2}, gt {3}, hit-{4}".format(len(truepos),
                                                            len(falsepos),
                                                            len(falseneg),
                                                            len(gtforfile),
                                                            len(hit))
            listnTp.append(len(truepos))
            listnFp.append(len(falsepos))
            listnGT.append(len(gtforfile))
            if plotResults:
                overlayandPlotEvalution(imInput, truepos, hit, falsepos, falseneg)

        if saveOutput:
            print "not ready yet"
            
    Tprate = float(np.sum(np.array(listnTp))) / float(np.sum(np.array(listnGT)))
    FPpIMG = np.sum(np.array(listnFp)) / float(k)
    print "TP:", TPrate
    print "FP:", FPpIMG
    return listnTp, listnFp
    
# main #####################################################
clf = loadClassifier(CLASSIFIERFILE_INIT)

np.random.seed(0)
# take the list of jpgs in the folder
listjpg = loadFileList(IMGLIST)
listcsv = loadFileList(CSVLIST)
traincsv, trainjpg, testcsv, testjpg = separateTrainTest(listcsv, listjpg)


# print listjpg, listcsv

# these lists accumulate samples collected from each image

# this counts newly collected samples from the last training
nPositiveSamples = 0
nNegativeSamples = 0
# sampleCollectStepSizeList = [100, 50, 25, 20, 10, 5]
# sampleCollectStepSizeList = [25, 10, 5, 3]
# sampleCollectStepSizeList = [25, 25, 25,  10, 7,4]
sampleCollectStepSizeList = [43, 21, 9, 7, 6, 5, 4, 3, 2, 1]
for stpSize in sampleCollectStepSizeList:
    k = 1
    allCollectedSampleSetX = []
    allCollectedSampleLabels = []
    listnTp = []
    listnFp = []
    listnGT = []



    for f, g in zip(trainjpg, traincsv):
        # load the ground truth marks
        gtforfile = loadCSV(g.rstrip('\n'))
        gtforfile = list(gtforfile)
        print f, g
        print "Wındow step:", stpSize
        imInput = loadImage(f.rstrip('\n'))
        # normalize image to [0-1.0]
        imInput = sampleFactory.normRGB(imInput)

        # classify each window with stepSize
        imBinary, imProb = slidingWindow(clf,
                                         imInput, WINDOWSIZE, stepSize=stpSize)

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
        #posList = gtforfile # + truepos
        posList = falseneg # + truepos
        if(len(posList)==0):
            posList = gtforfile
        # collect windows from the image in different scales (positive multiscale only)    
        newPos, newNeg = collectSamples(imInput, posList, falsepos,
                                        multiScale=[1.0, 1.1, 1.2])
        
        # create trainining  set and labels, augment multiply positive samples if required
        trainsetx, trainsety = createTrainSet(newPos, newNeg, AUGMENTPOSITIVESAMPLES)

        # sample length
        # newSamples = newSamples + len(trainsety)
        nPositiveSamples = nPositiveSamples +  np.sum(trainsety==1)
        nNegativeSamples = nNegativeSamples +  np.sum(trainsety==0)

        if k == 1:
            allCollectedSampleSetX = np.copy(trainsetx)
            allCollectedSampleLabels = np.copy(trainsety)
        else:
            allCollectedSampleSetX = np.concatenate((allCollectedSampleSetX, trainsetx), axis=0)
            allCollectedSampleLabels = np.concatenate((allCollectedSampleLabels, trainsety), axis=0)

    #   record new samples to the file
        
        if TRAININGMODE == 0:
            # we do not train
            # training will be done after the loop
            print "Samples are stored"

        elif TRAININGMODE == 1:
            print "Training X: ", np.shape(trainsetx)
            print "Training y: ", np.shape(trainsety)
            clf, oob_error = trainIncrementally(clf, trainsetx, trainsety)

        elif TRAININGMODE == 2 and nNegativeSamples > MINIMUMNEGATIVESAMPLESTOTRAIN:

            print "Training X: ", np.shape(allCollectedSampleSetX)
            print "Training y: ", np.shape(allCollectedSampleLabels)
            clf, oob_error = trainIncrementally(clf,
                                                allCollectedSampleSetX,
                                                allCollectedSampleLabels)
    #       reset sample counter
            newSamples = 0
        # now use new samples to train our classifier

        # leave time to recognize keyboard interrupt
        sleep(2)

        if k == NUMBEROFTRAININGIMAGES:
            # calculate the performance
            TPrate = float(np.sum(np.array(listnTp))) / float(np.sum(np.array(listnGT)))
            FPpIMG = np.sum(np.array(listnFp)) / float(k)
            print "TP:", TPrate
            print "FP:", FPpIMG
            break
        k = k+1

    if TRAININGMODE == 0:
        # training will be performed after loop ends
        print "Training X: ", np.shape(allCollectedSampleSetX)
        print "num Pos ", nPositiveSamples
        print "num Neg ", nNegativeSamples

        if nNegativeSamples > MINIMUMNEGATIVESAMPLESTOTRAIN:
            clf, oob_error = trainIncrementally(clf,
                                                allCollectedSampleSetX,
                                                allCollectedSampleLabels)
            nNegativeSamples = 0
            nPositiveSamples = 0
            print "Trained Error {0}".format(oob_error)
            # save the training
            _ = joblib.dump(clf, CLASSIFIERFILE_CONT, compress=9)
            print "trained", clf

        else:
            print "skipping train not enough samples"

    sleep(2)
