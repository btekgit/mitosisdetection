# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:47:53 2016

@author: btek
"""

# from oct2py import octavefrom numpy import random, fliplr, flipud, rot90, shape, mean
from numpy import clip, stack, float32, sqrt, zeros, arange, ceil, uint8
import numpy as np
from skimage.transform import resize
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
CROP_WIDTH_MAX = 5
CROP_HEIGHT_MAX = 5
MXSUBPLOTS = 36
DEBUG_plot = True




# load some pre recorded rgb values.
# these are used to produce random meaningful color transformations
# NOT USING THIS BECAUSE IT CAUSES PROBLEMS, SATURATIONS
#MEANS_ARR = np.load('mitosisData/mitosis_rgb_means.npy')


## load image processing package
#octave.pkg('load', 'image')
#
## add path
#octave.addpath('/home/btek/Dropbox/code/matlabcode/utility')
#
#
## call octave rotation method
#def generateRotatedSamples(img, numRotations):
#    return octave.generateRotatedData2(img, numRotations)
#
#
#def generateCroppedSamples(img, numCrops):
#    return octave.generateCroppedData(img, numCrops)
#
#
#def generateIntensitySamples(img, numInt):
#    return octave.generateIntensityNoise
#MEANS_ARR



def normRGB(X, mx=255.0, mn=0, doclip=True):
    """
    normalize a matrix of rgb values to zero mean variance
    """
    c = float32(1/mx*2.0)
    # print c
    Y = X*c
    Y -= 1.0
    if doclip:
        Y = clip(Y, -1.0, 1.0)    
    #print "normed data new max", np.max(X), " min:", np.min(X)
    return Y


def denormRGB(X, mx=1, mn=-1, doclip=True):
    Y = X - mn
    Y = Y * 127.5
    if doclip:
        Y = clip(Y, 0, 255)
    #print "DENOrmed data new max", np.max(Y), " min:", np.min(Y)
    return Y

def plotTrainSetInSubPlots(setX, wy, wx, nchannels, fignum=332):
    # find number of whole edges  
    mxplots = len(setX)
    if (mxplots > MXSUBPLOTS):
        mxplots = MXSUBPLOTS
    nedge = sqrt(mxplots)
    nedge_ceil = int(ceil(nedge))

    fig = plt.figure(fignum, (12., 12.))

    grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (nedge_ceil, nedge_ceil), # creates nedge x nedge grid of axes
                axes_pad=0.01 # pad between axes in inch.
                    )
    for i in range(mxplots):
        if isinstance (setX, list):
            grid[i].imshow(uint8(
                                 denormRGB(setX[i].reshape(wy, wx, nchannels)))) # The AxesGrid object work as a list of axes.
        else:
            #print i
            grid[i].imshow(uint8(
                                 denormRGB(setX[i, :].reshape(wy, wx, nchannels)))) # The AxesGrid object work as a list of axes.

    plt.show()


# generate new Samples with random transformations, multiplier times the orig
def generateSamples(inSet, multiplier):

    k = 0
    outSet = []
    indexSet = []

    if isinstance(inSet, list):
        print "generate samples convertin imgSet into a list of RGB imgs"
        n_img = len(inSet)
        listSet = inSet
    elif isinstance(inSet, np.ndarray):
        n_img = inSet.shape[0]
        listSet = [inSet[i, :, :, :] for i in range(0, n_img)]

    for i in range(0, n_img):
        img = listSet[i]
        for m in range(0, multiplier):
            #print "shape before", np.shape(img()
            cand = np.copy(img)
            cand = doRandomRotation(cand)
            #cand = doRandomCrop(cand)
            cand = doRandomColorShift(cand)
            #print " random and crop"
            if shape(cand) == shape(img):
                outSet.append(cand)
            else:
                print "sample shape changes: {0}".format(shape(cand))
            k = k+1

            indexSet.append(i)
    #return also indexes to be used for label matching. do
    #   ynew = y(indexSet)
    if isinstance(inSet, np.ndarray):
        return np.asarray(outSet, dtype='float32'), np.asarray(indexSet, dtype='int32')
    else:
        return outSet, indexSet



# produce a random rotation
def doRandomRotation(img):
    out = np.copy(img)
    r = random.randint(5)

    if (r == 0):
        out = fliplr(img)
    elif(r == 1):
        out = flipud(img)
    elif(r == 2):
        out = rot90(img)
    elif(r == 3):
        out = rot90(img, 3)

    if shape(out) != shape(img):
        print "rotate changes size in", out.shape, img.shape
    #print "shape before", shape(img), "shape after rot", shape(out)
    return out


# produce a random margin, resize image back to oriMEANS_ARRginal
def doRandomCrop(img):
    out = np.copy(img)
    originalSize = shape(out)

    x = random.randint(CROP_WIDTH_MAX)

    y = random.randint(CROP_HEIGHT_MAX)
    out = img[y:-CROP_HEIGHT_MAX, x:-CROP_WIDTH_MAX]
    #print "size:", originalSize, " x,y: ",x,y
    #print "cropped shape:", out.shape
    out = resize(out, originalSize)
    if shape(out) != shape(img):
        print "crop changes size in coords{0}, {1}".format(x,y)

    return out


# change the mean with a random mitosis mean
def doColorMeanShift(img):
    # choose a random RGB Mean
    out = np.copy(img)
    #nmeans = MEANS_ARR.shape[0]
    img_RGB = denormRGB(img)

    im_r = img_RGB[:, :, 0]
    im_g = img_RGB[:, :, 1]
    im_b = img_RGB[:, :, 2]
    mn_r = mean(im_r)
    mn_g = mean(im_g)
    mn_b = mean(im_b)

    #print "means:",mean(im_r), mean(im_g), mean(im_b)
    br = 0.0
    bg = 0.0
    gr = 0.0

    br = 0.05 * (random.rand(1) - 0.5)
    bg = 0.05 * (random.rand(1) - 0.5)
    gr = 0.05 * (random.rand(1) - 0.5)
    #print "color mix coeff",br, bg, gr
    im_r2 = im_r + im_g * gr + im_b * br
    im_g2 = im_g + im_r * gr + im_b * bg
    im_b2 = im_b + im_r * br + im_g * bg

    mx_r = np.max(im_r2)
    mx_g = np.max(im_g2)
    mx_b = np.max(im_b2)

    mn_r = np.min(im_r2)
    mn_g = np.min(im_g2)
    mn_b = np.min(im_b2)
    #print "means:",mx_r,mx_g,mx_b,mn_r,mn_g,mn_b

    im_r2 = np.clip(im_r2, 0, 255)
    im_g2 = np.clip(im_g2, 0, 255)
    im_b2 = np.clip(im_b2, 0, 255)
    # stretch may not be working very good
    #im_r2 = (im_r2-mn_r)/(mx_r-mn_r)*255.0
    #im_g2 = (im_g2-mn_g)/(mx_g-mn_g)*255.0
    #im_b2 = (im_b2-mn_b)/(mx_b-mn_b)*255.0


    #print "means:",mean(im_r2), mean(im_g2), mean(im_b2), np.max(im_r2), np.min(im_r2)
    out[:, :, 0] = im_r2
    out[:, :, 1] = im_g2
    out[:, :, 2] = im_b2

    #print "max:", np.max(out), "mean:", np.mean(out), "min:", np.min(out)
    checkmean = np.mean(out)
    if(checkmean > 250):
        print "Hig mean max:", np.max(out), "mean:", np.mean(out), "min:", np.min(out)
        #print("check here mean is high")
    
    out = normRGB(out, doclip=True)

    return out

def doRandomColorShift(img):

    out = np.copy(img)
    out = denormRGB(out)
    # value between 5...-4
    a = random.randint(10)-4
    
    # choose a random channel
    c = random.randint(3)
    
    # change that channel value
    out[:, :, c] = out[:, :, c] + a
    
    #print "means:", np.mean(out), np.max(out), np.min(out)
    out = np.clip(out, 0, 255)
    out = normRGB(out, doclip=True)

    
    return out


def doMinMaxNorm(img):
    
    out = np.copy(img)
    #nmeans = MEANS_ARR.shape[0]
    #img_RGB = denormRGB(img)
    
    im_r = img[:, :, 0]
    im_g = img[:, :, 1]
    im_b = img[:, :, 2]
    
    mx_r = np.max(im_r)
    mx_g = np.max(im_g)
    mx_b = np.max(im_b)

    mn_r = np.min(im_r)
    mn_g = np.min(im_g)
    mn_b = np.min(im_b)

    im_r2 = (im_r-mn_r)/(mx_r-mn_r)*2.
    im_g2 = (im_g-mn_g)/(mx_g-mn_g)*2.
    im_b2 = (im_b-mn_b)/(mx_b-mn_b)*2.

    out[:, :, 0] = im_r2-1.
    out[:, :, 1] = im_g2-1.
    out[:, :, 2] = im_b2-1.

    return out

def testNormDenorm():
    import scipy
    img = scipy.misc. imread('/home/btek/Dropbox/code/pythoncode/linuxsource/src/mitosisdetection/mitosisData/amida_train/01/09.jpg')
    img = np.float32(img)
    out = doColorMeanShift(normRGB(img))
    plt.imshow(np.uint8(denormRGB(out)))


def reshapeData(X, originalShape, targetShape, interpMethod='nearest'):
    """Reshape X matrix rows,
        originalShape is a tuple,e.g. 40,40,3
        targetShape is the target row Shape, e.g. (20,20,3)
        interpMethod:denotes the interpolation method
        """
    if (np.shape(X)[1]) != (np.prod(originalShape)):
        print "Input dimensions do not match with the originalShape", np.shape, "!=", np.prod(originalShape)

    if np.ndim(originalShape) == np.ndim(targetShape):
        print np.ndim(originalShape),"==", np.ndim(targetShape)

    nimg = X.shape[0]
    x_out = np.zeros((nimg, np.prod(targetShape)), dtype=X.dtype)

    for i in range(0, nimg):
        row_img = X[i, :].reshape(originalShape)
        row_img_out = resize(row_img, targetShape, mode=interpMethod)
        row_vec_out = row_img_out.reshape(1,-1)
        x_out[i, :] = row_vec_out

    return x_out


