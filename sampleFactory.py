# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 15:47:53 2016

@author: btek
"""

# from oct2py import octave
from numpy import random, fliplr, flipud, rot90, shape, mean
from numpy import clip, stack, float32, sqrt, zeros, arange, ceil, uint8
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
MEANS_FILE = sio.loadmat('/home/btek/Dropbox/code/matlabcode/mitosis_code/amida_elm/amida_mitos_means_v2.mat')
MEANS_ARR = MEANS_FILE['rgb_means']

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


# normalize a matrix of rgb values to zero mean variance
def normRGB(X, mx=255.0, mn=0):

    c = float32(1/mx*2.0)
    # print c
    X *= c
    X -= 1.0
    return X


def denormRGB(X, mx=1, mn=-1, doclip=True):
    Y = X - mn
    Y = Y * 127.5
    if doclip:
        Y = clip(Y, 0, 255)
    return Y

def plotTrainSetInSubPlots(setX, wy, wx, nchannels, fignum=332):
    # find number of whole edges  
    mxplots = len(setX)
    if (mxplots > MXSUBPLOTS):
        mxplots = MXSUBPLOTS
    nedge = sqrt(mxplots)
    nedge_ceil = int(ceil(nedge))

    fig = plt.figure(fignum, (8., 8.))
    
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                nrows_ncols = (nedge_ceil, nedge_ceil), # creates nedge x nedge grid of axes
                axes_pad=0.01 # pad between axes in inch.
                )

    for i in range(mxplots):
        grid[i].imshow(uint8(denormRGB(setX[i].reshape(wy, wx, nchannels)))) # The AxesGrid object work as a list of axes.

    plt.show()
    

# generate new Samples with random transformations, multiplier times the orig
def generateSamples(imgSet, multiplier):
    outSet = []
    k = 0
    for img in imgSet:
        for m in range(0, multiplier):
            cand = doRandomRotation(img)
            cand = doRandomCrop(cand)
            cand = doColorMeanShift(cand)
            if shape(cand) == shape(img):
                outSet.append(cand)
            else:
                print "sample shape changes: {0}".format(shape(cand))
                k = k+1
            
    return outSet


# produce a random rotation
def doRandomRotation(img):
    out = img
    r = random.randint(4)

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
    
    return out


# produce a random margin, resize image back to oriMEANS_ARRginal
def doRandomCrop(img):
    out = img
    originalSize = shape(out)
    x = random.randint(CROP_WIDTH_MAX)
    y = random.randint(CROP_HEIGHT_MAX)
    out = img[y:-CROP_HEIGHT_MAX, x:-CROP_WIDTH_MAX]
    #print "cropped shape:", out.shape
    out = resize(out, originalSize)
    if shape(out) != shape(img):
        print "crop changes size in coords{0},{1}".format(x,y)
       
    return out


# change the mean with a random mitosis mean
def doColorMeanShift(img, randommix=True):
    # choose a random RGB Mean
    # out = np.zeros(np.shape(img))
    nmeans = MEANS_ARR.shape[0]
    img_RGB = denormRGB(img)
    im_r = img_RGB[:, :, 0]
    im_g = img_RGB[:, :, 1]
    im_b = img_RGB[:, :, 2]
    mn_r = mean(im_r)
    mn_g = mean(im_g)
    mn_b = mean(im_b)

    rc = MEANS_ARR[random.randint(nmeans), :]
    # print rc
        
    im_ra = im_r / mn_r * rc[0] 
    im_ga = im_g / mn_g * rc[1]
    im_ba = im_b / mn_b * rc[2]

    br = 0.0
    bg = 0.0
    gr = 0.0
    if randommix:
        br = 0.05 * (random.rand(1) - 0.5)
        bg = 0.05 * (random.rand(1) - 0.5)
        gr = 0.05 * (random.rand(1) - 0.5)
#        print '.05'
    
    im_r2 = im_ra + im_ga * gr + im_ba * br
    im_g2 = im_ga + im_ra * gr + im_ba * bg
    im_b2 = im_ba + im_ra * br + im_ga * br
    
    out = stack((im_r2, im_g2, im_b2), axis=2)
    normRGB(out)

    return out
