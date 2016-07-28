# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:12:52 2016

@author: btek
"""

#INPUTFILE = u'amida_samples_org.npy'
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn.datasets
import sys
sys.path.insert(0, '/home/btek/src/Lasagne/build/lib/lasagne/')
import lasagne
import theano
import theano.tensor as T
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params

import sys

INPUTFILE = u'./mitosisData/AugmentedSampleAll_aug_X.npy'
INPUTTARGETFILE = u'./mitosisData/AugmentedSampleAll_orig_X.npy'
TARGETCLASSFILE = u'./mitosisData/AugmentedSampleAll_all_y.npy'
MODELFILE = u'elm_model_all'
WIDTH = 50
HEIGHT = 50
debug = False

conv = lasagne.layers.Conv2DLayer
pool = lasagne.layers.MaxPool2DLayer
NUM_EPOCHS = 500
BATCH_SIZE = 100 # set this LOW FOR SPTN
LEARNING_RATE = 0.01
GAUSSIAN_SIGMA = 0.3
DIM = 50
CHANNELS = 3


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


# normalize a matrix of rgb values to zero mean variance
def normRGB(X, mx = 255.0, mn =0):
    
    c = np.float32(1/mx*2.0)
    #print c
    X *= c
    X -= 1.0
    return X
    
def build_dense_net(input_width, input_height, batch_size=BATCH_SIZE):

    l_in = lasagne.layers.InputLayer(shape=(None, CHANNELS, input_width, input_height))

    l_sca1 = lasagne.layers.ScaleLayer(l_in)

    l_enc1 = lasagne.layers.DenseLayer(l_sca1,
                                     num_units=(input_width), nonlinearity=lasagne.nonlinearities.elu,
                                       name='encoder1')

    print "Encoder 1 : ", l_enc1.output_shape
    
    l_enc2 = lasagne.layers.DenseLayer(l_enc1, num_units=(input_width*input_height*CHANNELS),W=l_enc1.W.T,
                                       nonlinearity=lasagne.nonlinearities.linear, name='encoder2')

    print "Encoder 2 : ", l_enc2.output_shape
    #l_enc3 = lasagne.layers.DenseLayer(l_enc2, num_units=input_width * input_height,W=lasagne.init.Constant(1.0),
    #                                    nonlinearity=lasagne.nonlinearities.identity, name='encoder3')

    #l_enc3.add_param(l_enc3.W, l_enc3.W.get_value().shape, trainable=False)
    #l_enc3.add_param(l_enc3.b, l_enc3.b.get_value().shape, trainable=False)

    #l_sca2 = lasagne.layers.ScaleLayer(l_enc2)

    l_out = lasagne.layers.ReshapeLayer(l_sca1, shape=(-1, CHANNELS, input_width, input_height))
    
    print "Encoder 1 l_out: ", l_out.output_shape
    return l_out, l_enc1, l_enc2
    
    
def build_encoder1(input_width, input_height, batch_size=BATCH_SIZE):
    ini = lasagne.init.HeUniform()
    
    l_in = lasagne.layers.InputLayer(shape=(None, CHANNELS, input_width, input_height))

    l_norm = lasagne.layers.BatchNormLayer(l_in)
    # localization part
    b = np.zeros((2, 3), dtype=theano.config.floatX)
    b[0, 0] = 1
    b[1, 1] = 1
    b = b.flatten()


    l_enc1 = lasagne.layers.DenseLayer(l_norm,
                                       num_units =  int(input_width*input_height/16), nonlinearity = lasagne.nonlinearities.elu, W = lasagne.init.Constant(0.0), name='encoder1')
    print "Encoder 1 output shape: ", l_enc1.output_shape

    l_theta = lasagne.layers.DenseLayer(l_enc1, num_units=6, b=b, W=lasagne.init.Constant(0.0),
                                        nonlinearity=lasagne.nonlinearities.linear, name="theta")

    print "Theta input, output shape:  ", l_theta.input_shape, l_theta.output_shape

    # transformer
    l_trans1 = lasagne.layers.TransformerLayer(l_in, l_theta, downsample_factor=1)
    print "Transformer layer output shape: ", l_trans1.output_shape

    #l_sca2 = lasagne.layers.ScaleLayer(l_trans1)

    l_out = lasagne.layers.ReshapeLayer(l_trans1, shape=(-1, CHANNELS, input_width, input_height))

    print "Transformer network output shape: ", l_out.output_shape

    return l_out, l_trans1, l_theta


def build_conv_net(input_width, input_height, batch_size=BATCH_SIZE):
    ini = lasagne.init.HeUniform()

    l_in = lasagne.layers.InputLayer(shape=(None, CHANNELS, input_width, input_height))
    l_norm = lasagne.layers.BatchNormLayer(l_in)
    l_sca1 = lasagne.layers.ScaleLayer(l_norm)

    loc_l2 = conv(l_sca1, num_filters=10, filter_size=(5, 5), W=ini, nonlinearity=lasagne.nonlinearities.elu)
    loc_l3 = pool(loc_l2, pool_size=(2, 2))
    loc_l4 = conv(loc_l3, num_filters=10, filter_size=(3, 3), W=ini, nonlinearity=lasagne.nonlinearities.elu)
    loc_l5 = pool(loc_l4, pool_size=(2, 2))
    loc_l6 = conv(loc_l5, num_filters=10, filter_size=(3, 3), W=ini,nonlinearity=lasagne.nonlinearities.elu)
    loc_l7 = pool(loc_l6, pool_size=(2, 2))
    
    l_norm2 = lasagne.layers.BatchNormLayer(loc_l7)
    
    l_enc1 = lasagne.layers.DenseLayer(l_norm2, num_units=(CHANNELS*input_width * input_height), W=lasagne.init.GlorotUniform(),b=lasagne.init.Constant(0.0),
                                       nonlinearity=lasagne.nonlinearities.linear, name='dense')

  #  l_enc1.add_param(l_enc1.W, l_enc1.W.get_value().shape, trainable=False)
   # l_enc1.add_param(l_enc1.b, l_enc1.b.get_value().shape, trainable=False)
    #l_sca2 = lasagne.layers.ScaleLayer(l_enc1)

    l_out = lasagne.layers.ReshapeLayer(l_enc1, shape=(-1,CHANNELS, input_width, input_height))

    return l_out, l_enc1, loc_l7
# this function must actually take many transformed RGB images and output a single image
def train_epoch(Xin, Xout, train_func):
    num_samples = Xin.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    for i in range(num_batches):
        idx = range(i * BATCH_SIZE, np.minimum((i + 1) * BATCH_SIZE, num_samples))
        X_batch = Xin[idx]
        Y_batch = Xout[idx]
        #print X_batch.shape
        #print Y_batch.shape
        
        cost_batch, _ = train_func(X_batch, Y_batch)
        costs += [cost_batch]

    return np.mean(costs)


def evaluate(X, y, eval_func):
    output_eval, transform_eval = eval_func(X)
    preds = np.argmax(output_eval, axis=-1)
    acc = np.mean(preds == y)
    return acc, transform_eval
    

def showResults(test_in, test_out, eval_func):
    # theano.printing.pydotprint(train_func, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)
    ix = 1
    
    # [eval_X, eval_tr, eval_teta, eval_input] =eval_func(Xinp)
    [eval_X, enc_X, _] = eval_func(test_in)

    print "Transform params", enc_X.shape


    fig = plt.figure(figsize=(5, 3))
    ax11 = fig.add_subplot(341)
    ax12 = fig.add_subplot(342)
    ax13 = fig.add_subplot(343)
    ax14 = fig.add_subplot(344)

    ax11.imshow(test_in[ix, 0, :, :])
    ax11.set_title("Network input")

    ax12.imshow(enc_X[ix].reshape(np.sqrt(enc_X[ix].size), np.sqrt(enc_X[ix].size)).squeeze())
    print "Mean ", enc_X[ix].mean(), "Max ", enc_X[ix].max(), "Min ", enc_X[ix].min()
    ax12.set_title("Encoder input")
    ax12.axes.set_aspect('equal')

    ax13.imshow(eval_X[ix].squeeze())
    ax13.set_title("Network output")
    ax13.axes.set_aspect('equal')

    ax14.imshow(test_out[ix].squeeze())
    ax14.set_title("Target")

    test_neg = data['X_test_neg']
    [eval_neg, enc_neg, _] = eval_func(test_neg)

    ax21 = fig.add_subplot(345)
    ax22 = fig.add_subplot(346)
    ax23 = fig.add_subplot(347)
    ax24 = fig.add_subplot(348)
    bix =26
    ax21.imshow(test_in[bix, 0, :, :])
    ax21.set_title("Network input")

    ax22.imshow(enc_X[bix].reshape(np.sqrt(enc_X[bix].size), np.sqrt(enc_X[bix].size)).squeeze())
    print "Mean ", enc_X[bix].mean(), "Max ", enc_X[bix].max(), "Min ", enc_X[bix].min()
    ax22.set_title("Encoder input")
    ax22.axes.set_aspect('equal')

    ax23.imshow(eval_X[bix].squeeze())
    ax23.set_title("Network output")
    ax23.axes.set_aspect('equal')

    ax24.imshow(test_out[bix].squeeze())
    ax24.set_title("Target")

    test_neg = data['X_test_neg']
    [eval_neg, enc_neg, _] = eval_func(test_neg)


    ax31 = fig.add_subplot(3,4,9)
    ax32 = fig.add_subplot(3,4,10)
    ax33 = fig.add_subplot(3,4,11)
    ax34 = fig.add_subplot(3,4,12)
    ax31.imshow(test_neg[0].squeeze())
    ax31.set_title("Negative input")
    ax32.imshow(eval_neg[0].squeeze())
    ax32.set_title("Negative output")
    ax33.imshow(test_neg[2].squeeze())
    ax33.set_title("Negative input")
    ax34.imshow(eval_neg[2].squeeze())
    ax34.set_title("Negative output")
    plt.show(block=False)

#####################################################################################################################
#####################################################################################################################
    
gfilter = makeGaussian(50,)
#read the input file,,,
XXin = (np.load(INPUTFILE)).astype('float32')
XXout = (np.load(INPUTTARGETFILE)).astype('float32')
c = np.float32(1.0/255.0*2.0)
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

trainix = range(0,int(XXin.shape[0]/2),1)
testix = range(int(XXin.shape[0]/2), XXin.shape[0], 1)
XXtrainIn = XXin[trainix,]
XXtrainOut = XXout[trainix,]
YYtrain = YY[trainix,]
ytrain = y[trainix]

XXtest = XXin[testix,]
YYtest = YY[testix,]
ytest = y[testix]


Xinp = XXtrainIn.reshape([-1, 3,50,50], order='F')
Xout = XXtrainOut.reshape([-1,3,50,50], order='F')
print "Input shape ", Xinp.shape

#N = 200
#print "using only",N
#Xinp = Xinp[1:N]
#Xout = Xout[1:N]
#print "Input shape ", Xinp.shape

# model,l_enc = build_test_net(data['input_width'], data['input_height'])
#model, l_trans, l_theta = build_dense_net(WIDTH, HEIGHT, BATCH_SIZE)
model, l_trans, l_theta = build_conv_net(WIDTH, HEIGHT, BATCH_SIZE)
#model, l_trans, l_theta = build_encoder1(WIDTH, HEIGHT, BATCH_SIZE)
model_params = lasagne.layers.get_all_params(model, trainable=True)
model_regularizable = lasagne.layers.get_all_params(model, regularizable=True)
# model_params_sub = model_params[:-2]
## regularization
print "Params", model_params, "\n Regularizable", model_regularizable
all_weights_penalty = lasagne.regularization.regularize_network_params(model, l2)
#transfomer_penalty = lasagne.regularization.regularize_layer_params(l_theta, l2)

# creating trainer function

X = T.tensor4()
Y = T.tensor4()

# training output
output_train = lasagne.layers.get_output(model, X, deterministic=False)
output_eval, trans_eval, theta_eval = lasagne.layers.get_output([model, l_trans, l_theta], X, deterministic=True)
# output_eval, transform_eval, teta_eval, input_eval = lasagne.layers.get_output([model,l_transform, l_teta, l_inp], X, deterministic=True)

gaussian_cost_weights = makeGaussian(WIDTH,WIDTH*GAUSSIAN_SIGMA)
#cost = T.mean(lasagne.objectives.squared_error(output_train,
#                                               Y) * gaussian_cost_weights) + transfomer_penalty + 0.01 * all_weights_penalty
cost = T.mean(lasagne.objectives.squared_error(output_train,Y) * gaussian_cost_weights)  #+ all_weights_penalty
learn_rate = theano.shared(LEARNING_RATE, name='learn_rate')
updates = lasagne.updates.sgd(cost, model_params, learning_rate=learn_rate)
# updates = lasagne.updates.adam(cost, model_params_sub, learning_rate=LEARNING_RATE)

train_func = theano.function([X, Y], [cost, output_train], updates=updates)
eval_func = theano.function([X], [output_eval, trans_eval, theta_eval])

train_accs = []
an_example = Xinp[2]
print "start training"
try:
    for n in range(NUM_EPOCHS):
        #print n
        train_cost = train_epoch(Xinp, Xout, train_func)
        [out_k, trans_k, theta_k] = eval_func(np.expand_dims(an_example, 0))
        #print "teta: ", theta_k

        train_cost += [train_cost]
        if n % 20 == 0:
            new_lr = learn_rate.get_value() * 0.99
            print "New LR:", new_lr
            learn_rate.set_value(lasagne.utils.floatX(new_lr))
            #showResults(data,eval_func)
        print "Epoch {0}: Train cost {1}".format(n, train_cost)
except KeyboardInterrupt:
    pass

