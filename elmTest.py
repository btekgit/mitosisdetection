import os

import matplotlib
import numpy as np

import scipy.io


np.random.seed(123)
import matplotlib.pyplot as plt

from hpelm import ELM

ORGFILE = u'/home/btek/Downloads/mitosisData/01_01_org_60.npy'
TRNSFILE = u'/home/btek/Downloads/mitosisData/01_01_rot_60.npy'

import numpy as np


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


def load_single_data(orgfile, trnsfile):
    # skipping multichannel input for now.
    orgfile = u'/home/btek/Dropbox/linuxshare/mitosisData/01_01_org_50.npy'
    trnsfile = u'/home/btek/Dropbox/linuxshare/mitosisData/01_01_aff_50.npy'
    negativefile = u'/home/btek/Dropbox/linuxshare/mitosisData/01_01_nonmitos_50.npy'

    trn = np.load(trnsfile)
    trn = np.expand_dims(trn[:, :, 1, :], 2)
    trn_t = np.transpose(trn, [3, 2, 0, 1])

    org = np.load(orgfile)
    org_n = org[:, :, 1]
    org_n = np.expand_dims(org_n, 0)
    org_n = np.repeat(org_n, trn_t.shape[0], 0)
    org_n = np.expand_dims(org_n, 1)

    tst_neg = np.load(negativefile)
    tst_neg_n  = np.expand_dims(tst_neg[:, :, 2, :], 2).transpose([3, 2, 0, 1])

    print "Train input samples:", trn_t.shape
    print "Train output samples:", org_n.shape

    d = dict(X_input=(trn_t[np.arange(0,61,2),:,:,:]).astype('float'),
             X_output=(trn_t[np.arange(60,-1,-2),:,:,:]).astype('float'),
			 X_test_in = (trn_t[range(1,-1,2),:,:,:]).astype('float'),
			 X_test_out = (org_n[range(1,-1,2),:,:,:]).astype('float'),
             X_test_neg = (tst_neg_n[:,:,:,:]).astype('float'),
             num_examples_input=trn_t.shape[0],
             num_examples_output=org_n.shape[0],
             input_height=trn_t.shape[2],
             input_width=trn_t.shape[3],
             output_height=org_n.shape[2],
             output_width=org_n.shape[3])

    return d

def build_ELM_encoder(xinput, target, num_neurons):


    elm = ELM(xinput.shape[1], target.shape[1])
    elm.add_neurons(num_neurons, "sigm")
    elm.add_neurons(num_neurons, "lin")
    #elm.add_neurons(num_neurons, "rbf_l1")
    elm.train(xinput, target, "r")
    ypred = elm.predict(xinput)
    print "mse error", elm.error(ypred, target)
    return elm, ypred


## this part is the main.
data = load_single_data(ORGFILE, TRNSFILE)
Xinp = data['X_input']
Xout = data['X_output']
xflatten = np.reshape(Xinp,[-1,2500])#.transpose(1,0)
yflatten = np.reshape(Xout,[-1,2500])#.transpose(1,0)
print "Input shape ", xflatten.shape,
print "output shape: ", yflatten.shape

elm_model, ypred = build_ELM_encoder(xflatten, yflatten,50)

m = {"inputs": elm_model.nnet.inputs,
             "outputs": elm_model.nnet.outputs,
             "Classification": elm_model.classification,
             "Weights_WC": elm_model.wc,
             "neurons": elm_model.nnet.neurons,
             "norm": elm_model.nnet.norm,  # W and bias are here
             "Beta": elm_model.nnet.get_B()}
scipy.io.savemat('myelm.mat', m)


fig = plt.figure(figsize=(5, 3))
ax31 = fig.add_subplot(331)
ax32 = fig.add_subplot(332)
ax33 = fig.add_subplot(333)
ax34 = fig.add_subplot(334)
ax35 = fig.add_subplot(335)
ax36 = fig.add_subplot(336)

ax37 = fig.add_subplot(337)
ax38 = fig.add_subplot(338)
ax39 = fig.add_subplot(339)


ix = 5
ax31.set_title(" input")
xin = xflatten[ix,:]
ax31.imshow(xin.reshape(np.sqrt(xin.size), np.sqrt(xin.size)).squeeze())
xt = yflatten[ix,:]
ax32.imshow(xt.reshape(np.sqrt(xt.size), np.sqrt(xt.size)).squeeze())
ax32.set_title(" target")
xout = ypred[ix,:]
ax33.imshow(xout.reshape(np.sqrt(xout.size), np.sqrt(xout.size)).squeeze())
ax33.set_title("output")

ix = 20
xin = xflatten[ix,:]
ax34.imshow(xin.reshape(np.sqrt(xin.size), np.sqrt(xin.size)).squeeze())
ax34.set_title(" input")
xt = yflatten[ix,:]
ax35.imshow(xt.reshape(np.sqrt(xt.size), np.sqrt(xt.size)).squeeze())
ax35.set_title(" target")
xout = ypred[ix,:]
ax36.imshow(xout.reshape(np.sqrt(xout.size), np.sqrt(xout.size)).squeeze())
ax36.set_title("output")




ix = 8
tst_neg_sample = data['X_test_neg']
tst_neg_sample_i = tst_neg_sample[ix]
xin = elm_model.predict(tst_neg_sample_i.reshape([1,2500]))
ax37.imshow(tst_neg_sample_i.reshape(np.sqrt(xin.size), np.sqrt(xin.size)).squeeze())
ax37.set_title(" input")

ax38.imshow(xin.reshape(np.sqrt(xin.size), np.sqrt(xin.size)).squeeze())
ax38.set_title(" target")

plt.show(block=True)
