import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T 

from utils import load_data, tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class cA(object):
    def __init__(self, np_rng, input=None, n_visible=784, n_hidden=100,
        n_batchsize=1, W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_batchsize = n_batchsize

        if W is None:
            W_value = np.asarray(
                    np_rng.uniform(
                            low=-4*np.sqrt(6./(n_hidden + n_visible)),
                            high=4*np.sqrt(6./(n_hidden + n_visible)),
                            size=(n_visible, n_hidden)
                        ),
                    dtype=theano.config.floatX
                )
            W = theano.shared(W_value, name='W', borrow=True)

        if not bhid:
            bhid = theano.shared(
                    value=np.zeros(n_hidden, dtype=theano.config.floatX),
                    name='bhid',
                    borrow=True
                )

        if not bvis:
            bvis = theano.shared(
                    value=np.zeros(n_visible, dtype=theano.config.floatX),
                    name='bvis',
                    borrow=True
                )
        self.W = W
        self.b = bhid
        self.b_prime = bvis
        self.W_prime = self.W.T

        if input is None:
            self.x = T.matrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]

    def get_hidden_values(self, input):
        pre_act = T.dot(self.x, self.W) + self.b
        return T.nnet.sigmoid(pre_act)

    def get_jacobian(self, hidden, W):
        """
        Compute the jacobian of the hidden layer with respect 
        to the input, reshape are necessary for broadcasting the element-wise
        product on the right axis
        """
        return T.reshape(hidden * (1 - hidden), (self.n_batchsize, 1, self.n_hidden)) * T.reshape(W, (1, self.n_visible, self.n_hidden))

    def get_reconstructed_input(self, hidden):
        pass