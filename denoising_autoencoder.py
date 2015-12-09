"""
Construct the denoising autoencoder
"""

import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from utils import load_data, tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image


class dA(object):
    """ Denoising Auto-Encoder class """
    def __init__(self, np_rng, theano_rng=None, input=None, n_visible=784, n_hidden=500, W=None, bhid=None, bvis=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2 ** 30))

        # create a theano random generator 
        if not W:
            W_value = np.asarray(
                np_rng.uniform(
                    low=-4*np.sqrt(6./(n_visible + n_hidden)), 
                    high=4*np.sqrt(6./(n_visible + n_hidden)),
                    size=(n_visible, n_hidden)),
                dtype=theano.config.floatX
                )
            W = theano.shared(value=W_value, name="W", borrow=True)

        if not bhid:
            bhid = theano.shared(
                value=np.zeros(n_hidden, dtype=theano.config.floatX),
                name="bhid", borrow=True
                )

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(n_visible, dtype=theano.config.floatX),
                name="bvis", borrow=True
                )

        self.W = W
        self.W_prime = self.W.T # using tiled weight
        self.b = bhid
        self.b_prime = bvis
        self.theano_rng = theano_rng

        if input is None:
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]


    def get_corrupted_input(self, input, corruption_rate):
        return self.theano_rng.binomial(size=input.shape, n=1,
            p=1-corruption_rate,
            dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_rate, learning_rate):
        tilde_x = self.get_corrupted_input(self.x, corruption_rate)
        hidden_value = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(hidden_value)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z))
        cost = T.mean(L)
        gparams = T.grad(cost, self.params)
        updates = [(p, p - learning_rate*g) for p, g in zip(self.params, gparams)]
        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=15,
    dataset='mnist.pkl.gz',
    batch_size=20, output_folder="dA_plots"):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    index = T.lscalar()
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    os.chdir(output_folder)

    #################################
    # Build the model no corruption #
    #################################
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    da = dA(
        rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500
        )
    cost, updates = da.get_cost_updates(
        corruption_rate=0.,
        learning_rate=learning_rate
        )
    train_da = theano.function(
        [index], cost, updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index + 1)*batch_size]
        }
        )
    start_time = timeit.default_timer()

    ############
    # training #
    ############
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print "Training epoch %d , cost " % epoch, np.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file' + 
        os.path.split(__file__)[1] + 
        ' ran for %0.2fm' % (training_time/60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
            img_shape=(28,28), tile_shape=(10,10),
            tile_spacing=(1,1)
        ))
    image.save('filters_corruption_0.png')

    ################################
    # Training with corruption 30% #
    ################################
    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28*28,
        n_hidden=500
        )
    cost, updates = da.get_cost_updates(corruption_rate=0.3,
        learning_rate=learning_rate)


    train_da = theano.function(
        [index], cost, updates=updates,
        givens={
            x: train_set_x[index*batch_size:(index + 1)*batch_size]
        }
        )
    start_time = timeit.default_timer()

    ############
    # training #
    ############
    for epoch in xrange(training_epochs):
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        print "Training epoch %d , cost " % epoch, np.mean(c)

    end_time = timeit.default_timer()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 0.3 corruption code for file' + 
        os.path.split(__file__)[1] + 
        ' ran for %0.2fm' % (training_time/60.))
    image = Image.fromarray(
        tile_raster_images(X=da.W.get_value(borrow=True).T,
            img_shape=(28,28), tile_shape=(10,10),
            tile_spacing=(1,1)
        ))
    image.save('filters_corruption_30.png')
    os.chdir("../")


if __name__ == "__main__":
    test_dA()