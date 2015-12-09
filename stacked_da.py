"""
Stacked Denoising Autoencoder
"""

import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from nn_layers import LogisticRegressionLayer, HiddenLayer
from utils import load_data
from denoising_autoencoder import dA

def sigmoid(x):
    return T.nnet.sigmoid(x)

class sdA(object):
    """Stacked denoising_autoencoder"""
    def __init__(self, np_rng, theano_rng=None, n_ins=784,
        hidden_layer_sizes=[500, 500],
        n_outs=10,
        corruption_rates=[0.1, 0.1]):
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layer_sizes)

        if theano_rng is None:
            theano_rng = RandomStreams(np_rng.randint(2**30))

        self.x = T.matrix('x')
        self.y = T.ivector('y')

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
                layer_input = self.x
            else:
                input_size = hidden_layer_sizes[i-1]
                layer_input = self.sigmoid_layers[-1].output

            hidden_layer = HiddenLayer(
                np_rng, layer_input, input_size, hidden_layer_sizes[i], sigmoid)

            self.sigmoid_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

            # construc the denoising autoencoder layer
            da = dA(np_rng, theano_rng, layer_input, 
                n_visible=input_size, n_hidden=hidden_layer_sizes[i],
                W=hidden_layer.W, bhid=hidden_layer.b)
            self.dA_layers.append(da)

        # LogisticRegression Layer for classification
        self.logLayer = LogisticRegressionLayer(self.sigmoid_layers[-1].output, n_in=hidden_layer_sizes[-1],
            n_out=n_outs)
        
        self.params.extend(self.logLayer.params)
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        index = T.lscalar('index')
        corruption_rate = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size
        pretrain_fns = []
        for da in self.dA_layers:
            cost, updates = da.get_cost_updates(corruption_rate, 
                learning_rate)
            fn = theano.function(
                inputs=[index, theano.Param(corruption_rate, default=0.2),
                theano.Param(learning_rate, default=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                self.x: train_set_x[batch_begin:batch_end]
                }
                )
            pretrain_fns.append(fn)
        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[1]

        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost, self.params)
        updates = [(p, p - learning_rate*g) for p, g in 
                   zip(self.params, gparams)]
        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index*batch_size:(index+1)*batch_size],
                self.y: train_set_y[index*batch_size:(index+1)*batch_size]
            },
            name='train'
            )
        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                self.y: test_set_y[index*batch_size:(index+1):batch_size]
            },
            name='test'
            )
        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                self.y: valid_set_y[index*batch_size:(index+1)*batch_size]
            },
            name='valid'
            )

        # create a function that scans the entire validation set
        def valid_score(self):
            return [valid_score_i(i) for i in range(n_valid_batches)]

        def test_score(self):
            return [test_score_i(i) for i in range(n_test_batches)]
        return train_fn, valid_score, test_score


def test_sda(finetune_lr=0.1, pretraining_epochs=15,
    pretrain_lr=0.001, training_epochs=1000,
    dataset='mnist.pkl.gz', batchsize=1):
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0]/batchsize

    np_rng = np.random.RandomState(1234)
    sda = sdA(np_rng, n_ins=28*28, hidden_layer_sizes=[1000,1000,1000], n_outs=10)
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x, batch_size=batchsize)

    print '...pretraining model'
    start_time = timeit.default_timer()
    corruption_rates = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        for epoch in xrange(pretraining_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                    corruption=corruption_rates[i],
                    lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print np.mean(c)
    end_time = timeit.default_timer()

    print "The pretraining process run over %f minutes" % ((end_time-start_time)/60.)

    #######################
    # Fine Tune the model #
    #######################
    train_fn, validate_model, test_model = sda.build_finetune_functions(datasets, batch_size=batch_size, learning_rate=finetune_lr)
    print '... finetune the model'
    patience = 10 * n_train_batches
    patience_increases = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2.)
    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibach_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = np.mean(validation_losses)
                print "Iter %d with validation loss %f " % (iter + 1, this_validation_loss)
                if this_validation_loss <= best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increases)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = test_model()
                    test_score = np.mean(test_losses)
                    print "Iter %d with test score %f under best validation loss %f" % (iter + 1, test_score, best_validation_loss)
            if patience < iter:
                done_looping = True 
    end_time = timeit.default_timer()

    print "Optimization complete using %f m, with test performance %f under best validation loss %f in iter %d" % ((end_time - start_time)/60.,
        test_score, best_validation_loss, best_iter)


if __name__ == "__main__":
    test_sda()