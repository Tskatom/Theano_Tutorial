import os
import sys
import timeit
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from nn_layers import LogisticRegressionLayer, HiddenLayer
from utils import load_data
from rbm import RBM

# start snippet -1 
class DBN(object):
    """Deep belief network"""
    def __init__(self, np_rng, theano_rng=None, n_ins=784,
        hidden_layers_sizes=[500, 500], n_outs=10):
        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        if theano_rng is None:
            theano_rng = MRG_RandomStreams(np_rng.randint(2**30))

        # allocate symbolic variables for the data
        self.x = T.matrix('x')
        self.y = T.ivector('y')
        # end snippet -1

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of 
            # the hidden layer below or the input of the DBN if you
            # are on the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=np_rng,
                input=layer_input, n_in=input_size,
                n_out=hidden_layers_sizes[i],
                activation=T.nnet.sigmoid)
            # add the sigmoid layer to layer list
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            # construct an RBM that shared weights with this layer
            rbm_layer = RBM( np_rng=np_rng, 
                theano_rng=theano_rng, input=layer_input,
                n_visible=input_size,
                n_hidden=hidden_layers_sizes[i],
                W=sigmoid_layer.W,
                hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)

        # We need to add a Logistic layer on top of MLP
        self.logLayer = LogisticRegressionLayer(input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined
        # as the negative log likelihood of the logistic regression
        self.finetune_cost = self.logLayer.negative_log_likehihood(self.y)

        # compuate the gradientes with respect to the model parameters
        # symbolic variable that points to the number of errors
        # made on the mibibatch given by self.x and self.y
        self.erros = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        index = T.lscalar('index')
        learning_rate = T.scalar('lr')
        # compute number of batches
        batch_begin = index * batch_size
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:
            # get the cost and the updates list
            # using CD-k here
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=None, k=k)

            # compile the theano function
            fn = theano.function(inputs=[index, theano.Param(learning_rate, default=0.1)],
                                 outputs=cost,
                                 updates=updates,
                                 givens={
                                        self.x: train_set_x[batch_begin:batch_end]
                                        })

            pretrain_fns.append(fn)
        return pretrain_fns


    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]
        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size
        index = T.lscalar('index')
        gparams = T.grad(self.finetune_cost, self.params)
        updates = []
        for param, gparam in zip(self.params, gparams):
            updates.append((param, param - gparam*learning_rate))

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={

                self.x: train_set_x[index*batch_size:(index+1)*batch_size],
                self.y: train_set_y[index*batch_size:(index+1)*batch_size]
            }
            )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[index*batch_size:(index+1)*batch_size],
                self.y: test_set_y[index*batch_size:(index+1)*batch_size]
            }
            )

        valid_score_i = theano.function(
                [index],
                outputs=self.erros,
                givens={
                    self.x: valid_set_x[index*batch_size:(index+1)*batch_size],
                    self.y: valid_set_y[index*batch_size:(index+1)*batch_size]
                }
            )

        # construct a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


def test_DBN(finetune_lr=0.1, pretraininig_epochs=100,
    pretrain_lr=0.01, k=1, training_epochs=1000,
    dataset='mnist.pkl.gz', batch_size=10):
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    np_rng = np.random.RandomState(123)
    print '----building the model'
    dbn = DBN(np_rng=np_rng,n_ins=28*28,hidden_layers_sizes=[1000,1000,1000],
        n_outs=10)

    # start code snippet 2
    # pretrain the model
    print '---getting the pretraining functions'
    pretraining_fns = dbn.pretraining_functions(train_set_x=train_set_x,
        batch_size=batch_size,k=k)
    start_time = timeit.default_timer()
    #pretrain layer-wise
    for i in xrange(dbn.n_layers):
        for epoch in xrange(pretraininig_epochs):
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,lr=pretrain_lr))

            print 'Pre-training layer %d, epoch %d, cost' % (i, epoch)
            print np.mean(c)

    end_time = timeit.default_timer()
    print 'Pretrain phase takes time %f m' % ((end_time - start_time)/60.)

    # end code snippet 2
    # finetune phase
    print '... getting the finetuning functions'
    train_fn, valid_model, test_model = dbn.build_finetune_functions(
        datasets=datasets,
        batch_size=batch_size,
        learning_rate=finetune_lr
        )

    print '...finetune the model'
    # early-stopping parameters
    patience = 4 * n_train_batches
    patience_increase = 2.
    improvement_threshold = 0.995
    validation_frequency = min(n_train_batches, patience/2)

    best_validation_loss = np.inf
    test_score = 0.
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    while (epoch < training_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = valid_model()
                this_valid_loss = np.mean(validation_losses)
                print 'Iter %d in epch %d as batch_index %d with validation loss %f' % (iter, epoch, minibatch_index, this_valid_loss)
                if this_valid_loss < best_validation_loss:
                    if this_valid_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter*patience_increase)
                    best_validation_loss = this_valid_loss
                    best_iter = iter

                    test_losses = test_model()
                    test_score = np.mean(test_losses)

                    print 'Test Score %f under iter %d' % (test_score, iter)
            if patience <= iter:
                done_looping = True
                break
    end_time = timeit.default_timer()
    print 'Finetune finished with test score %f using time %f m' % (test_score, (end_time-start_time)/60.)

if __name__ == "__main__":
    test_DBN() 

