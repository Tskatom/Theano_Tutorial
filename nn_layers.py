#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano import shared
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample


"""
Implement the Neural Network Layers
"""
__author__ = "Wei Wang"
__email__ = "tskatom@vt.edu"


class HiddenLayer(object):
    """ Hidden Layer class"""
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):
        """
            :type rng: numpy.random.randomstate
            :param rng: random number generator

            :type input: theano.tensor.fvector
            :param input: input vector to the hidden layer

            :type n_in: int
            :param n_in: dimention of input

            :type n_out: int
            :param n_out: number of hiddent units

            :type activation: function
            :param activation: non-linear activation function

            :type W: None or theano shared variable
            :param W: the weights for hidden Layer

            :type b: None or theano shared variable
            :param b: the bias for hidden Layer
        """
        self.input = input,
        self.activation = activation

        if W is None:
            if self.activation.func_name == "ReLU":
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in,
                                                                       n_out)),
                                      dtype=theano.config.floatX)
            else:
                w_bound = np.sqrt(6./(n_in + n_out))
                W_values = np.asarray(rng.uniform(-w_bound,
                                                  w_bound,
                                                  size=(n_in, n_out)),
                                      dtype=theano.config.floatX)
            self.W = shared(value=W_values, name="hidden_W")
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = shared(value=b_values, name="hidden_b")
        else:
            self.b = b

        pre_activation = T.dot(input, self.W) + self.b

        self.output = self.activation(pre_activation)
        self.params = [self.W, self.b]


class MLP(object):
    """
        Multi-Layer Neural Network
    """
    def __init__(self, rng, input, n_in, n_hidden, activation, n_out):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=activation)

        # construct the LogisticRegression Layer
        self.logisticLayer = LogisticRegressionLayer(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
            )

        # cost function
        self.negative_log_likelihood = self.logisticLayer.negative_log_likelihood
        self.errors = self.logisticLayer.errors

        self.params = self.hiddenLayer.params + self.logisticLayer.params


class LogisticRegressionLayer(object):
    """
        Loginstic Regression Layer for classfication
    """
    def __init__(self, input, n_in, n_out, W=None, b=None):
        """
        :type input: theano.tensor.TensorType
        :param input: the input vector to the classifier

        :type n_in: int
        :param n_in: the dimention of input vector

        :type n_out: int
        :param n_out: the number of output classes

        :type W: theano shared variable or None
        :param W: the weights of LogisticRegression layer

        :type b: theano shared variable
        :param b: the bias vector of LogisticRegression layer
        """
        self.input = input
        if W is None:
            W_values = np.zeros((n_in, n_out), dtype=theano.config.floatX)
            self.W = shared(value=W_values, name="logis_W")
        else:
            self.W = W

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = shared(value=b_values, name="logis_b")
        else:
            self.b = b

        pre_activation = T.dot(input, self.W) + self.b

        self.p_y_given_x = T.nnet.softmax(pre_activation)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """
        the cost function of LogisticRegression layer
        :type y: theano.tensor.TensorType
        :param y: the vector which contains the correct
                  labels of the input samples
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """
        The error rate of the LogisticRegressionLayer
        :type y: theano.tensor.TensorType
        :param y: the vector which contains the correct
                  labels of the input samples
        """
        return T.mean(T.neq(self.y_pred, y))


class ConvPoolLayer(object):
    """Convolution and Max Pool Layer"""
    def __init__(self, rng, input, filter_shape, input_shape,
                 pool_size, activation):
        """
        :type rng: numpy.random.randomstate
        :param rng: the random number generator

        :type input: theano.tensor.TensorType
        :param input: input tensor

        :type filter_shape: list of int with length 4
        :param filter_shape: (number of filters, number of input feature maps,
                              filter height, filter width)
        :type input_shape: list of int with length 4
        :param input_shape: (batch_size, number input feature maps,
                             doc height[the word embedding dimention],
                             doc width[length of doc])

        :type pool_size: list of int with length 2
        :param pool_size: the shape of max pool

        :type activation: function
        :param activation: the non-linear activation function
        """
        self.input = input
        self.filter_shape = filter_shape
        self.input_shape = input_shape
        self.pool_size = pool_size
        self.activation = activation

        fan_in = np.prod(self.filter_shape[1:])
        fan_out = (filter_shape[0] *
                   np.prod(filter_shape[2:]))/np.prod(pool_size)

        if self.activation.func_name == "ReLU":
            W_values = np.asarray(rng.uniform(low=-0.01,
                                              high=0.01, size=filter_shape),
                                  dtype=theano.config.floatX)
        else:
            W_bound = np.sqrt(6./(fan_in + fan_out))
            W_values = np.asarray(rng.uniform(low=-W_bound,
                                              high=W_bound,
                                              size=filter_shape),
                                  dtype=theano.config.floatX)
        self.W = shared(value=W_values, borrow=True, name="conv_W")

        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = shared(value=b_values, borrow=True, name="conv_b")

        conv_out = conv.conv2d(input=self.input,
                               filters=self.W,
                               filter_shape=self.filter_shape,
                               image_shape=self.input_shape)
        act_conv_out = self.activation(conv_out +
                                       self.b.dimshuffle('x', 0, 'x', 'x'))
        pool_out = downsample.max_pool_2d(input=act_conv_out,
                                          ds=self.pool_size,
                                          ignore_border=True)
        self.output = pool_out
        self.params = [self.W, self.b]

    def predict(self, new_data, batch_size):
        image_shape = (batch_size, 1, self.input_shape[2], self.input_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W,
                               filter_shape=self.filter_shape,
                               image_shape=image_shape)
        act_conv_out = self.activation(conv_out +
                                       self.b.dimshuffle('x', 0, 'x', 'x'))
        pool_out = downsample.max_pool_2d(input=act_conv_out,
                                          ds=self.pool_size,
                                          ignore_border=True)
        return pool_out


def dropout_from_layer(rng, layer, p):
    """General frunction for Dropout Layer
    :type rng: numpy.random.randomstate
    :param rng: random number generator

    :type layer: theano.tensor
    :param layer: the output of the Neural Network Layer

    :type p: float
    :param p: the probability to drop out the units in the output
    """

    srng = theano.tensor.shared_randomstreams.RandomStreams(rng.randint(999999))
    mask = srng.binomial(n=1, p=1-p, size=layer.shape)
    output = layer * T.cast(mask, theano.config.floatX)
    return output


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, dropout_rate, W=None, b=None):
        super(DropoutHiddenLayer, self).__init__(
            rng=rng, input=input, n_in=n_in,
            n_out=n_out, activation=activation,
            W=W, b=b
            )
        self.output = dropout_from_layer(rng, self.output, p=dropout_rate)


class MLPDropout(object):
    """A multi Layer Neural Network with dropout"""
    def __init__(self, rng, input, layer_sizes, dropout_rates, activations):
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations

        next_layer_input = input
        next_dropout_layer_input = dropout_from_layer(rng,
                                                      input,
                                                      p=dropout_rates[0])
        layer_count = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = DropoutHiddenLayer(rng=rng,
                input=next_dropout_layer_input,
                activation=self.activations[layer_count],
                n_in=n_in,
                n_out=n_out,
                dropout_rate=dropout_rates[layer_count])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output

            # resuse the parameters from the dropout layer here
            next_layer = HiddenLayer(rng=rng,
                input=next_layer_input,
                activation=self.activations[layer_count],
                W=next_dropout_layer.W * (1 - dropout_rates[layer_count]),
                b=next_dropout_layer.b,
                n_in=n_in,
                n_out=n_out)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output
            layer_count += 1

        # construct final classification Layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegressionLayer(
            input=next_dropout_layer_input,
            n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # reuse the parameters again
        output_layer = LogisticRegressionLayer(
            input=next_layer_input,
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # drop out params
        self.params = [param for layer in self.dropout_layers
                       for param in layer.params]

    def predict(self, newdata):
        next_layer_input = newdata
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,
                                                             layer.W) +
                                                       layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) +
                                             layer.b)
        y_pred = T.argmax(p_y_given_x, axis=1)
        return y_pred

    def predict_p(self, newdata):
        next_layer_input = newdata
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                next_layer_input = self.activations[i](T.dot(next_layer_input,
                                                             layer.W) +
                                                       layer.b)
            else:
                p_y_given_x = T.nnet.softmax(T.dot(next_layer_input, layer.W) +
                                             layer.b)
        return p_y_given_x