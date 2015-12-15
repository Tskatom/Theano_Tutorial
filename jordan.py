import theano
import numpy as np
import os
import theano.tensor as T
from collections import OrderedDict

class model(object):
    def __init__(self, nh, nc, ne, de, cs):
        self.emb = theano.shared(0.2*np.random.uniform(-1.0, 1.0, \
            (ne+1, de)).astype(theano.config.floatX), borrow=True)
        self.Wx = theano.shared(0.2*np.random.uniform(-1.0, 1.0, \
            (cs*de, nh)).astype(theano.config.floatX), borrow=True)
        self.Ws = theano.shared(0.2*np.random.uniform(-1.0,1.0, \
            (nc, nh)).astype(theano.config.floatX),borrow=True)
        self.W = theano.shared(0.2*np.random.uniform(-1.0,1.0,\
            (nh,nc)).astype(theano.config.floatX),borrow=True)
        self.bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX),borrow=True)
        self.b = theano.shared(nb.zeros(nc, dtype=theano.config.floatX),borrow=True)
        self.s0 = theano.shared(nb.zeros(nc, dtype=theano.config.floatX),borrow=True)

        self.params = [self.emb, self.Wx, self.Ws, self.W, self.bh, self.b, self.s0]
        self.names = ['embedding','Wx', 'Ws', 'W', 'bh', 'b', 's0']

        idxs = T.imatrix()
        x = self.emb[idxs].reshape((idxs.shape[0], cs*de))
        y = T.iscalar('y')

        def recurrence(x_t, s_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx)\
             + T.dot(s_tm1, self.Ws) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)[0]
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[None, self.s0], \
            n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1, 0, :]
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # theano functions
        self.classify = theano.function(inputs=[index], outputs=y_pred)
        self.train = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=updates)
        self.normalize = theano.function(inputs=[],
            updates={self.emb:\
            self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):
        for param, name in zip(self.params, self.names):
            np.save(os.path.join(folder, name + '.npy'), param.get_value())
