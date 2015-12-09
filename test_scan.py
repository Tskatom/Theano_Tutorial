import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

theano.config.exception_verbosity='high'

theano_rng = RandomStreams(1234)
rng = np.random.RandomState(12)

w_value = np.ones((2,1), dtype=theano.config.floatX)
W = theano.shared(value=w_value)
hbias = theano.shared(value=np.asarray(np.arange(1), dtype=theano.config.floatX))
vbias = theano.shared(value=np.asarray(np.arange(2), dtype=theano.config.floatX))

def one_step(v_sample):
    h_mean = T.nnet.sigmoid(T.dot(v_sample, W) + hbias)
    h_sample = theano_rng.binomial(p=h_mean, size=h_mean.shape, n=1)
    v_mean = T.nnet.sigmoid(T.dot(h_sample, W.T) + vbias)
    v_sample = theano_rng.binomial(p=v_mean, size=v_mean.shape, n=1)
    return v_sample

sample = theano.tensor.lvector()

values, updates = theano.scan(one_step, outputs_info=sample, n_steps=10)

gibbs10 = theano.function([sample], values, updates=updates)
gibbs10_no = theano.function([sample], values, no_default_updates=True)

print gibbs10([1, 1])
print gibbs10([1, 1])
print '------'
print gibbs10_no([1, 1])
print gibbs10_no([1, 1])


def sp():
    return theano_rng.uniform(low=-1,high=1,size=(1,))

s_val, s_ups = theano.scan(sp, n_steps=2)

s_val2, s_ups2 = theano.scan(sp, n_steps=2)

s_func = theano.function([],outputs=s_val, updates=s_ups)
s_func_no = theano.function([],outputs=s_val2, no_default_updates=True)

print s_func()
print s_func()
print '===='
print s_func_no()
print s_func_no()


state = theano.shared(0)
inc = T.iscalar('inc')
accu = theano.function([inc], state, updates=[(state, state+inc)])

for i in range(3):
    print accu(1)

a = theano.shared(1)
values, updates = theano.scan(lambda: {a:a+1}, n_steps=10)
b = a + 1
c = updates[a] + 1
f = theano.function([], [b, c], updates=updates)

print(b)
print (c)
print f()
print a.get_value()    