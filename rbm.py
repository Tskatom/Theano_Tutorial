""" Implement the Restricted Boltzman machine"""
try:
    import PIL.Image as Image
except:
    import Image

import numpy as np
import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams

from utils import tile_raster_images
from utils import load_data
import timeit


class RBM(object):
    def __init__(
        self, 
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        hbias=None,
        vbias=None,
        np_rng=None,
        theano_rng=None):
        self.n_hidden = n_hidden
        self.n_visible = n_visible

        if np_rng is None:
            np_rng = np.random.RandomState(1234)
        if theano_rng  is None:
            theano_rng = RandomStreams(np_rng.randint(2**30))

        if W is None:
            w_bound = 4*np.sqrt(6./(n_visible + n_hidden))
            W = theano.shared(np.asarray(np_rng.uniform(-w_bound, w_bound, 
                size=(n_visible, n_hidden)), dtype=theano.config.floatX),
            borrow=True, name='W')
        if hbias is None:
            h_bias_value = np.zeros((n_hidden,), dtype=theano.config.floatX)
            hbias = theano.shared(h_bias_value, name='hbias', borrow=True)
            
        if vbias is None:
            v_bias_value = np.zeros((n_visible,), dtype=theano.config.floatX)
            vbias = theano.shared(v_bias_value, name='vbias', borrow=True)

        if input:
            self.input = input
        else:
            self.input = T.matrix('input')

        self.W = W
        self.hbias = hbias
        self.vbias = vbias
        self.theano_rng = theano_rng

        self.params = [self.W, self.hbias, self.vbias]


    def free_energy(self, v_sample):
        """ function to compute the free energy"""
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        """
        propogate the activations from visible units to hidden units
        """
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        """ Infer the state of hidden units given visible units"""
        # compute the activations of hidden units given a sample of visible units
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        # get a sample of hidden units given their activation
        # The theano_rng.binomial returns a symbolic sample of dtype int64
        # by default. If want to use GPU, we need to cast it as floatX
        h1_sample = self.theano_rng.binomial(p=h1_mean, n=1, size=h1_mean.shape,dtype=theano.config.floatX)

        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        """Propogate the activations from hidden units to visible units"""
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, T.nnet.sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        """Sample the state of visible units given hidden units"""
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = self.theano_rng.binomial(p=v1_mean, size=v1_mean.shape, n=1, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        """This function implements one step of Gibbs sampling, start from the hidden state"""
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        """This function implements one step of gibbs sampling, start from visible state"""
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        """
        This function implements one step of CD-k or PCD-k

        :param persistent: None for CD, for PCD, shared variable containing old states of Gibbs chain.
        This must be a shared varibale of size (batch-size, number of hidden units)

        :param k: number of steps to do in CD-k/PCD-k

        returns a proxy for the cost and the update rules for weights and bias but also 
        an update of the shared variable used to store the persistent chain, if one is used
        """
        # compute the positive phase
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)

        # decide how to initiate persistent chain
        # for CD, we use the newly generated hidden units
        # for PCD, we initialize from the old state of the chain
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent


        # perform actual negative phase
        # in order to implement CD-k/PCD-k, we need to scan over 
        # the function that inplements one gibbs step k-times
        # The scan function will return the entire Gibbs chain
        ([
            pre_sigmoid_nvs,
            nv_means,
            nv_samples,
            pre_sigmoid_nhs,
            nh_means,
            nh_samples
         ],
         updates) = theano.scan(
            self.gibbs_hvh,
            outputs_info=[None, None, None, None, None, chain_start],
            n_steps=k
         )

        # determine gradients on RBM parameters
        # note that we only need the sample at the end of the chain
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

        # we must not compute the gradient through the gibbs sampling
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])

        # construct the updates dictionary
        for gparam, param in zip(gparams, self.params):
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        if persistent:
            # only work if persistent is a shared variable
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        """Stochastic approximation to the pseudo-likelihood"""
        # index of bit i in expression P(x_i | x_{\i})
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')

        # binarize the input image by rounding to nearest integer
        xi = T.round(self.input)

        # calculate free energy for the given bit configuration
        fe_xi = self.free_energy(xi)

        # flip bit x_i of matrix xi and preserve all other bits x_{\i}
        # Equivalent to xi[:, bit_i_idx] = 1 - xi[:, bit_i_idx]
        # , but assigns the result to xi_flip, instead of working in place on xi.
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])

        # calculate free energy with bit flipped
        fe_xi_flip = self.free_energy(xi_flip)

        # equivalent to e^(-FE(x_i)) / (e^(-FE(x_i)) + e^(-FE(x_{\i})))
        cost = T.mean(self.n_visible * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv):
        cross_entropy = T.mean(
            T.sum(
                    self.input * T.log(T.nnet.sigmoid(pre_sigmoid_nv)) + 
                    (1 - self.input) * T.log(1 - T.nnet.sigmoid(pre_sigmoid_nv)),
                    axis=1
                )
            )
        return cross_entropy


def test_rbm(learning_rate=0.1, training_epochs=15,
    dataset='mnist.pkl.gz', batch_size=20,
    n_chains=20, n_samples=10, output_folder='rbm_plots',
    n_hidden=500):
    
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    test_set_x, test_set_y = datasets[2]

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variable for the data
    index = T.lscalar()
    x = T.matrix('x')

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2**30))

    # initialize storage for the persistent chain (state = hidden layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)

    # construct the RBM class
    rbm = RBM(input=x, n_visible=28*28,
        n_hidden=n_hidden, np_rng=rng, theano_rng=theano_rng)
    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate, 
        persistent=persistent_chain, k=15)

    #############################
    # Train the RBM #############
    #############################
    print "...Train the RBM"
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    os.chdir(output_folder)

    # construct RBM training 
    train_rbm = theano.function(
        [index], cost, updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index+1) * batch_size]
        }, name='train_rbm'
        )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through the training epochs
    for epoch in xrange(training_epochs):
        # go through the training set
        mean_cost = []
        for batch_index in xrange(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print 'Training epoch %d with mean cost %f' % (epoch, np.mean(mean_cost))
        plotting_start = timeit.default_timer()
        # Construct image from the weight matrix
        image = Image.fromarray(
            tile_raster_images(
                X=rbm.W.get_value(borrow=True).T,
                img_shape=(28, 28),
                tile_shape=(10, 10),
                tile_spacing=(1, 1)
            )
        )
        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time

    print ('Training took %f minutes' % (pretraining_time / 60.))
    # end-snippet-5 start-snippet-6

    #############################
    # Sampling from RBM #########
    #############################

    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        np.asarray(
                test_set_x.get_value(borrow=True)[test_idx:test_idx+n_chains],
                dtype=theano.config.floatX
            )
        )
    plot_every = 1000
    # define one step of Gibbs sampling define a function that does plot_every steps before
    # return sample for plotting
    (
        [
            presig_hids,
            hid_mfs,
            hid_samples,
            presig_vis,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(

        rbm.gibbs_vhv,
        outputs_info=[None, None, None, None, None, persistent_vis_chain],
        n_steps=plot_every
    )

    sample_fn = theano.function(
        [],
        [
            vis_mfs[-1],
            vis_samples[-1]
        ],
        updates=updates,
        name='smaple_fn'
        )

    # create space to store the image for plotting
    image_data = np.zeros(
        (29 * n_samples + 1, 29 * n_chains - 1),
        dtype='uint8'
    )
    for idx in xrange(n_samples):
        # generate `plot_every` intermediate samples that we discard,
        # because successive samples in the chain are too correlated
        vis_mf, vis_sample = sample_fn()
        print ' ... plotting sample ', idx
        image_data[29 * idx:29 * idx + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, n_chains),
            tile_spacing=(1, 1)
        )

    # construct image
    image = Image.fromarray(image_data)
    image.save('samples.png')
    # end-snippet-7
    os.chdir('../')

if __name__ == "__main__":
    test_rbm()
