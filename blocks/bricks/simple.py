"""Some of the simplest individual bricks."""
import logging
import numpy as np

from theano import tensor


from blocks.bricks.base import application, Brick, lazy
from blocks.bricks.interfaces import Activation, Feedforward, Initializable
from blocks.bricks.interfaces import Random

from blocks.bricks.wrappers import WithExtraDims
from blocks.roles import add_role, WEIGHT, BIAS
from blocks.utils import shared_floatx_nans
from blocks.initialization import *
from blocks.bricks import *


logger = logging.getLogger(__name__)


class LinearTensor(Initializable, Feedforward):
    def __init__(self, dims, *args, **kwargs):
        super(LinearTensor, self).__init__(*args, **kwargs)
        self.dims = dims

    def _allocate(self):
        for i in range(self.dims[0]):
            W = shared_floatx_nans(self.dims[1:], name='W_{0}'.format(i))
            add_role(W, WEIGHT)
            self.parameters.append(W)

    @property
    def W(self):
        return self.parameters[0:self.dims[0]]

    def _initialize(self):
        for i in range(self.dims[0]):
            W = self.parameters[i]
            self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        output = None
        for i in range(self.dims[0]):
            W = self.parameters[i]
            if output is None:
                # tmp = input_[:,i][:,None,None]
                output = input_[:, i][:,None,None]*W[None,:,:]
            else:
                output += input_[:, i][:,None,None]*W[None,:,:]
        return output


class Vector(Bias):
    '''
    Construct a float type shared vector
    '''
    def __init__(self, *args, **kwargs):
        super(Vector, self).__init__(*args, **kwargs)

    @property
    def W(self):
        b, = self.parameters
        return b


class BaseAttention(Initializable, Feedforward):
    r"""A linear attention with optional bias.

        weight_of_X = f(XWy + b)

        where X is a vector or matrix of dimension: N*input_dim, y is a vector of dimension: attention_dim,
        W is a matrix of dimension (input_dim, attention_dim) and b is a scalar type bias.

        The function f is defined by activation.apply

       """

    @lazy(allocation=['input_dim', 'attention_dim'])
    def __init__(self, input_dim, attention_dim, activation=None, **kwargs):
        super(BaseAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        if activation is None:
            self.activation = Softmax()
        else:
            self.activation = activation
        self.children = [self.activation.apply.brick]

    @property
    def W(self):
        return self.parameters[0]

    @property
    def b(self):
        return self.parameters[1]

    def _allocate(self):
        W = shared_floatx_nans((self.input_dim, self.attention_dim), name='W')
        add_role(W, WEIGHT)
        self.parameters.append(W)
        self.add_auxiliary_variable(W.norm(2), name='W_norm')
        if self.use_bias:
            b = shared_floatx_nans((1,), name='b')
            add_role(b, BIAS)
            self.parameters.append(b)

    def _initialize(self):
        if self.use_bias:
            W, b = self.parameters
            if self.biases_init is None:
                self.biases_init = Constant(0.)
            self.biases_init.initialize(b, self.rng)
        else:
            W, = self.parameters
        if self.weights_init is None:
            self.weights_init = IsotropicGaussian(std = np.sqrt(6)/np.sqrt(self.input_dim + self.attention_dim))
        self.weights_init.initialize(W, self.rng)

    @application(inputs=['input_', 'attention_'], outputs=['output'])
    def apply(self, input_, attention_):
        """Apply the linear transformation.

        Parameters
        ----------
        input_ : :class:`~tensor.TensorVariable`
            The input on which to apply the transformation

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            The transformed input plus optional bias

        """
        if self.use_bias:
            W, b = self.parameters
        else:
            W, = self.parameters
        tmp = tensor.dot(W, attention_)
        output = tensor.dot(input_, tmp)
        if self.use_bias:
            output += b[0]
        output = self.activation.apply(output)
        return output

    def get_dim(self, name):
        if name == 'input_':
            return self.input_dim
        if name == 'attention':
            return self.attention_dim
        super(BaseAttention, self).get_dim(name)


