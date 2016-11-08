'''
This module define some common models. Currently, only logistical regression model supported!
'''

import theano
import numpy as np
from theano import tensor
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Constant, IsotropicGaussian
from blocks import theano_expressions
from entrance import *
from pml.blocks.bricks.simple import *
from blocks.bricks import *
import warnings
from pml.blocks.bricks.simple import Vector

from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractModel(object):
    def __init__(self, *args, **kwargs):
        pass

    @property
    def train_cg_generator(self):
        '''
        Get computing graph generator for traing
        :return: TheanoVariable.scalar
            The training cost
        '''
        if hasattr(self, '_train_cg_generator'):
            return self._train_cg_generator
        else:
            raise NotImplementedError('train_cg_generator is not defined!')

    @property
    def valid_cg_generator(self):
        if hasattr(self, '_valid_cg_generator'):
            return self._valid_cg_generator
        else:
            raise NotImplementedError('valid_cg_generator is not defined!')

    @property
    def test_cg_generator(self):
        if hasattr(self, '_test_cg_generator'):
            return self._test_cg_generator
        else:
            raise NotImplementedError('test_cg_generator is not defined!')

    @property
    def consider_constant(self):
        if hasattr(self, '_consider_constant'):
            return self._consider_constant
        else:
            return []

    @property
    def train_monitors(self):
        '''
        Get variables for monitoring traing process
        :return: List of class: TheanoVariable.scalar
            The cost and error_rate
        '''
        if hasattr(self, '_train_monitors'):
            return self._train_monitors
        else:
            raise NotImplementedError('train_monitors are not defined!')

    @property
    def valid_monitors(self):
        '''
        Get symbol variables to build validation functions on validation data set. 
        :return: List of class: TheanoVariable.scalar
            Commonly, the cost and error_rate
        '''
        if hasattr(self, '_valid_monitors'):
            return self._valid_monitors
        else:
            raise NotImplementedError('valid_monitors are not defined!')

    @property
    def test_monitors(self):
        '''
          Get symbol variables to build testation functions on test data set.
          :return: List of class: TheanoVariable.scalar
              Commonly, the most likely prediction of given input
          '''
        if hasattr(self, '_test_monitors'):
            return self._test_monitors
        else:
            raise NotImplementedError('test_monitors are not defined!')

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs(*args, **kwargs)
        self._build_bricks(*args, **kwargs)

    def _define_inputs(self, *args, **kwargs):
        '''
        Define the input of this model
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError('_define_inputs is not defined!')

    def _build_bricks(self, *args, **kwargs):
        '''
        Define bricks of this model
        :param args:
        :param kwargs:
        :return:
        '''
        raise NotImplementedError('_build_bricks is not defined')

    def log_probabilities(self, input_):
        """Normalize log-probabilities.

        Converts unnormalized log-probabilities (exponents of which do not
        sum to one) into actual log-probabilities (exponents of which sum
        to one).

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row contains unnormalized log-probabilities of a
            distribution.

        Returns
        -------
        output : :class:`~theano.Variable`
            A matrix with normalized log-probabilities in each row for each
            distribution from `input_`.

        """
        shifted = input_ - input_.max(axis=1, keepdims=True)
        return shifted - tensor.log(
            tensor.exp(shifted).sum(axis=1, keepdims=True))

    def categorical_cross_entropy(self, y, x):
        """Computationally stable cross-entropy for pre-softmax values.

        Parameters
        ----------
        y : :class:`~tensor.TensorVariable`
            In the case of a matrix argument, each row represents a
            probabilility distribution. In the vector case, each element
            represents a distribution by specifying the position of 1 in a
            1-hot vector.
        x : :class:`~tensor.TensorVariable`
            A matrix, each row contains unnormalized probabilities of a
            distribution.

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            A vector of cross-entropies between respective distributions
            from y and x.

        """
        x = self.log_probabilities(x)
        if y.ndim == x.ndim - 1:
            indices = tensor.arange(y.shape[0]) * x.shape[1] + y
            cost = -x.flatten()[indices]
        elif y.ndim == x.ndim:
            cost = -(x * y).sum(axis=1)
        else:
            raise TypeError('rank mismatch between x and y')
        return cost


class LRModel(AbstractModel):
    def __init__(self,
                 label_num,
                 input_dim,
                 input_name,
                 output_name,
                 input_mask_name = None,
                 noised_output_name = None,
                 norm_type='l2_norm',
                 norm_scale=1e-4,
                 label_weight = None,
                 *args, **kwargs):
        '''
        Define a logistical regression model

        :param label_num: int
                The number of label of this model
        :param input_dim: int
                Define the dimension of the input
        :param input_name: str
                Define the name of input. This is used to interact with fuel.stream object
        :param output_name: str
                Define the name of output.
        :param input_mask_name: str
                Define the name of the input mask. If not given, input_name+'_mask' is assigned
        :param noised_output_name: str
                Define the name of the noised output which is multi-hot output.
                If given, this value will be used to calculate cost otherwise, output is used.
        :param norm_type: str
                Norm type for regularization, options: l2_norm (default), l1_norm, None.
                If None is assigned, no regularization is applied
        :param norm_scale: float
                Regularization degree is defined by: norm_scale*norm_value
                If norm_type is None, this value is useless.
        :param label_weight: list, or tuple of float or one dimension float type numpy.array
                This define the weight of each label. The cost of each prediction on label is
                weighted by label_weight[label]
        '''
        super(LRModel, self).__init__(*args, **kwargs)
        self.label_num = label_num
        self.input_dim = input_dim
        self.input_name = input_name
        if input_mask_name is None:
            input_mask_name = input_name + '_mask'
        self.input_mask_name = input_mask_name
        self.output_name = output_name
        self.noised_output_name = noised_output_name
        self.norm_type = norm_type
        self.norm_scale = norm_scale
        if label_weight is None:
            self.label_weight = theano.shared(np.ones((self.label_num,),dtype=theano.config.floatX), name='label_weight')
        else:
            self.label_weight = theano.shared(np.array(label_weight, dtype=theano.config.floatX), name = 'label_weight')
        self._consider_constant = [self.label_weight]
        self._build_model()

    def _build_model(self, *args, **kwargs):
        # Define inputs
        self._define_inputs()
        self._build_bricks()
        self._get_cost()
        self._apply_reg()

    def _define_inputs(self, *args, **kwargs):
        self.input = tensor.imatrix(self.input_name)
        self.input_mask = tensor.matrix(self.input_mask_name, dtype=theano.config.floatX)
        self.output = tensor.ivector(self.output_name)
        if self.noised_output_name is not None:
            self.noised_output = tensor.matrix(self.noised_output_name, dtype=theano.config.floatX)
        
    def _build_bricks(self, *args, **kwargs):
        # Build lookup tables
        self.input_lookup = self._build_lookup(name = self.input_name+'_lookup',
                                               word_num=self.input_dim,
                                               dim = self.label_num)
        self.input_bias = Vector(name=self.input_name+'_bias', dim=self.label_num)
        self.input_bias.biases_init = Constant(0.)
        self.input_bias.initialize()

    def _build_lookup(self, name, word_num, dim=1, *args, **kwargs):
        lookup = LookupTable(length=word_num, dim=dim, name=name)
        lookup.weights_init = Constant(1. / word_num ** 0.25)
        lookup.initialize()
        return lookup

    def _get_pred_dist(self, *args, **kwargs):

        outputs = self.input_bias.apply((self.input_lookup.apply(self.input) * self.input_mask[:, :, None]).sum(axis=1))
        return outputs

    def _get_cost(self, *args, **kwargs):
        self._get_train_cost(*args, **kwargs)
        self._get_valid_cost(*args, **kwargs)
        self._get_test_cost(*args, **kwargs)

    def _get_train_cost(self, *args, **kwargs):
        pred_output = self._get_pred_dist(*args, **kwargs)

        self.pred = tensor.argmax(pred_output, axis=1)
        if self.noised_output_name is not None:
            cost = self.categorical_cross_entropy(self.noised_output, pred_output) * self.label_weight[self.output]
        else:
            cost = self.categorical_cross_entropy(self.output, pred_output) * self.label_weight[self.output]
        cost = cost.mean()
        cost.name = 'cost'
        accuracy = tensor.eq(self.output, self.pred).mean()
        accuracy.name = 'accuracy'
        self._train_monitors = [cost, accuracy]
        self._train_cg_generator = cost

    def _get_valid_cost(self, *args, **kwargs):
        self._valid_monitors = self.train_monitors
        self._valid_cg_generator = self.pred

    def _get_test_cost(self, *args, **kwargs):
        self._test_cg_generator = self.pred
        self._test_monitors = [self.pred]

    def _apply_reg(self, params=None, *args, **kwargs):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and word embedding) to computing
        graph of self.cg_generator
        :param params: A list of parameters to which regularization applied
        '''
        if self.norm_type is not None:
            params = [self.input_lookup.W]
            if self.norm_type == 'l2_norm':
                self._train_cg_generator = self._train_cg_generator + self.norm_scale * theano_expressions.l2_norm(tensors=params) ** 2
            elif self.norm_type == 'l1_norm':
                norm = 0.
                for param in params:
                    norm += tensor.abs_(param).sum()
                self._train_cg_generator = self._train_cg_generator + self.norm_scale * norm
            else:
                raise ValueError('{0} norm type is not supported!'.format(self.norm_type))
        else:
            pass