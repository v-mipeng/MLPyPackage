'''This module define some common models. Currently, only logistical regression model supported!'''

import numpy as np
import theano
from blocks import theano_expressions
from blocks.bricks.lookup import LookupTable
from blocks.initialization import Constant
from theano import tensor

from base import AbstractModel
from pml.blocks.bricks.simple import Vector


class LRModel(AbstractModel):
    def __init__(self,
                 label_num,
                 input_dim,
                 input_name,
                 output_name,
                 input_mask_name=None,
                 noised_output_name=None,
                 norm_type='l2_norm',
                 norm_scale=1e-4,
                 label_weight=None,
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

    def _define_inputs(self, *args, **kwargs):
        self.input = tensor.imatrix(self.input_name)
        self.input_mask = tensor.matrix(self.input_mask_name, dtype=theano.config.floatX)
        self.output = tensor.ivector(self.output_name)
        if self.noised_output_name is not None:
            self.noised_output = tensor.matrix(self.noised_output_name, dtype=theano.config.floatX)
        
    def _build_bricks(self, *args, **kwargs):
        # Build lookup tables
        self.input_lookup = self._build_lookup(name=self.input_name+'_lookup',
                                               word_num=self.input_dim,
                                               dim=self.label_num)
        self.input_bias = Vector(name=self.input_name+'_bias', dim=self.label_num)
        self.input_bias.biases_init = Constant(0.)
        self.input_bias.initialize()

    def _build_lookup(self, name, word_num, dim=1):
        lookup = LookupTable(length=word_num, dim=dim, name=name)
        lookup.weights_init = Constant(1. / word_num ** 0.25)
        lookup.initialize()
        return lookup

    def _get_pred_dist(self):
        outputs = self.input_bias.apply((self.input_lookup.apply(self.input) * self.input_mask[:, :, None]).sum(axis=1))
        return outputs

    def _get_train_cost(self):
        pred_output = self._get_pred_dist()

        self.pred = tensor.argmax(pred_output, axis=1)
        self.pred.name = self.output_name
        if self.noised_output_name is not None:
            cost = self.categorical_cross_entropy(self.noised_output, pred_output) * self.label_weight[self.output]
        else:
            cost = self.categorical_cross_entropy(self.output, pred_output) * self.label_weight[self.output]
        cost = cost.mean()
        cost.name = 'cost'
        accuracy = tensor.eq(self.output, self.pred).mean()
        accuracy.name = 'accuracy'
        self._train_cg_generator = cost
        self._train_monitors = [accuracy, cost]
        self._valid_monitors = [accuracy]
        self._predict_monitor = [self.pred]

    def _apply_reg(self, params=None):
        '''Apply regularization (default L2 norm) on parameters (default user, hashtag and word embedding) to computing
           graph of self.cg_generator

        :param params: A list of parameters to which regularization applied
        '''
        if self.norm_type is not None:
            params = [self.input_lookup.W]
            if self.norm_type == 'l2_norm':
                self._train_cg_generator += self.norm_scale * theano_expressions.l2_norm(tensors=params) ** 2
            elif self.norm_type == 'l1_norm':
                norm = 0.
                for param in params:
                    norm += tensor.abs_(param).sum()
                self._train_cg_generator += self.norm_scale * norm
            else:
                raise ValueError('{0} norm type is not supported!'.format(self.norm_type))
        else:
            pass


