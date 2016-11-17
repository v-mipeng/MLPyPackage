import numpy
import theano
from blocks import theano_expressions
from blocks.bricks import Identity, Tanh, MLP
from blocks.bricks.lookup import LookupTable
from blocks.graph import apply_dropout
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import ComputationGraph
from theano import tensor

from pml.model.base import AbstractModel
from pml.model.simple import LRModel


class SogouMultiTaskModel(AbstractModel):
    '''
    Train sample by user
    '''

    def __init__(self, config, dataset, **kwargs):
        super(SogouMultiTaskModel, self).__init__(**kwargs)
        self.config = config
        self.dataset = dataset

    def _define_inputs(self, *args, **kwargs):
        self.query = tensor.imatrix('query')
        self.query_mask = tensor.matrix('query_mask', dtype=theano.config.floatX)
        self.age = tensor.ivector('age')
        self.age_mask = tensor.vector('age_mask', dtype=theano.config.floatX)
        self.gender = tensor.ivector('gender')
        self.gender_mask = tensor.vector('gender_mask', dtype=theano.config.floatX)
        self.edu = tensor.ivector('edu')
        self.edu_mask = tensor.vector('edu_mask', dtype=theano.config.floatX)
        self.age_noised_label = tensor.matrix('age_noised_label', dtype=theano.config.floatX)
        self.gender_noised_label = tensor.matrix('gender_noised_label', dtype=theano.config.floatX)
        self.edu_noised_label = tensor.matrix('edu_noised_label', dtype=theano.config.floatX)

    def _build_bricks(self):
        # Build lookup tables
        self.word_embed = self._embed(len(self.dataset.word2index), self.config.word_embed_dim, name="word_embed")
        self.age_mlp = self._build_output_mlp(name='age_mlp',
                                              activations=[Identity()],
                                              dims=[self.config.age_transform_dim, self.config.age_label_num],
                                              use_bias=True)
        self.gender_mlp = self._build_output_mlp(name='gender_mlp',
                                                 activations=[Identity()],
                                                 dims=[self.config.gender_transform_dim, self.config.gender_label_num],
                                                 use_bias=True)
        self.edu_mlp = self._build_output_mlp(name='edu_mlp',
                                              activations=[Identity()],
                                              dims=[self.config.edu_transform_dim, self.config.edu_label_num],
                                              use_bias=True)
        self.age_transform = self._build_transform(name='age_transform',
                                                   input_dim=self.config.edu_transform_dim,
                                                   output_dim=self.config.age_transform_dim)
        self.gender_transform = self._build_transform(name='gender_transform',
                                                      input_dim=self.config.word_embed_dim,
                                                      output_dim=self.config.gender_transform_dim)
        self.edu_transform = self._build_transform(name='edu_transform',
                                                   input_dim=self.config.word_embed_dim,
                                                   output_dim=self.config.edu_transform_dim)

    def _build_transform(self, name, input_dim, output_dim, trans_times=1, activations=None):
        '''Transfer word embedding.

        This is designed to isolate embeddings of each task

        :param name: str
                Brick name
        :param input_dim: int
                Brick input dimension
        :param output_dim: int
                Brick output dimension
        :param trans_times: int
                Number of mlp layer
        :return: MLP
        '''
        if activations is None:
            activations = [Tanh()]*trans_times
        transform = MLP(activations=activations, dims=[input_dim] + [output_dim] * trans_times, name=name)
        transform.weights_init = IsotropicGaussian(std=numpy.sqrt(6) / numpy.sqrt(input_dim + output_dim))
        transform.biases_init = Constant(0)
        transform.initialize()
        return transform

    def _build_output_mlp(self, name, activations, dims, use_bias=True):
        mlp = MLP(activations=activations, dims=dims, name=name, use_bias=use_bias)
        mlp.weights_init = IsotropicGaussian(
            std= numpy.sqrt(len(dims)) / numpy.sqrt(numpy.array(dims).sum()))
        mlp.biases_init = Constant(0)
        mlp.initialize()
        return mlp

    def _get_classifier_input(self):
        '''Get input of last mlp layer'''

        text_vec = self._get_vectorized_query(self.query)
        encoded_queries = self._encode_query(text_vec, self.query_mask)
        return encoded_queries

    def _get_vectorized_query(self, mat):
        '''Get embedding representation of query'''
        return self.word_embed.apply(mat)

    def _encode_query(self, text_vec, text_vec_mask):
        '''Encode vector representation of textual'''

        norm = text_vec_mask.sum(axis=1)[:, None]

        # Get representation of gender by mean pooling
        gender_mean = text_vec.sum(axis=1) / (norm + 1e-9)
        gender_vec = gender_mean

        edu_transformed_text_vec = self.edu_transform.apply(text_vec) * text_vec_mask[:, :, None]
        edu_mean = edu_transformed_text_vec.sum(axis=1) / (norm + 1e-9)
        edu_vec = edu_mean

        # Get representation of age
        age_transformed_text_vec = self.age_transform.apply(edu_transformed_text_vec) * text_vec_mask[:, :, None]
        age_mean = age_transformed_text_vec.sum(axis=1) / (norm + 1e-9)
        age_vec = edu_vec

        return age_vec, gender_vec, edu_vec

    def _embed(self, sample_num, dim, name, *args, **kwargs):
        embed = LookupTable(sample_num, dim, name=name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed

    def _get_pred_dist(self, input_vec, *args, **kwargs):

        age_mlp_outputs = self.age_mlp.apply(input_vec[0])
        gender_mlp_outputs = self.gender_mlp.apply(input_vec[1])
        edu_mlp_outputs = self.edu_mlp.apply(input_vec[2])
        return age_mlp_outputs, gender_mlp_outputs, edu_mlp_outputs

    def _get_cost(self):
        self._get_class_weight()
        self._get_train_cost(self._get_classifier_input())
        self._apply_reg()
        self._apply_noise()

    def _get_train_cost(self, input_vec):
        age_mlp_outputs, gender_mlp_outputs, edu_mlp_outputs = self._get_pred_dist(input_vec)

        age_preds = age_mlp_outputs
        self.age_pred = tensor.argmax(age_preds, axis=1)
        age_cost = self.categorical_cross_entropy(self.age_noised_label,
                                                  age_preds) * self.age_mask * self.age_weight[self.age]
        age_cost = age_cost.sum() / (self.age_mask.sum() + 1e-9)
        age_cost.name = 'age_cost'
        age_accuracy = (tensor.eq(self.age, self.age_pred) * self.age_mask).sum() / (self.age_mask.sum() + 1e-9)
        age_accuracy.name = 'age_accuracy'

        # Get cost on gender class
        gender_preds = gender_mlp_outputs
        self.gender_pred = tensor.argmax(gender_preds, axis=1)
        gender_cost = self.categorical_cross_entropy(self.gender_noised_label,
                                                  gender_preds) * self.gender_mask * self.gender_weight[self.gender]
        gender_cost = gender_cost.sum() / (self.gender_mask.sum() + 1e-9)
        gender_cost.name = 'gender_cost'
        gender_accuracy = (tensor.eq(self.gender, self.gender_pred) * self.gender_mask).sum() / (self.gender_mask.sum() + 1e-9)
        gender_accuracy.name = 'gender_accuracy'

        # Get cost on edu class
        edu_preds = edu_mlp_outputs
        self.edu_pred = tensor.argmax(edu_preds, axis=1)
        edu_cost = self.categorical_cross_entropy(self.edu_noised_label, edu_preds) * self.edu_mask * self.edu_weight[self.edu]
        edu_cost = edu_cost.sum() / (self.edu_mask.sum()+1e-9)
        edu_cost.name = 'edu_cost'
        edu_accuracy = (tensor.eq(self.edu, self.edu_pred) * self.edu_mask).sum() / (self.edu_mask.sum() + 1e-9)
        edu_accuracy.name = 'edu_accuracy'
        # Get sum cost
        # Balance tasks
        cost = tensor.add(1.*age_cost, 1.7*gender_cost, 1.*edu_cost)
        average_accuracy = tensor.add(1. * age_accuracy, 1. * gender_accuracy, 1. * edu_accuracy) / 3.
        cost.name = 'cost'
        average_accuracy.name = 'average_accuracy'
        self._train_cg_generator = cost
        self._train_monitors = [cost, average_accuracy]
        self._valid_monitors = [average_accuracy, cost]
        self._predict_monitors = [age_preds, gender_preds, edu_preds]

    def _get_class_weight(self):
        self.age_weight = self._get_weight('age_weight')
        self.gender_weight = self._get_weight('gender_weight')
        self.edu_weight = self._get_weight('edu_weight')
        self.consider_constant.extend([self.age_weight, self.gender_weight, self.edu_weight])

    def _get_weight(self, name):
        freqs = []
        try:
            for i in range(self.dataset.get_label_num(name)):
                freqs.append(self.dataset.get_label2freq(name))
            freqs = numpy.array(freqs, dtype=theano.config.floatX)
            weights = (freqs.mean()/freqs)**0.
            return theano.shared(weights, name+'_weight')
        except:
            raise Exception('Label should be integer for training!')

    def _apply_noise(self):
        '''Apply dropout on computing graph of train, valid and test outputs (default not)'''
        cgs = ComputationGraph(self.train_cg_generator)
        mlps = [self.age_transform, self.gender_transform, self.edu_transform]
        mlp_params = []
        for mlp in mlps:
            mlp_params += [transformation.W for transformation in mlp.linear_transformations]
        cgs = apply_dropout(cgs, mlp_params, self.config.mlp_dropout)
        embed_params = [self.word_embed.W]
        cgs = apply_dropout(cgs, embed_params, self.config.embed_dropout)
        self._train_cg_generator, = cgs.outputs

    def _apply_reg(self, params=None):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and word embedding) to computing
        graph of self.cg_generator
        :param params: A list of parameters to which regularization applied
        '''
        mlps = [self.age_mlp, self.gender_mlp, self.edu_mlp, self.age_transform, self.gender_transform,
                self.edu_transform]
        mlp_params = []
        for mlp in mlps:
            mlp_params += [transformation.W for transformation in mlp.linear_transformations]
        self.cg_generator = self.cg_generator + self.config.l2_norm_embed * theano_expressions.l2_norm(
            tensors=[self.word_embed.W]) ** 2 + \
                            self.config.l2_norm_mlp * theano_expressions.l2_norm(tensors=mlp_params) ** 2


class SogouLRModel(LRModel):
    def __init__(self, config, dataset):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        super(SogouLRModel, self).__init__(
            label_num=dataset.label_num,
            input_dim=dataset.token_num,
            input_name='query',
            output_name='combined_label',
            input_mask_name='query_mask',
            noised_output_name='combined_label_noised_label',
            norm_type='l2_norm',
            norm_scale=config.l2_norm,
            label_weight=None)

import os

from pml.config.base import BasicConfig


class SogouConfig(BasicConfig):
    def __init__(self):
        super(SogouConfig, self).__init__()

        cur_path = os.path.abspath(__file__)
        self.project_dir = cur_path[0:cur_path.index('source/pml')]

        # Training process parameters
        self.print_freq = 50

        # Query sample
        self.query_sample_num = 50

        # Output noise
        self.age_max_noise = 0.20

        self.age_decay_rate = 2.

        self.gender_max_noise = 0.10

        self.gender_decay_rate = 2.

        self.edu_max_noise = 0.20

        self.edu_decay_rate = 2.

        # Class balance
        self.age_up_sample_k = 0.25

        self.gender_up_sample_k = 0.0

        self.edu_up_sample_k = 0.25

        # Model save load
        self.model_load_from = os.path.join(self.project_dir,
                                       "output/model/none/query_sample_30_output_noise_030_020_030.pkl")

        self.model_save_to = os.path.join(self.project_dir, "output/model/multi_task/"
                                                              "query_sample_50_output_noise_030_020_030_stack_gender_edu_age_age_is_edu.pkl")

        self.word2vec_load_from = os.path.join(self.project_dir, 'data/word2vec.vec')

        self.tain_data_load_from = os.path.join(self.project_dir, "data/train_tok_2w.txt")

        self.predict_data_load_from = os.path.join(self.project_dir, "data/test_tok_2w.txt")

        self.predict_result_save_to = os.path.join(self.project_dir, "output/result/lr_model/result.csv")

        self.valid_proportion = 0.2
        self.test_proportion = 0.

        # Dataset parameters
        self.sparse_char_freq = 10

        # Model parameters

        self.char_embed_dim = 15

        self.query_encode_dim = 15

        self.age_transform_dim = 50

        self.gender_transform_dim = 15

        self.edu_transform_dim = 50

        # alpha of l2 norm (small) for parameter explosion
        self.l2_norm_embed = 1e-6

        self.l2_norm_mlp = 1e-6

        self.mlp_dropout = 0.5

        self.embed_dropout = 0.6

    def get_dataset(self):
        pass

