import numpy
import numpy as np
import theano
import theano.tensor as tensor
from blocks import theano_expressions
from blocks.bricks.recurrent import LSTM

from pml.model.base import AbstractModel
from pml.blocks.bricks.lookup import LookupTable
from pml.blocks.bricks import *
from pml.blocks.graph import apply_dropout, ComputationGraph, apply_noise
from pml.model.simple import LRModel


class SogouMultiTaskModel(AbstractModel):
    '''
    Train sample by user
    '''

    def __init__(self, config, **kwargs):
        super(SogouMultiTaskModel, self).__init__(**kwargs)
        self.config = config

    def _define_inputs(self):
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
        self.token_embed = self._embed(len(self.config.dataset.token2index), self.config.token_embed_dim, name="token_embed")
        self.age_mlp = self._build_output_mlp(name='age_mlp',
                                              activations=[Identity()],
                                              dims=[self.config.age_transform_dim, self.config.dataset.get_label_num('age')],
                                              use_bias=True)
        self.gender_mlp = self._build_output_mlp(name='gender_mlp',
                                                 activations=[Identity()],
                                                 dims=[self.config.gender_transform_dim, self.config.dataset.get_label_num('gender')],
                                                 use_bias=True)
        self.edu_mlp = self._build_output_mlp(name='edu_mlp',
                                              activations=[Identity()],
                                              dims=[self.config.edu_transform_dim,  self.config.dataset.get_label_num('edu')],
                                              use_bias=True)
        self.age_transform = self._build_transform(name='age_transform',
                                                   input_dim=self.config.edu_transform_dim,
                                                   output_dim=self.config.age_transform_dim)
        self.gender_transform = self._build_transform(name='gender_transform',
                                                      input_dim=self.config.token_embed_dim,
                                                      output_dim=self.config.gender_transform_dim)
        self.edu_transform = self._build_transform(name='edu_transform',
                                                   input_dim=self.config.token_embed_dim,
                                                   output_dim=self.config.edu_transform_dim)

    def _build_transform(self, name, input_dim, output_dim, trans_times=1, activations=None):
        '''Transfer token embedding.

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
            std=numpy.sqrt(len(dims)) / numpy.sqrt(numpy.array(dims).sum()))
        mlp.biases_init = Constant(0)
        mlp.initialize()
        return mlp

    def _get_classifier_input(self):
        '''Get input of last mlp layer'''

        text_vec = self.token_embed.apply(self.query)
        self.token_embed_subset = text_vec
        encoded_queries = self._encode_query(text_vec, self.query_mask)
        return encoded_queries

    def _encode_query(self, text_vec, text_vec_mask):
        '''Encode vector representation of textual'''
        norm = text_vec_mask.sum(axis=1)[:, None]

        # Get representation of gender
        gender_transformed_text_vec = self.gender_transform.apply(text_vec) * text_vec_mask[:, :, None]
        gender_mean = gender_transformed_text_vec.sum(axis=1) / (norm + 1e-9)
        # gender_weight = tensor.abs_(text_vec - gender_mean[:, None, :])
        # gender_weight = self.gender_dist.apply(gender_weight)[:, :, 0]
        # gender_weight /= (gender_weight.sum(axis=1)[:, None] + 1e-9)
        # gender_vec = (text_vec * gender_weight[:, :, None]).sum(axis=1)
        gender_vec = gender_mean

        # Get representation of edu
        edu_transformed_text_vec = self.edu_transform.apply(text_vec) * text_vec_mask[:, :, None]

        edu_mean = edu_transformed_text_vec.sum(axis=1) / (norm + 1e-9)
        # edu_weight = tensor.abs_(edu_transformed_text_vec - edu_mean[:, None, :])
        # edu_weight = self.edu_dist.apply(edu_weight)[:, :, 0]
        # edu_weight /= edu_weight.sum(axis=1)[:, None] + 1e-9
        # edu_vec = (edu_transformed_text_vec * edu_weight[:, :, None]).sum(axis=1)
        edu_vec = edu_mean

        # Get representation of age
        age_transformed_text_vec = self.age_transform.apply(edu_transformed_text_vec) * text_vec_mask[:, :, None]
        age_mean = age_transformed_text_vec.sum(axis=1) / (norm + 1e-9)
        # age_weight = tensor.abs_(age_transformed_text_vec - age_mean[:, None, :])
        # age_weight = self.age_dist.apply(age_weight)[:, :, 0]
        # age_weight /= age_weight.sum(axis=1)[:,None] + 1e-9
        # age_vec = (age_transformed_text_vec * age_weight[:,:,None]).sum(axis=1)
        age_vec = age_mean

        return age_vec, gender_vec, edu_vec

    def _embed(self, sample_num, dim, name):
        embed = LookupTable(sample_num, dim, name=name)
        embed.weights_init = IsotropicGaussian(std=1 / numpy.sqrt(dim))
        embed.initialize()
        return embed

    def _get_pred_dist(self, input_vec):

        age_mlp_outputs = self.age_mlp.apply(input_vec[0])
        gender_mlp_outputs = self.gender_mlp.apply(input_vec[1])
        edu_mlp_outputs = self.edu_mlp.apply(input_vec[1])
        return age_mlp_outputs, gender_mlp_outputs, edu_mlp_outputs

    def _get_train_cost(self):
        self._get_class_weight()
        input_vec = self._get_classifier_input()
        age_mlp_outputs, gender_mlp_outputs, edu_mlp_outputs = self._get_pred_dist(input_vec)

        age_preds = age_mlp_outputs
        self.age_pred = tensor.argmax(age_preds, axis=1)
        self.age_pred.name = 'age'
        age_cost = self.categorical_cross_entropy(self.age_noised_label,
                                                  age_preds) * self.age_mask * self.age_weight[self.age]
        age_cost = age_cost.sum() / (self.age_mask.sum() + 1e-9)
        age_cost.name = 'age_cost'
        age_accuracy = (tensor.eq(self.age, self.age_pred) * self.age_mask).sum() / (self.age_mask.sum() + 1e-9)
        age_accuracy.name = 'age_accuracy'

        # Get cost on gender class
        gender_preds = gender_mlp_outputs
        self.gender_pred = tensor.argmax(gender_preds, axis=1)
        self.gender_pred.name = 'gender'
        gender_cost = self.categorical_cross_entropy(self.gender_noised_label,
                                                  gender_preds) * self.gender_mask * self.gender_weight[self.gender]
        gender_cost = gender_cost.sum() / (self.gender_mask.sum() + 1e-9)
        gender_cost.name = 'gender_cost'
        gender_accuracy = (tensor.eq(self.gender, self.gender_pred) * self.gender_mask).sum() / (self.gender_mask.sum() + 1e-9)
        gender_accuracy.name = 'gender_accuracy'

        # Get cost on edu class
        edu_preds = edu_mlp_outputs
        self.edu_pred = tensor.argmax(edu_preds, axis=1)
        self.edu_pred.name = 'edu'
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
        average_accuracy.name = 'accuracy'
        self._train_cg_generator = cost
        self._train_monitors = [cost, average_accuracy, age_accuracy, gender_accuracy, edu_accuracy]
        self._valid_monitors = [average_accuracy, age_accuracy, gender_accuracy, edu_accuracy]
        self._predict_monitors = [self.age_pred, self.gender_pred, self.edu_pred]

    def _get_class_weight(self):
        self.age_weight = self._get_weight('age')
        self.gender_weight = self._get_weight('gender')
        self.edu_weight = self._get_weight('edu')
        self.consider_constant.extend([self.age_weight, self.gender_weight, self.edu_weight])

    def _get_weight(self, name):
        freqs = []
        label2freq = self.config.dataset.get_label2freq(name)
        label_num = len(label2freq)
        try:
            for i in range(label_num):
                freqs.append(label2freq[i])
            freqs = numpy.array(freqs, dtype=theano.config.floatX)
            weights = (freqs.mean()/freqs)**0.
            return theano.shared(weights, name+'_weight')
        except:
            raise Exception('Label should be integer for training!')

    def _apply_noise(self):
        '''Apply dropout on computing graph of train, valid and test outputs (default not)'''
        cgs = ComputationGraph(self._train_cg_generator)
        mlps = [self.age_transform, self.gender_transform, self.edu_transform]
        mlp_params = []
        for mlp in mlps:
            mlp_params += [transformation.W for transformation in mlp.linear_transformations]
        cgs = apply_dropout(cgs, mlp_params, self.config.dropout_other)
        embed_params = self.token_embed.get_subsets()
        if embed_params is not None:
            cgs = apply_dropout(cgs, embed_params, self.config.dropout_embed)
        self._train_cg_generator, = cgs.outputs

    def _apply_reg(self, params=None):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and token embedding) to computing
        graph of self.cg_generator
        :param params: A list of parameters to which regularization applied
        '''
        mlps = [self.age_mlp, self.gender_mlp, self.edu_mlp, self.age_transform, self.gender_transform,
                self.edu_transform]
        mlp_params = []
        for mlp in mlps:
            mlp_params += [transformation.W for transformation in mlp.linear_transformations]
        self._train_cg_generator = self._train_cg_generator + self.config.l2_norm_embed * theano_expressions.l2_norm(
            tensors=[self.token_embed_subset]) ** 2 + \
                            self.config.l2_norm_other * theano_expressions.l2_norm(tensors=mlp_params) ** 2


class SogouSingleTaskModel(SogouMultiTaskModel):
    def __init__(self, *args, **kwargs):
        super(SogouSingleTaskModel, self).__init__(*args, **kwargs)
        self.task_name = self.config.task_name

    def _define_inputs(self):
        self.query = tensor.imatrix('query')
        self.query_mask = tensor.matrix('query_mask', dtype=theano.config.floatX)
        self.output = tensor.ivector(self.task_name)
        self.noised_output = tensor.matrix(self.task_name + '_noised_label', dtype=theano.config.floatX)

    def _build_bricks(self):
        # Build lookup tables
        self.token_embed = self._embed(len(self.config.dataset.token2index), self.config.token_embed_dim, name="token_embed")
        self.output_mlp = self._build_output_mlp(name=self.task_name + '_mlp',
                                              activations=[Identity()],
                                              dims=[self.config.token_embed_dim, self.config.dataset.label_num],
                                              use_bias=True)

    def _encode_query(self, text_vec, text_vec_mask):
        '''Encode vector representation of textual'''
        norm = text_vec_mask.sum(axis=1)[:, None]
        return text_vec.sum(axis=1) / (norm + 1e-9)

    def _get_pred_dist(self, input_vec):
        return self.output_mlp.apply(input_vec)

    def _get_train_cost(self):
        input_vec = self._get_classifier_input()
        mlp_outputs = self._get_pred_dist(input_vec)

        preds = mlp_outputs
        self.pred = tensor.argmax(preds, axis=1)
        self.pred.name = self.task_name
        cost = self.categorical_cross_entropy(self.noised_output, preds).mean()
        cost.name = self.task_name+'_cost'
        accuracy = tensor.eq(self.age, self.pred).mean()
        accuracy.name = self.task_name+'_accuracy'
        self._train_cg_generator = cost
        self._train_monitors = [accuracy, cost]
        self._valid_monitors = [accuracy]
        self._predict_monitors = [self.pred]

    def _get_class_weight(self):
        self.class_weight = self._get_weight(self.task_name)
        self.consider_constant.extend([self.class_weight])

    def _get_weight(self, name):
        freqs = []
        label2freq = self.config.dataset.labe2freq
        label_num = len(label2freq)
        try:
            for i in range(label_num):
                freqs.append(label2freq[i])
            freqs = numpy.array(freqs, dtype=theano.config.floatX)
            weights = (freqs.mean() / freqs) ** 0.
            return theano.shared(weights, name + '_weight')
        except:
            raise Exception('Label should be integer for training!')

    def _apply_noise(self):
        '''Apply dropout on computing graph of train, valid and test outputs (default not)'''
        cgs = ComputationGraph(self.train_cg_generator)
        cgs = apply_dropout(cgs, [self.output_mlp.W], self.config.dropout_other)
        embed_params = self.token_embed.get_subsets()
        if embed_params is not None:
            cgs = apply_dropout(cgs, embed_params, self.config.dropout_embed)
        self._train_cg_generator, = cgs.outputs

    def _apply_reg(self, params=None):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and token embedding) to computing
        graph of self.cg_generator
        :param params: A list of parameters to which regularization applied
        '''
        self._train_cg_generator = self._train_cg_generator + self.config.l2_norm_embed * theano_expressions.l2_norm(
            tensors=[self.token_embed_subset]) ** 2 + self.config.l2_norm_other * theano_expressions.l2_norm(
            tensors=[self.output_mlp.W])**2


class SogouLRModel(AbstractModel):
    def __init__(self, config, dataset, **kwargs):
        '''
        Define User-text-hashtag model
        :param config:
        :param dataset: User-text-hashtag dataset
        '''
        super(SogouLRModel, self).__init__(**kwargs)
        self.config = config
        self.config.dataset = dataset
        self.lr_model = None

    def build_model(self):
        self.lr_model = LRModel(
            label_num=self.config.dataset.label_num,
            input_dim=self.config.dataset.token_num,
            input_name='query',
            output_name=self.config.task_name,
            input_mask_name='query_mask',
            noised_output_name=self.config.task_name+'_noised_label',
            norm_type='l2_norm',
            norm_scale=self.config.l2_norm,
            label_weight=None)
        self._train_monitors = self.lr_model.train_monitors
        self._valid_monitors = self.lr_model.valid_monitors
        self._predict_monitors = self.lr_model.predict_monitors
        self._train_cg_generator = self.lr_model.train_cg_generator
        self._valid_cg_generator = self.lr_model.valid_cg_generator
        self._predict_cg_generator = self.lr_model.predict_cg_generator
        self._initialized = True


class SogouMultiTaskCharacterModel(SogouMultiTaskModel):

    def _define_inputs(self):
        self.query = tensor.tensor3('query', dtype=self.config.int_type)
        self.query_mask = tensor.tensor3('query_mask', dtype=theano.config.floatX)
        self.age = tensor.vector('age', self.config.int_type)
        self.age_mask = tensor.vector('age_mask', dtype=theano.config.floatX)
        self.gender = tensor.vector('gender', self.config.int_type)
        self.gender_mask = tensor.vector('gender_mask', dtype=theano.config.floatX)
        self.edu = tensor.vector('edu', dtype=self.config.int_type)
        self.edu_mask = tensor.vector('edu_mask', dtype=theano.config.floatX)
        self.age_noised_label = tensor.matrix('age_noised_label', dtype=theano.config.floatX)
        self.gender_noised_label = tensor.matrix('gender_noised_label', dtype=theano.config.floatX)
        self.edu_noised_label = tensor.matrix('edu_noised_label', dtype=theano.config.floatX)

    def _build_bricks(self):
        # Build lookup tables
        self.token_embed = self._embed(len(self.config.dataset.token2index), self.config.token_embed_dim,
                                       name="token_embed")
        self.age_mlp = self._build_output_mlp(name='age_mlp',
                                              activations=[
                                                  # Logistic(),
                                                  Identity()],
                                              dims=[
                                                  # self.config.topic_num,
                                                  self.config.age_topic_num,
                                                  self.config.dataset.get_label_num('age')],
                                              use_bias=True)
        self.gender_mlp = self._build_output_mlp(name='gender_mlp',
                                                 activations=[
                                                     # Logistic(),
                                                     Identity()],
                                                 dims=[
                                                     # self.config.topic_num,
                                                     self.config.gender_topic_num,
                                                     self.config.dataset.get_label_num('gender')],
                                                 use_bias=True)
        self.edu_mlp = self._build_output_mlp(name='edu_mlp',
                                              activations=[
                                                  # Logistic(),
                                                  Identity()],
                                              dims=[
                                                  # self.config.topic_num,
                                                  self.config.edu_topic_num,
                                                  self.config.dataset.get_label_num('edu')],
                                              use_bias=True)
        # LSTM for encoding queries
        self.embed_query_linear = Linear(input_dim=self.config.token_embed_dim,
                                         output_dim=4*self.config.query_embed_dim,
                                         name='embed_query_linear',
                                         use_bias=False)
        self.embed_query_linear.weights_init = IsotropicGaussian(std=1./np.sqrt(self.config.query_embed_dim))
        self.embed_query_linear.initialize()
        self.forward_lstm = self._build_lstm(dim=self.config.query_embed_dim, name='forward_lstm')
        self.backward_lstm = self._build_lstm(dim=self.config.query_embed_dim, name='backward_lstm')
        # Map encoded queries into topic distributions (should there different number of topics for different tasks?)
        # self.topic_embed = self._build_topic_embed(name='topic_embed',
        #                                            input_dim=2*self.config.query_embed_dim,
        #                                            topic_num=self.config.topic_num
        #                                            )
        self.age_topic_embed = self._build_topic_embed(name='age_topic_embed',
                                                       input_dim=2*self.config.query_embed_dim,
                                                       topic_num=self.config.age_topic_num)
        self.gender_topic_embed = self._build_topic_embed(name='gender_topic_embed',
                                                          input_dim=2*self.config.query_embed_dim,
                                                          topic_num=self.config.gender_topic_num)
        self.edu_topic_embed = self._build_topic_embed(name='edu_topic_embed',
                                                          input_dim=2*self.config.query_embed_dim,
                                                          topic_num=self.config.edu_topic_num)

    def _build_lstm(self, name, dim, activation=None):
        lstm = LSTM(dim=dim, activation=activation, name=name)
        lstm.weights_init = IsotropicGaussian(1/np.sqrt(dim))
        lstm.biases_init = Constant(0)
        lstm.initialize()
        return lstm

    def _build_topic_embed(self, name, input_dim, topic_num):
        topic_embed = theano.shared(
            (np.random.randn(input_dim, topic_num)*np.sqrt(6)/np.sqrt(input_dim+topic_num)).astype(theano.config.floatX),
            name=name)
        add_role(topic_embed, WEIGHT)
        return topic_embed

    def _get_classifier_input(self):
        '''Get input of last mlp layer'''
        # For each user, get a matrix: query_num * (2 * query_embed_dim)
        encoded_queries = self._encode_query(self.query, self.query_mask)

        return self._get_topic_dist(encoded_queries)

    def _get_topic_dist(self, encoded_queries):
        '''Get topic distribution of batch of users

        :param encoded_queries: Tensor.Variable
                Dim: batch_size * query_num_a_user * query_embedding_dim
        :return:
        '''

        # topics = tensor.exp(tensor.dot(encoded_queries, self.topic_embed))
        # topics /= topics.sum(axis=2)[:, :, None]  # Softmax: batch_size * query_num * query_embedding_dim
        # topics *= self.query_mask[:, :, 0][:, :, None]  # Remove empty query
        # topics = topics.sum(axis=1)  # batch_size * query_embedding_dim
        # topics /= topics.sum(axis=1)[:, None]  # Normalization

        # Age topic distribution
        age_topics = tensor.exp(tensor.dot(encoded_queries, self.age_topic_embed))
        age_topics /= age_topics.sum(axis=2)[:, :, None]            # Softmax: batch_size * query_num * query_embedding_dim
        age_topics *= self.query_mask[:, :, 0][:, :, None]          # Remove empty query
        age_topics = age_topics.sum(axis=1)                         # batch_size * query_embedding_dim
        age_topics /= age_topics.sum(axis=1)[:, None]               # Normalization

        # Gender topic distribution
        gender_topics = tensor.exp(tensor.dot(encoded_queries, self.gender_topic_embed))
        gender_topics /= gender_topics.sum(axis=2)[:, :, None]  # Softmax: batch_size * query_num * query_embedding_dim
        gender_topics *= self.query_mask[:, :, 0][:, :, None]  # Remove empty query
        gender_topics = gender_topics.sum(axis=1)  # batch_size * query_embedding_dim
        gender_topics /= gender_topics.sum(axis=1)[:, None]  # Normalization

        # Edu topic distribution
        edu_topics = tensor.exp(tensor.dot(encoded_queries, self.edu_topic_embed))
        edu_topics /= edu_topics.sum(axis=2)[:, :, None]  # Softmax: batch_size * query_num * query_embedding_dim
        edu_topics *= self.query_mask[:, :, 0][:, :, None]  # Remove empty query
        edu_topics = edu_topics.sum(axis=1)  # batch_size * query_embedding_dim
        edu_topics /= edu_topics.sum(axis=1)[:, None]  # Normalization
        return age_topics, gender_topics, edu_topics
        # return topics, topics, topics

    def _encode_query(self, text_vec, text_vec_mask):
        '''Encode vector representation of textual'''
        shape = text_vec.shape
        mask = text_vec_mask.reshape((-1, shape[2])).dimshuffle(1, 0)
        queries = self.query.reshape((-1, shape[2])).dimshuffle(1, 0)
        query_vec = self.embed_query_linear.apply(self.token_embed.apply(queries))
        forward, _ = self.forward_lstm.apply(inputs=query_vec, mask=mask)
        backward, _ = self.backward_lstm.apply(inputs=query_vec[::-1], mask=mask[::-1])
        forward = forward[-1]
        backward = backward[-1]
        encoded_queries = tensor.concatenate([forward, backward], axis=1).reshape((shape[0], shape[1], -1))
        # return forward
        return encoded_queries

    def _apply_noise(self):
        '''Apply dropout on computing graph of train, valid and test outputs (default not)'''
        cgs = ComputationGraph(self._train_cg_generator)
        self.token_embed.remove_subsets()
        embed_params = [self.token_embed.W] + [self.age_topic_embed, self.gender_topic_embed,
                        self.edu_topic_embed]
        # embed_params = [self.token_embed.W] + [self.topic_embed]
        if embed_params is not None:
            cgs = apply_noise(cgs, embed_params, self.config.noise_embed)
        self._train_cg_generator, = cgs.outputs

    def _apply_reg(self, params=None):
        '''
        Apply regularization (default L2 norm) on parameters (default user, hashtag and token embedding) to computing
        graph of self.cg_generator
        :param params: A list of parameters to which regularization applied
        '''
        mlps = [self.age_mlp, self.gender_mlp, self.edu_mlp]
        mlp_params = []
        for mlp in mlps:
            mlp_params += [transformation.W for transformation in mlp.linear_transformations]
        self.token_embed.remove_subsets()
        embed_params = [self.token_embed.W] + \
                       [self.age_topic_embed, self.gender_topic_embed, self.edu_topic_embed, self.embed_query_linear.W]
        # embed_params = [self.token_embed.W] + \
        #                [self.topic_embed, self.embed_query_linear.W]
        self._train_cg_generator = self._train_cg_generator \
                                   + self.config.l2_norm_embed * theano_expressions.l2_norm(tensors=embed_params) ** 2\
                                   + self.config.l2_norm_other * theano_expressions.l2_norm(tensors=mlp_params) ** 2
