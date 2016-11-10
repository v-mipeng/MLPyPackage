import os
import cPickle
from warnings import warn
from collections import OrderedDict
from abc import abstractproperty, abstractmethod

import theano
import numpy as np

from pml.dataset.tasks.sogou import SequentialScheme, ShuffledExampleScheme, SequentialExampleScheme, ConstantScheme
from pml.dataset.tasks.sogou import DataStream, Batch, Mapping, MatrixPadding, SortMapping, Unpack
from pml.dataset.tasks.sogou import QuerySample, QueryMerge, TokenSample, OutputNoise, MatrixPadding
from pml.dataset.tasks.sogou import _balanced_batch_helper
from pml.dataset.base import AbstractDocClassificationDataset


class SingleTaskDataset(AbstractDocClassificationDataset):
    def __init__(self, config, task_name, true_label2pred_label = None, *args, **kwargs):
        super(SingleTaskDataset, self).__init__(*args, **kwargs)
        self.config = config
        self.task_name = task_name
        self.true_label2pred_label = true_label2pred_label
        self.provide_train_sources = ('id', task_name, 'query')
        self.provide_test_sources = ('id', 'query',)
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        return len(self.true_label2pred_label)

    @property
    def token_num(self):
        return len(self.token2index)

    @property
    def token2index(self):
        if hasattr(self, '_token2index'):
            return self._token2index.copy()
        else:
            return None

    def initialize(self, param_load_from=None, *args, **kwargs):
        if param_load_from is None:
            if hasattr(self.config, 'dataset_param_load_from'):
                param_load_from = self.config.dataset_param_load_from
        super(SingleTaskDataset, self).initialize(param_load_from)

    def reset(self,preserved_attributes = None, *args, **kwargs):
        preserved_attrs = {'config','task_name','_true_label2pred_label','_pred_label2true_label',
                          'provide_train_sources','provide_test_sources', 'need_mask_sources',
                          'compare_source'}
        super(SingleTaskDataset, self).reset(preserved_attrs)

    def save(self, param_save_to=None):
        if param_save_to is None:
            if hasattr(self.config, 'dataset_param_save_to'):
                param_save_to = self.config.dataset_param_save_to
        super(SingleTaskDataset, self).save(param_save_to)

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_label(raw_dataset)
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self._get_token_index(token) for token in query])),
                          dtype=self.config.int_type)   # Construct one hot representation
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        self._train_sample_num = len(ids)
        return [ids, labels, queries_per_user]

    def _map_for_valid(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_label(raw_dataset)
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index(token) for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        self._valid_sample_num = len(ids)
        return [ids, labels, queries_per_user]

    def _map_for_test(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index(token) for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        self._test_sample_num = len(ids)
        return [ids, queries_per_user]

    def _map_label(self, raw_dataset):
        if self.true_label2pred_label is None:
            labels = np.array([self._get_label(label) for label in raw_dataset[self.task_name]])
            idxes = np.arange(len(labels))
        else:
            idxes, labels = zip(*[(i, self.true_label2pred_label[label])
                                  for i, label in enumerate(raw_dataset[self.task_name])
                                  if label in self.true_label2pred_label])
            idxes = np.array(idxes)
            labels = np.array(labels)
        return idxes, labels

    def _get_label(self, label):
        if self.true_label2pred_label is None:
            self.true_label2pred_label = {}
        return self.true_label2pred_label.setdefault(label, len(self.true_label2pred_label))

    def _get_token_index(self, token):
        if self.true_label2pred_label is None:
            self.true_label2pred_label = {}
        return self.token2index.setdefault(token, len(self.token2index))

    def _process_after_mapping_for_train(self, dataset):
        labels = dataset[self.task_name]
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.label2freq = dict(zip(unique_labels, counts))
        return dataset

    def _construct_shuffled_stream(self, dataset, for_type='train'):
        '''Construc a shuffled stream from an IndexableDataset object

        Subclass should add transformation on the stream, e.g.,
                1.Sort samples by size
                2.Batch dataset
                3.Add mask on samples
        :param dataset: fuel.IndexableDataset
                This is constructed by self._construct_dataset method.
        :return: fuel.stream.Datastream
                An object of fuel.stream.Datastream with ShuffledExampleScheme
                A fuel shuffled stream with basic transformations,
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        return self._add_transform(stream, for_type)

    def _construct_sequential_stream(self, dataset, for_type='train'):
        '''Construc a sequencial stream from an IndexableDataset object

        Subclass should add transformation on the stream, e.g.,
                1.Sort samples by size
                2.Batch dataset
                3.Add mask on samples
        :param dataset: fuel.IndexableDataset
                This is constructed by self._construct_dataset method.
        :return: fuel.stream.Datastream
                An object of fuel.stream.Datastream with SequentialExampleScheme
                A fuel sequential stream with basic transformations,
        '''
        it = SequentialExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        return self._add_transform(stream, for_type)

    def _add_transform(self, stream, for_type):
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = MatrixPadding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        if for_type == 'train':
            stream = QuerySample(stream,
                                 sample_source='query',
                                 sample_prob=self.config.query_sample_prob,
                                 for_type=for_type)
        stream = QueryMerge(stream,
                            merge_source='query')

        if for_type == 'train':
            stream = TokenSample(stream, sample_source='query_mask',
                                 sample_prob=self.config.token_sample_prob)
            stream = OutputNoise(stream, output_source=self.task_name,
                                 label2freq=self.label2freq,
                                 max_noise_prob=self.config.output_noise_prob,
                                 decay_rate=self.config.decay_rate)
        return stream


class MultiTaskDataset(SingleTaskDataset):
    def __init__(self, config, task_names, true_label2pred_labels=None, *args, **kwargs):
        '''

        :param config:
        :param task_names: list or tuple
                List of task names, e.g., age, gender and edu
        :param true_label2pred_labels: list of dict
                
        :param args:
        :param kwargs:
        :return:
        '''
        super(SingleTaskDataset, self).__init__(*args, **kwargs)
        self.config = config
        self.task_names = task_names
        self.true_label2pred_labels = true_label2pred_labels
        self.provide_train_sources = ('id', )+tuple(task_names) +('query',)
        self.provide_test_sources = ('id', 'query',)
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        return len(self.true_label2pred_label)

    @property
    def token_num(self):
        return len(self.token2index)

    @property
    def token2index(self):
        if hasattr(self, '_token2index'):
            return self._token2index.copy()
        else:
            return None

    def initialize(self, param_load_from=None, *args, **kwargs):
        if param_load_from is None:
            if hasattr(self.config, 'dataset_param_load_from'):
                param_load_from = self.config.dataset_param_load_from
        super(SingleTaskDataset, self).initialize(param_load_from)

    def reset(self, preserved_attributes=None, *args, **kwargs):
        preserved_attrs = {'config', 'task_name', '_true_label2pred_label', '_pred_label2true_label',
                           'provide_train_sources', 'provide_test_sources', 'need_mask_sources',
                           'compare_source'}
        super(SingleTaskDataset, self).reset(preserved_attrs)

    def save(self, param_save_to=None):
        if param_save_to is None:
            if hasattr(self.config, 'dataset_param_save_to'):
                param_save_to = self.config.dataset_param_save_to
        super(SingleTaskDataset, self).save(param_save_to)

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_label(raw_dataset)
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self._get_token_index(token) for token in query])),
                       dtype=self.config.int_type)  # Construct one hot representation
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        self._train_sample_num = len(ids)
        return [ids, labels, queries_per_user]

    def _map_for_valid(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_label(raw_dataset)
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index(token) for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        self._valid_sample_num = len(ids)
        return [ids, labels, queries_per_user]

    def _map_for_test(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        # You should filter sparse word in the pre-process step
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index(token) for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        self._test_sample_num = len(ids)
        return [ids, queries_per_user]

    def _map_label(self, raw_dataset):
        if self.true_label2pred_label is None:
            labels = np.array([self._get_label(label) for label in raw_dataset[self.task_name]])
            idxes = np.arange(len(labels))
        else:
            idxes, labels = zip(*[(i, self.true_label2pred_label[label])
                                  for i, label in enumerate(raw_dataset[self.task_name])
                                  if label in self.true_label2pred_label])
            idxes = np.array(idxes)
            labels = np.array(labels)
        return idxes, labels

    def _get_label(self, label):
        if self.true_label2pred_label is None:
            self.true_label2pred_label = {}
        return self.true_label2pred_label.setdefault(label, len(self.true_label2pred_label))

    def _get_token_index(self, token):
        if self.true_label2pred_label is None:
            self.true_label2pred_label = {}
        return self.token2index.setdefault(token, len(self.token2index))

    def _construct_shuffled_stream(self, dataset, for_type='train'):
        '''Construc a shuffled stream from an IndexableDataset object

        Subclass should add transformation on the stream, e.g.,
                1.Sort samples by size
                2.Batch dataset
                3.Add mask on samples
        :param dataset: fuel.IndexableDataset
                This is constructed by self._construct_dataset method.
        :return: fuel.stream.Datastream
                An object of fuel.stream.Datastream with ShuffledExampleScheme
                A fuel shuffled stream with basic transformations,
        '''
        it = ShuffledExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # Sort sets of multiple batches to make batches of similar sizes
        stream = Batch(stream,
                       iteration_scheme=ConstantScheme(self.config.batch_size * self.config.sort_batch_count))
        comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        stream = Mapping(stream, SortMapping(comparison))
        stream = Unpack(stream)
        stream = Batch(stream, iteration_scheme=ConstantScheme(self.config.batch_size))
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = MatrixPadding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        stream = BaggedQuerySample(stream,
                                   sample_source='query',
                                   sample_prob=self.config.query_sample_prob,
                                   for_type=for_type)
        if for_type == 'train':
            stream = FeatureSample(stream, 'query_mask',
                                   self.config.token_sample_prob)
            stream = OutputNoise(stream, output_source='age',
                                 label2freq=self.age2freq,
                                 max_noise_prob=self.config.age_max_noise,
                                 decay_rate=self.config.age_decay_rate)
            stream = OutputNoise(stream, output_source='gender',
                                 label2freq=self.gender2freq,
                                 max_noise_prob=self.config.gender_max_noise,
                                 decay_rate=self.config.gender_decay_rate)
            stream = OutputNoise(stream, output_source='edu',
                                 label2freq=self.edu2freq,
                                 max_noise_prob=self.config.edu_max_noise,
                                 decay_rate=self.config.edu_decay_rate)
        return stream

    def _construct_sequential_stream(self, dataset, for_type='train'):
        '''Construc a sequencial stream from an IndexableDataset object

        Subclass should add transformation on the stream, e.g.,
                1.Sort samples by size
                2.Batch dataset
                3.Add mask on samples
        :param dataset: fuel.IndexableDataset
                This is constructed by self._construct_dataset method.
        :return: fuel.stream.Datastream
                An object of fuel.stream.Datastream with SequentialExampleScheme
                A fuel sequential stream with basic transformations,
        '''
        it = SequentialExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # # Batch examples
        # stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        # Add mask on inputs
        # for source in self.need_mask_sources.iteritems():
        #     stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream



