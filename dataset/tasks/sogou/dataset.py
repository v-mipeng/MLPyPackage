import numpy as np
import theano

from pml.dataset.base import AbstractDocClassificationDataset, DatasetContainer
from pml.dataset.tasks.sogou import DataStream, Batch, Mapping, SortMapping, Unpack
from pml.dataset.tasks.sogou import QuerySample, QueryMerge, OutputNoise, MatrixPadding
from pml.dataset.tasks.sogou import ShuffledExampleScheme, SequentialExampleScheme, ConstantScheme
from pml.dataset.tasks.sogou import _balanced_batch_helper



class BaseSogouDataset(AbstractDocClassificationDataset):
    def __init__(self, config, **kwargs):
        super(BaseSogouDataset, self).__init__(**kwargs)
        self.config = config
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def token_num(self):
        return len(self.token2index)

    @property
    def token2index(self):
        if hasattr(self, '_token2index'):
            return self._token2index.copy()
        else:
            self._token2index ={}
            return self._token2index

    def initialize(self, param_load_from=None):
        if param_load_from is None:
            if hasattr(self.config, 'dataset_param_load_from'):
                param_load_from = self.config.dataset_param_load_from
        super(BaseSogouDataset, self).initialize(param_load_from)

    def reset(self, preserved_attributes=None):
        preserved_attrs = {'config', 'task_name', '_true_label2pred_label', '_pred_label2true_label',
                           'provide_train_sources', 'provide_test_sources', 'need_mask_sources',
                           'compare_source'}
        super(BaseSogouDataset, self).reset(preserved_attrs, *args, **kwargs)

    def save(self, param_save_to=None):
        if param_save_to is None:
            if hasattr(self.config, 'dataset_param_save_to'):
                param_save_to = self.config.dataset_param_save_to
        super(BaseSogouDataset, self).save(param_save_to)

    def _map_labels(self, labels, label2index):
        ts = [(idx, label2index[label]) for idx, label in enumerate(labels)
              if label in label2index]
        idxes, used_labels = zip(*ts)
        idxes = np.array(idxes)
        used_labels = np.array(used_labels)
        return idxes, used_labels

    def _construct_label_mapping(self, labels):
        unique_labels = np.unique(labels)
        return dict(zip(unique_labels, range(len(unique_labels))))

    def _get_token_index(self, token, label2index):
        return label2index.setdefault(token, len(self.token2index))

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
        #
        # if for_type == 'train':
        #     stream = TokenSample(stream, sample_source='query_mask',
        #                          sample_prob=self.config.token_sample_prob)
        return stream


class SingleTaskSogouDataset(BaseSogouDataset):
    '''Dataset for single-task training

    The information this dataset provides includes:
        1. user id: str
        2. labels of given task: int (start from 0). The samples whose true label are 0 are removed.
        3. user queries: int. Queries are grouped by user
    Transformation applied:
        1. Batch
        2. Padding queries
        3. Sample queries for training dataset by proportion
        4. Bag queries
        5. Add output noise
    '''
    def __init__(self, task_name, true_label2pred_label=None, *args, **kwargs):
        super(SingleTaskSogouDataset, self).__init__(*args, **kwargs)
        self.task_name = task_name
        self.true_label2pred_label = true_label2pred_label
        self.provide_train_sources = ('id', task_name, 'query')
        self.provide_test_sources = ('id', 'query',)
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        return len(self.true_label2pred_label)

    def _process_before_mapping_for_train(self, raw_dataset):
        if self.true_label2pred_label is None:
            l = len(np.unique(raw_dataset[self.task_name]))
            self.true_label2pred_label = dict(zip(range(1, l), range(0, l - 1)))
        return raw_dataset

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_labels(raw_dataset[self.task_name], self.true_label2pred_label)
        queries_per_user = np.array(
            [[np.array(list(set([self._get_token_index(token, self.token2index) for token in query])),
                          dtype=self.config.int_type)   # Construct one hot representation
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        return DatasetContainer(dict(zip(('id',self.task_name, 'query'), (ids, labels, queries_per_user))))

    def _map_for_valid(self, raw_dataset):
        dataset = self._map_for_test(raw_dataset)
        idxes, labels = self._map_labels(raw_dataset[self.task_name], self.true_label2pred_label)
        dataset['id'] = dataset['id'][idxes]
        dataset['query'] = dataset['query'][idxes]
        dataset[self.task_name] = labels
        return dataset

    def _map_for_test(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index[token] for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        return DatasetContainer(dict(zip(('id', 'query'), (ids, queries_per_user))))

    def _process_after_mapping_for_train(self, dataset):
        super(SingleTaskSogouDataset, self)._process_after_mapping_for_train(dataset)
        labels = dataset[0]
        unique_labels, counts = np.unique(labels, return_counts=True)
        self.label2freq = dict(zip(unique_labels, counts))
        return dataset

    def _add_transform(self, stream, for_type):
        # Add mask
        stream = super(SingleTaskSogouDataset, self)._add_transform(stream, for_type)
        if for_type == 'train':
            stream = OutputNoise(stream, output_source=self.task_name,
                                 label2freq=self.label2freq,
                                 max_noise_prob=self.config.output_noise_prob,
                                 decay_rate=self.config.decay_rate)
        return stream


class MultiTaskSogouDataset(BaseSogouDataset):
    '''Dataset for multiple-task training

    The information this dataset provides includes:
        1. user id: str
        2. labels of given tasks: int (start from 0)
        3. label masks: {0.,1.}. 0. indicate the true label is 0 which actually means unknown.
        4. user queries: int. Queries are grouped by user
    Transformation applied:
        1. Batch
        2. Padding queries
        3. Sample queries for training dataset by proportion
        4. Bag queries
        5. Add output noise

    '''
    def __init__(self, task_names, true_label2pred_labels=None, *args, **kwargs):
        '''
        :param config:
        :param task_names: list or tuple
                List of task names, e.g., age, gender and edu
        :param true_label2pred_labels: dict of dicts
        :return:
        '''
        super(BaseSogouDataset, self).__init__(**kwargs)
        self.task_names = task_names
        self.true_label2pred_labels = true_label2pred_labels
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    def get_label_num(self, task_name):
        return self.true_label2pred_labels[task_name]

    def get_label_freq(self, task_name):
        return self.label2freqs[task_name]

    def get_true_label2pred_label(self, task_name):
        return self.true_label2pred_labels[task_name]

    def get_pred_label2true_label(self, task_name):
        return {v:k for k, v in self.get_true_label2pred_label(task_name).iteritems()}

    def _process_before_mapping_for_train(self, raw_dataset):
        for task_name in self.task_names:
            if self.true_label2pred_labels.get(task_name, None) is None:
                l = len(np.unique(raw_dataset[task_name]))
                self.true_label2pred_labels[task_name] = dict(zip(range(1,l), range(0,l-1)))
        return raw_dataset

    def _process_after_mapping_for_train(self, dataset):
        self.label2freqs = dict()
        for task_name in self.task_names:
            label_idx = self.provide_train_sources.index(task_name)
            mask_idx = self.provide_train_sources.index(task_name+'_mask')
            labels = dataset[label_idx][dataset[mask_idx]>0.]
            unique_labels, counts = np.unique(labels, return_counts=True)
            self.label2freqs[task_name] = dict(zip(unique_labels, counts))
        return dataset

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [[np.array(list(set([self._get_token_index(token, self.token2index) for token in query])),
                       dtype=self.config.int_type)  # Construct one hot representation
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        task_labels = []
        label_masks = []
        for task_name in self.task_names:
            labels, mask = self._map_labels(raw_dataset[task_name], self.true_label2pred_labels[task_name])
            task_labels.append(labels)
            label_masks.append(mask)
        return (ids, ) + tuple(task_labels) + tuple(label_masks) + (queries_per_user,)

    def _map_labels(self, labels, label2index):
        mask = []
        mapped_labels = []
        for label in labels:
            mapped_label = label2index.get(label, -1)
            if mapped_label == -1:
                mapped_labels.append(0)
                mask.append(0.)
            else:
                mapped_labels.append(mapped_label)
                mask.append(1.)
        mask = np.array(mask, dtype=theano.config.floatX)
        mapped_labels = np.array(mapped_labels, dtype=self.config.int_type)
        return mapped_labels, mask

    def _map_for_valid(self, raw_dataset):
        ids, queries_per_user = self._map_for_test(raw_dataset)
        task_labels = []
        label_masks = []
        for task_name in self.task_names:
            labels, mask = self._map_labels(raw_dataset[task_name], self.true_label2pred_labels[task_name])
            task_labels.append(labels)
            label_masks.append(mask)
        return (ids,) + tuple(task_labels) + tuple(label_masks) + (queries_per_user,)

    def _map_for_test(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [[np.array(list(set([self.token2index[token] for token in query if token in self.token2index])),
                       dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset['query']], dtype='O')
        return (ids, queries_per_user)

    def _add_transform(self, stream, for_type):
        stream = super(MultiTaskSogouDataset, self)._add_transform(stream, for_type)
        if for_type == 'train':
            for task_name in self.task_names:
                stream = OutputNoise(stream, output_source=task_name,
                                     label2freq=self.label2freqs[task_name],
                                     max_noise_prob=self.config.output_noise_probs[task_name],
                                     decay_rate=self.config.decay_rates[task_name])
        return stream