from collections import OrderedDict
import numpy as np
import theano

from pml.dataset.base import AbstractDocClassificationDataset, DatasetContainer


class SogouBaseDataset(AbstractDocClassificationDataset):
    def __init__(self, config, **kwargs):
        super(SogouBaseDataset, self).__init__(**kwargs)
        self.config = config
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'
        self._token2index = {}

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
        super(SogouBaseDataset, self).initialize(param_load_from)

    def reset(self, preserved_attributes=None):
        preserved_attrs = {'config', 'task_name', '_true_label2pred_label', '_pred_label2true_label',
                           'need_mask_sources',
                           'compare_source'}
        super(SogouBaseDataset, self).reset(preserved_attrs)

    def save(self, param_save_to=None):
        if param_save_to is None:
            if hasattr(self.config, 'dataset_param_save_to'):
                param_save_to = self.config.dataset_param_save_to
        super(SogouBaseDataset, self).save(param_save_to)

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

    def _get_token_index(self, token, token2index):
        return token2index.setdefault(token, len(token2index))


class SogouSingleTaskDataset(SogouBaseDataset):
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
    def __init__(self, *args, **kwargs):
        super(SogouSingleTaskDataset, self).__init__(*args, **kwargs)
        self.task_name = self.config.task_name
        if hasattr(self.config, 'true_label2pred_label'):
            self.true_label2pred_label = self.config.true_label2pred_label
        else:
            self.true_label2pred_label = None
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        return len(self.true_label2pred_label)

    @property
    def label2freq(self):
        return self._label2freq

    def _process_before_mapping_for_train(self, raw_dataset):
        if self.true_label2pred_label is None:
            l = len(np.unique(raw_dataset[self.task_name]))
            self.true_label2pred_label = dict(zip(range(1, l), range(0, l - 1)))
        return raw_dataset

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        idxes, labels = self._map_labels(raw_dataset[self.task_name], self.true_label2pred_label)
        queries_per_user = np.array(
            [np.array([np.array(list(set([self._get_token_index(token, self._token2index) for token in query])),
                          dtype=self.config.int_type)   # Construct one hot representation
              for query in queries]) for queries in raw_dataset['query']], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]
        return DatasetContainer(dict(zip(('id', self.task_name, 'query'), (ids, labels, queries_per_user))))

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
            [np.array([np.array(list(set([self._token2index[token] for token in query if token in self._token2index])),
                       dtype=self.config.int_type)
              for query in queries]) for queries in raw_dataset['query']], dtype='O')
        return DatasetContainer(dict(zip(('id', 'query'), (ids, queries_per_user))))

    def _process_after_mapping_for_train(self, dataset):
        # Get instance frequency of each label
        super(SingleTaskSogouBaseDataset, self)._process_after_mapping_for_train(dataset)
        labels = dataset[0]
        unique_labels, counts = np.unique(labels, return_counts=True)
        self._label2freq = dict(zip(unique_labels, counts))
        return dataset


class SogouMultiTaskDataset(SogouBaseDataset):
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
    def __init__(self, *args, **kwargs):
        '''
        :param config:
        :param task_names: list or tuple
                List of task names, e.g., age, gender and edu
        :param true_label2pred_labels: dict of dicts
        :return:
        '''
        super(SogouMultiTaskDataset, self).__init__(*args, **kwargs)
        if hasattr(self.config, 'task_names') and self.config.task_names is not None:
            self.task_names = self.config.task_names
        else:
            self.task_names = ['age', 'gender', 'edu']
        if hasattr(self.config, 'true_label2pred_labels'):
            self.true_label2pred_labels = self.config.true_label2pred_labels
        else:
            self.true_label2pred_labels = None
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    def get_label_num(self, task_name):
        return len(self.true_label2pred_labels[task_name])

    def get_label2freq(self, task_name):
        return self.label2freqs[task_name]

    def get_true_label2pred_label(self, task_name):
        return self.true_label2pred_labels[task_name]

    def get_pred_label2true_label(self, task_name):
        return {v:k for k, v in self.get_true_label2pred_label(task_name).iteritems()}

    def _process_before_mapping_for_train(self, raw_dataset):
        if self.true_label2pred_labels is None:
            self.true_label2pred_labels = {}
            for task_name in self.task_names:
                if self.true_label2pred_labels.get(task_name, None) is None:
                    l = len(np.unique(raw_dataset[task_name]))
                    self.true_label2pred_labels[task_name] = dict(zip(range(1, l), range(0, l-1)))
        return raw_dataset

    def _process_after_mapping_for_train(self, dataset):
        self.label2freqs = dict()
        for task_name in self.task_names:
            labels = dataset[task_name][dataset[task_name+'_mask'] > 0.]
            unique_labels, counts = np.unique(labels, return_counts=True)
            self.label2freqs[task_name] = dict(zip(unique_labels, counts))
        return dataset

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [np.array([np.array(list(set([self._get_token_index(token, self._token2index) for token in query])),
                       dtype=self.config.int_type)  # Construct one hot representation
              for query in queries]) for queries in raw_dataset['query']], dtype='O')
        task_labels = []
        label_masks = []
        for task_name in self.task_names:
            labels, mask = self._map_labels(raw_dataset[task_name], self.true_label2pred_labels[task_name])
            task_labels.append(labels)
            label_masks.append(mask)
        names = ['id'] + self.task_names + [task_name+'_mask' for task_name in self.task_names] + ['query']
        values = [ids] + task_labels + label_masks + [queries_per_user]
        return DatasetContainer(OrderedDict(zip(names, values)))

    def _map_labels(self, labels, label2index):
        mask = np.ones_like(labels, dtype=theano.config.floatX)
        mapped_labels = np.array(labels, dtype=self.config.int_type)
        for idx, label in enumerate(labels):
            mapped_label = label2index.get(label, -1)
            if mapped_label == -1:
                mapped_label = 0
                mask[idx] = 0.
            mapped_labels[idx] = mapped_label
        return mapped_labels, mask

    def _map_for_valid(self, raw_dataset):
        valid_dataset = self._map_for_predict(raw_dataset)
        for task_name in self.task_names:
            labels, mask = self._map_labels(raw_dataset[task_name], self.true_label2pred_labels[task_name])
            valid_dataset[task_name] = labels
            valid_dataset[task_name+'_mask'] = mask
        return valid_dataset

    def _map_for_predict(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [np.array([np.array(list(set([self._token2index[token] for token in query if token in self._token2index])),
                       dtype=self.config.int_type)
              for query in queries]) for queries in raw_dataset['query']], dtype='O')
        return DatasetContainer(OrderedDict(zip(['id', 'query'], [ids, queries_per_user])))


class SogouMultiTaskCharacterDataset(SogouMultiTaskDataset):

    def _map_for_train(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [np.array([np.array(
                [self._get_token_index(token, self._token2index) for token in query.split(' ')],
                dtype=self.config.int_type)
              for query in queries]) for queries in raw_dataset['query']], dtype='O')
        task_labels = []
        label_masks = []
        for task_name in self.task_names:
            labels, mask = self._map_labels(raw_dataset[task_name], self.true_label2pred_labels[task_name])
            task_labels.append(labels)
            label_masks.append(mask)
        names = ['id'] + self.task_names + [task_name+'_mask' for task_name in self.task_names] + ['query']
        values = [ids] + task_labels + label_masks + [queries_per_user]
        return DatasetContainer(OrderedDict(zip(names, values)))

    def _map_for_predict(self, raw_dataset):
        ids = np.array(raw_dataset['id'])
        queries_per_user = np.array(
            [np.array([np.array(
                [self._token2index[token] for token in query.split(' ') if token in self._token2index],
                dtype=self.config.int_type)
                       for query in queries]) for queries in raw_dataset['query']], dtype='O')
        return DatasetContainer(OrderedDict(zip(['id', 'query'], [ids, queries_per_user])))


def test_dataset():
    import os
    from pml.tasks.sogou.config import SogouMultiTaskConfig
    from pml.tasks.sogou.preprocess import SogouTrainRawDatasetReaderWriter
    cur_path = os.path.abspath(__file__)
    project_dir = cur_path[0:cur_path.index('source')]
    config = SogouMultiTaskConfig()
    dataset = SogouMultiTaskCharacterDataset(config)

    # Process training dataset
    read_from = os.path.join(project_dir, 'data/debug/train.txt')           # original data path
    train_reader_witer = SogouTrainRawDatasetReaderWriter(read_from=read_from, save_to=None)
    raw_train_dataset = train_reader_witer.read_dataset()
    dataset.get_train_stream(raw_train_dataset)


if __name__ == '__main__':
    test_dataset()