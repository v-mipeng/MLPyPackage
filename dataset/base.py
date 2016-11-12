import cPickle
import itertools
import os
from abc import abstractmethod, abstractproperty
from collections import OrderedDict

import numpy as np
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledExampleScheme, SequentialExampleScheme
from fuel.streams import DataStream


class RawDataset(object):
    '''Raw dataset container

    '''
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
        self.check_dim()

    def __getitem__(self, source):
        return self.raw_dataset[source]

    def __setitem__(self, key, value):
        self.raw_dataset[key] = value

    def __len__(self):
        return self.sample_num

    def __iter__(self):
        '''Iterate over raw dataset by sample over fields defined in self.iter_order

        The returned fields are ordered by the order of self.iter_order
        '''
        iters = (iter(self.raw_dataset[source]) for source in self.iter_order)
        return itertools.izip(*iters)

    @property
    def iter_order(self):
        if hasattr(self, '_iter_order'):
            return self._iter_order
        else:
            return self.sources

    @iter_order.setter
    def iter_order(self, value):
        '''Set source order to iterate.

        :param value: list of str
                The source names to iterate over. The returned fields are ordered by given name order
        '''
        self._iter_order = value

    @property
    def sources(self):
        return self.raw_dataset.keys()

    @property
    def sample_num(self):
        return len(self.raw_dataset.values()[0])

    def keys(self):
        return self.raw_dataset.keys()

    def values(self):
        return self.raw_dataset.values()

    def check_dim(self):
        '''Check dimension of each field of the dataset

        If field dimensions mismatch, raise ValueError
        '''
        dim = len(self.raw_dataset.values()[0])
        if not all([len(field) == dim for field in self.raw_dataset.values()]):
            raise ValueError('Dimensions of raw dataset fields mismatch!')

    def cross_split(self, proportion, shuffled=True, seed=None):
        '''Split dataset for cross validation

        Split dataset into part for training and that for validation multiple times.
        Each time the validation dataset is unique without any samples overriding.
        :param proportion: float
                Validation proportion
                size(validation) = portion * size(all dataset)
        :param shuffled: bool
                Indicator if shuffle the dataset before splitting, default is True
        :param seed: int
                Seed for shuffled the dataset. If shuffled is True and no seed given or
                set to be None (default), the dataset will be shuffled randomly.
        :return: tuple of instances of RawDataset
                Training dataset and testing dataset
        '''
        sample_num = self.sample_num
        valid_num = int(sample_num * proportion)
        idxes = np.arange(sample_num)
        if shuffled:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(idxes)

        for i in range(sample_num / valid_num):
            train_raw_dataset = {}
            valid_raw_dataset = {}
            train_idxes = np.concatenate([idxes[:i * valid_num],
                                          idxes[(i + 1) * valid_num:]])
            valid_idxes = idxes[i * valid_num:(i + 1) * valid_num]
            for key, value in self.raw_dataset.items():
                train_raw_dataset[key] = value[train_idxes]
                valid_raw_dataset[key] = value[valid_idxes]
            yield (RawDataset(train_raw_dataset), RawDataset(valid_raw_dataset))

    def split(self, proportion, shuffled=True, seed=None):
        '''Split the dataset into two parts

        Split the dataset into two parts with the first part consisting 'proportion' that of
        the original dataset.
        This operation usually used to split dataset into training and validation parts. Of course,
        you can use this method to split out validation and testing parts.

        :param proportion: float
                The proportion of the first part
                size(first part) = portion * size(all dataset)
        :param shuffled: bool
                Indicator if shuffle the dataset before splitting, default is True
        :param seed: int
                Seed for shuffled the dataset. If shuffled is True and no seed given or
                set to be None (default), the dataset will be shuffled randomly.
        :return: tuple of instances of RawDataset
                Training dataset and testing dataset
        '''
        sample_num = self.sample_num
        part_one_num = int(sample_num * proportion)
        idxes = np.arange(sample_num)
        if shuffled:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(idxes)
        idxes_one = idxes[:part_one_num]
        idxes_two = idxes[part_one_num:]
        raw_dataset_one = {}
        raw_dataset_two = {}
        for key, value in self.raw_dataset.items():
            raw_dataset_one[key] = value[idxes_one]
            raw_dataset_two[key] = value[idxes_two]
        return (RawDataset(raw_dataset_one), RawDataset(raw_dataset_two))


class AbstractDataset(object):
    '''Convert preprocessed raw dataset into stream and extract necessary dataset information

    Initialize dataset:
        1. from file: invoke Initialize() method given file path to load dataset information
        2. pass training dataset first: invoke get_train_datastream() method given training
           raw dataset

        Note: self.initialized is an indicator of weather the dataset have been initialized.
        The above two methods will set it to be True if initializing successfully.
        You cannot process validation or testing dataset without initialization. The

    Save dataset:
        Invoke save() method given file path to save.
        This method will save all the attributes of current instance into given file with cPickle.dump.

    Reset dataset:
        By default, some dataset information will not be erased even you deal with another training dataset
        by invoking get_train_stream() method again. To make sure you can extract exactly new dataset information,
        you can make a new instance or invoke reset() method given attribute names you want to preserve. This will
        erase all the attributes of current instance except those you want to preserve.

    Process dataset:
        It offer get_train_stream(), get_valid_stream() and get_test_stream() methods for dealing with training,
        validation and testing dataset. You should define the sources provided by training or validation stream in
        self.provide_train_sources and those by testing stream in self.provide_test_sources.
        Note: you should make sure the dimension of each field of the raw dataset to be the same, otherwise an
        error will raise.

    What subclass should implement:

        1. Implement _map[_for_<train|valid|test>] methods to convert feature representation. Do common work for
        training, valid and testing dataset in _map method and do specific work in corresponding methods
        2. If you want to do some work before converting feature representation,
        invoke: _process_before_mapping[_for_<train|valid|test>]. All the type of dataset will be processed by
        _process_before_mapping() method
        3. If you want to do some work after converting feature representation,
        invoke: _process_after_mapping[_for_<train|valid|test>]. All the type of dataset will be processed by
        _process_after_mapping() method
        4. Add some transformation to the stream in _construct_shuffled_stream method and _construct_sequential_stream
        methods

    '''
    def __init__(self):
        self.provide_train_sources = None
        self.provide_test_sources = None
        self.initialized = False

    @property
    def train_sample_num(self):
        '''Get number of training sample

        :return: int
        '''
        if hasattr(self, '_train_sample_num'):
            assert isinstance(self._train_sample_num, int)
            return self._train_sample_num
        else:
            return 0

    @property
    def valid_sample_num(self):
        '''Get number of validation sample

        :return: int
        '''
        if hasattr(self, '_valid_sample_num'):
            assert isinstance(self._valid_sample_num, int)
            return self._valid_sample_num
        else:
            return 0

    @property
    def test_sample_num(self):
        '''Get number of testing sample

        :return: int
        '''
        if hasattr(self, '_test_sample_num'):
            assert isinstance(self._test_sample_num, int)
            return self._test_sample_num
        else:
            return 0

    @abstractmethod
    def _process_before_mapping_for_train(self, raw_dataset):
        '''Process training raw dataset before mapping samples into integer representation

         This method is designed to do pre-processing, e.g. statistic word frequency on training dataset.
        :param raw_dataset: list, tuple or numpy.ndarray
                A container of training samples
        :return: list
                A list of processed samples
        '''
        return raw_dataset

    @abstractmethod
    def _process_before_mapping_for_valid(self, raw_dataset):
        '''Process valid raw dataset before mapping samples into integer representation

         This method is designed to do pre-processing, e.g. statistic word frequency, on raw dataset
         on valid raw dataset.
        :param raw_dataset: list, tuple or numpy.ndarray
                A contaniner of valid samples
        :return: list
                A list of processed samples
        '''
        return raw_dataset

    @abstractmethod
    def _process_before_mapping_for_test(self, raw_dataset):
        '''Process test raw dataset before mapping samples into integer representation

         This method is designed to do pre-processing, e.g. statistic word frequency, on raw dataset
         on test raw dataset.
        :param raw_dataset: list, tuple or numpy.ndarray
                A contaniner of test samples
        :return: list
                A list of processed samples
        '''
        return raw_dataset

    @abstractmethod
    def _map_for_train(self, raw_dataset):
        '''Map training raw dataset into integer representation dataset

        :param raw_dataset: dict
                    This stores the attributes of training samples. Every item corresponds an attribute and
                    the length of each attribute should be the same.
        :return: tuple
                A tuple of numpy.ndarray. Each element corresponds, in order, one field of the dataset defined in
                self.provide_train_sources.
        '''
        raise NotImplementedError

    @abstractmethod
    def _map_for_valid(self, raw_dataset):
        raise NotImplementedError

    @abstractmethod
    def _map_for_test(self, raw_dataset):
        raise NotImplementedError

    @abstractmethod
    def _process_after_mapping_for_train(self, dataset):
        '''Process mapped training dataset
        '''
        self._train_sample_num = len(dataset[0])
        return dataset

    @abstractmethod
    def _process_after_mapping_for_valid(self, dataset):
        self._valid_sample_num = len(dataset[0])
        return dataset

    @abstractmethod
    def _process_after_mapping_for_test(self, dataset):
        self._test_sample_num = len(dataset[0])
        return dataset

    @abstractmethod
    def _construct_shuffled_stream(self, dataset, for_type = 'train'):
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
        # Sort samples by size and compact samples with similar size into a batch.
        # stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size * self.sort_batch_count))
        # comparison = _balanced_batch_helper(stream.sources.index(self.compare_source))
        # stream = Mapping(stream, SortMapping(comparison))
        # stream = Unpack(stream)
        # stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        # # Add mask on inputs
        # for source in self.need_mask_sources.iteritems():
        #     stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    @abstractmethod
    def _construct_sequential_stream(self, dataset, for_type = 'train'):
        '''Construc a sequential stream from an IndexableDataset object
        '''
        it = SequentialExampleScheme(dataset.num_examples)
        stream = DataStream(dataset, iteration_scheme=it)
        # # Batch examples
        # stream = Batch(stream, iteration_scheme=ConstantScheme(self.batch_size))
        # Add mask on inputs
        # for source in self.need_mask_sources.iteritems():
        #     stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    def initialize(self, param_load_from):
        '''Initialize dataset information, like word table.


        :param param_load_from: str
                The file path from which to load the dataset parameter. The parameters stored
                in the file should be dumped by cPickle with cPickle.dump(obj, file).
                If it is None (default), dataset information should extracted from training data set.
        :exception: NotImplementedError
                If param_load_from is None, raise NotImplementedError. Subclass may obtain
                param_load_from from config object.
        '''
        try:
            if param_load_from is None:
                raise ValueError('param_load_from should not be None!')
            elif not os.path.exists(param_load_from):
                raise IOError('File:{0} does not exist!'.format(param_load_from))
            else:
                self._load_parameter(param_load_from)
            self.initialized = True
        except Exception as e:
            self.initialized = False
            raise e

    def save(self, param_save_to):
        if param_save_to is None:
            raise ValueError('param_save_to should not be None!')
        elif not os.path.exists(os.path.dirname(param_save_to)):
            os.makedirs(os.path.dirname(param_save_to))
        else:
            pass
        self._save_parameter(param_save_to)

    def reset(self, preserved_attributes=None):
        '''Reset all the dataset parameters, except those in reserved_attributes, to None

        :param preserved_attributes: set
                Set of attributes to be preserved during resetting
        '''
        preserved_attrs = {'provide_train_sources', 'provide_test_sources'}
        if preserved_attrs is not None:
            preserved_attrs.update(preserved_attrs)
        for attr in self.__dict__.keys():
            if attr not in preserved_attrs:
                setattr(self, attr, None)
        self.initialized = False

    def _load_parameter(self, param_load_from):
        '''Load dataset from a pickled file

        :param param_load_from: str
                File path to load the former dataset. It should store a pickled instance
                of class AbstractDataset with cPickle.dump().
                The attributes of curent object will be updated by the loaded object.
        '''
        if param_load_from is None or not os.path.exists(param_load_from):
            raise Exception('Cannot load dataset from {0}'.format(param_load_from))
        with open(param_load_from, 'rb') as f:
            attributes = cPickle.load(f)
            for attr, value in attributes.iteritems():
                setattr(self, attr, value)
            self.initialized = True

    def _save_parameter(self, param_save_to):
        '''Dump current dataset into file

        :param param_save_to: str
                File path to dump current object.
        '''
        with open(param_save_to, 'wb+') as f:
            cPickle.dump(self.__dict__, f)

    def get_train_stream(self, raw_dataset, it='shuffled'):
        '''Construct a fuel.stream object for training from the raw dataset.

        It should do some processings on the raw dataset, like mapping string type of words into integer
        representation, mapping true labels and predicted labels, extracting training data information,
        padding and shuffling samples and so on.

        :param raw_dataset: dic
                    This stores the attributes of training samples. Every item corresponds an attribute and
                    the length of each attribute should be the same.
                    e.g., raw_dataset = [np.array(ids),np.array(ages),np.array(genders)] and
                    len(ids) == len(ages) == len(genders)
        :param it: str
                Assign the iteration schema used in fuel.stream. If it is 'shuffled' (default), samples
                will be shuffled before feeding into training, this is necessary for trainig with Stochastic
                Gradient Descent (SGD) method.
                Optional, 'sequential' is also supported, this option is usually used from testing or
                validation where sample order is important for index consistent.
        :return: fuel.stream
                This is typically feeded into the training processing.
        '''
        try:
            self.initialized = True
            return self._get_stream(raw_dataset, it, for_type='train')
        except Exception as e:
            self.initialized = False
            raise e

    def get_valid_stream(self, raw_dataset, it='sequential'):
        '''Construct a fuel.stream object for validation from the raw dataset.
        '''
        if self.initialized:
            return self._get_stream(raw_dataset, it, for_type='valid')
        else:
            raise Exception('Cannot obtain validation stream for dataset has not been initialized!')

    def get_test_stream(self, raw_dataset, it='sequential'):
        '''Construct a fuel.stream object for test from the raw dataset.
                This is typically fed into the test processing.
        '''
        if self.initialized:
            return self._get_stream(raw_dataset, it, for_type='test')
        else:
            raise Exception('Cannot obtain testing stream for dataset has not been initialized!')

    def _get_stream(self, raw_dataset, it='shuffled', for_type='train'):
        if not isinstance(raw_dataset, dict) or len(raw_dataset) == 0:
            raise ValueError('raw_dataset should be a non empty dict!')
        compared_source = raw_dataset.values()[0]
        if not all([len(field) == len(compared_source) for field in raw_dataset.values()]):
            raise ValueError('Field dimension mismatch!')
        raw_dataset = self._process_before_mapping(raw_dataset, for_type)
        dataset = self._map(raw_dataset, for_type)
        dataset = self._process_after_mapping(dataset, for_type)
        dataset = self._construct_dataset(dataset, for_type)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset, for_type)
        elif it == 'sequential':
            return self._construct_sequential_stream(dataset, for_type)
        else:
            raise ValueError('it should be "shuffled" or "sequential"!')

    def _map(self, raw_dataset, for_type):
        '''Map raw dataset into integer representation dataset

        :param raw_dataset: dict
                This stores the attributes of training samples. Every item corresponds an attribute and
                the length of each attribute should be the same.
        :param for_type: str
                Indicator of the usage of the raw dataset
        :return: tuple
                A tuple of numpy.ndarray. Each element corresponds, in order, one field of the dataset defined in
                self.provide_train_sources.
        '''
        if for_type == 'train':
            return self._map_for_train(raw_dataset)
        elif for_type == 'valid':
            return self._map_for_valid(raw_dataset)
        elif for_type == 'test':
            return self._map_for_test(raw_dataset)
        else:
            raise ValueError('{0} is not supported!'.format(for_type))

    def _process_before_mapping(self, raw_dataset, for_type):
        '''Process raw dataset before mapping samples into integer representation

         This method is designed to do pre-processing, e.g. statistic word frequency, on raw dataset 
         depending on the usage of the dataset. 
        :param raw_dataset: list, tuple or numpy.ndarray
                A contaniner of samples
        :param for_type: str
                Indicator of the usage of this dataset: 'train','valid' or 'test'
        :return: list
                A list of samples
        '''
        if for_type == 'train':
            return self._process_before_mapping_for_train(raw_dataset)
        elif for_type == 'valid':
            return self._process_before_mapping_for_valid(raw_dataset)
        elif for_type == 'test':
            return self._process_before_mapping_for_valid(raw_dataset)
        else:
            raise ValueError('{0} for "for_type" is not supported!'.format(for_type))

    def _process_after_mapping(self, dataset, for_type='train'):
        '''Process mapped dataset

        :param dataset: tuple
                Tuple of numpy.ndarray or list. Each element of the list corresponds a field of the dataset
                defined in self.provide_souces
        :param for_type: str
                Indicator of the usage of this dataset: 'train','valid' or 'test'
        :return: list
                A new list of dataset.
        '''
        if for_type == 'train':
            return self._process_after_mapping_for_train(dataset)
        elif for_type == 'valid':
            return self._process_after_mapping_for_valid(dataset)
        elif for_type == 'test':
            return self._process_after_mapping_for_valid(dataset)
        else:
            raise ValueError('{0} for "for_type" is not supported!'.format(for_type))

    def _construct_dataset(self, dataset, for_type):
        '''Construct an fuel indexable dataset.

        Every field corresponds to the name.
        :param dataset: A tuple of numpy.ndarray
        :return: instance of IndexableDataset
        '''
        if for_type == 'train' or for_type == 'valid':
            sources = self.provide_train_sources
        else:
            sources = self.provide_test_sources
        return IndexableDataset(indexables=OrderedDict(zip(sources, dataset)))


class AbstractClassificationDataset(AbstractDataset):

    @abstractproperty
    def label_num(self):
        '''Define the number of labels for classification

        :return: int
            The number of labels. This is typical for classification problem.
        '''
        raise NotImplementedError

    @property
    def true_label2pred_label(self):
        '''Get the mapping from predicted label to true label

        Sometimes, the true labels are in symbolic representation or not start from zero. For classification
        task, these labels should be mapped into integers starting with 0. User can manually define the mapping
        rules here. Make sure your mapped label should start with 0 and be continuous.

        :return: dict
            A dictionary mapping the true label to predicted label
        '''
        if hasattr(self, '_true_label2pred_label'):
            return self._true_label2pred_label
        else:
            return None

    @true_label2pred_label.setter
    def true_label2pred_label(self, value):
        self._true_label2pred_label = value

    @property
    def pred_label2true_label(self):
        '''Get the mapping from predicted label to true label

        Sometimes, the true labels are in symbolic representation or not start from zero. For classification
        task, these labels should be mapped into integers starting with 0. In order to make the prediction
        of system easy to understand, it should map the predicted labels back to the true labels. This provide
        the mapping relationship from predicted label to true label.

        :return: dict
            A dictionary mapping the predicted label to true label
        '''
        if hasattr(self, '_pred_label2true_label'):
            return self._pred_label2true_label
        elif self.true_label2pred_label is not None:
            key, value = self.true_label2pred_label.items()
            self._pred_label2true_label = dict(zip(value, key))
            return self._pred_label2true_label.copy()
        else:
            return None


class AbstractDocClassificationDataset(AbstractClassificationDataset):

    @abstractproperty
    def token_num(self):
        '''Define token number, usually used as feature dimension

        :return: int
            The number of tokens (word number, character number and so on)
        '''
        raise NotImplementedError


def test_raw_dataset():
    sources = ['name', 'age', 'gender']
    values = [np.random.rand(10), np.random.rand(10), np.random.rand(10)]
    raw_dataset = RawDataset(dict(zip(sources, values)))
    for train, valid in raw_dataset.cross_split(proportion=0.2, shuffled=False):
        pass
    print(len(raw_dataset))
    print(raw_dataset.sources)
    raw_dataset.iter_order = sources
    for sample in raw_dataset:
        print(sample)
    print('\n')
    raw_dataset.iter_order = raw_dataset.sources
    for sample in raw_dataset:
        print(sample)


if __name__ == '__main__':
    test_raw_dataset()