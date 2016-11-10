import os
import errno
from collections import Iterable
from warnings import warn
from abc import abstractmethod, abstractproperty
from collections import OrderedDict

import theano
import numpy as np
from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledExampleScheme, ConstantScheme, SequentialScheme, SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers import Batch, Padding, Unpack

class AbstractDataset(object):
    def __init__(self, *args, **kwargs):
        self.provide_souces = None
        self.initialized = False

    @abstractmethod
    def get_parameter_to_save(self):
        '''Return parameters that need to be saved with model

        :return: OrderedDict
                This dictionary stores the information of training dataset used for training.
                Traing dataset information usually contains word table, sparse word threshold,
                mapping of true labels to predicted labels and so on.
        '''
        raise NotImplementedError('get_parameter_to_save is not implemented!')

    @abstractmethod
    def save_paramter(self, param_save_to = None, *args, **kwargs):
        pass

    @abstractmethod
    def _initialize(self, param_load_from = None):
        '''Initialize dataset information

        Note:
        If the dataset has been initialized successfully, self.initialized should be set True else False
        '''
        raise NotImplementedError

    @abstractmethod
    def _process_before_mapping_for_train(self, raw_dataset):
        '''Process training raw dataset before mapping samples into integer representation

         This method is designed to do pre-processing, e.g. statistic word frequency, on raw dataset
         on training raw dataset.
        :param raw_dataset: list, tuple or numpy.ndarray
                A contaniner of training samples
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
    def _map(self, raw_dataset, for_type='train'):
        '''Map raw dataset into integer representation dataset

        :param raw_dataset: list or tuple
                    This stores the attributes of training samples. Every item corresponds an attribute and
                    the length of each attribute should be the same.
                    e.g., raw_dataset = [list_of_ids,list_of_ages,list_of_genders] and
                    len(list_of_ids) == len(list_of_ages) == len(list_of_genders)
        :param for_type: str
                Indicator of the usage of this dataset: 'train','valid' or 'test'
        :return: list
                A list of numpy.ndarray or lists. Each element corresponds, in order, one field of the dataset defined in
                self.provide_sources.
        '''
        raise NotImplementedError

    @abstractmethod
    def _process_after_mapping_for_train(self, dataset):
        '''Process mapped training dataset

        :return: list
                A new list of dataset.
        '''
        return dataset

    @abstractmethod
    def _process_after_mapping_for_valid(self, dataset):
        '''Process mapped valid dataset

        :param dataset:
        :return: list
                A new list of dataset.
        '''
        return dataset

    @abstractmethod
    def _process_after_mapping_for_test(self, dataset):
        '''Process mapped test dataset

        :param dataset:
        :return: list
                A new list of dataset.
        '''
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

    def initialize(self, param_load_from=None):
        '''Initialize dataset information, like word table.

        If param_load_from is None, traing dataset should be provided and the dataset information will
        be extracted or overrided from the training data set.
        :param param_load_from: str
                The file path from which to load the dataset parameter. The paramters stored
                in the file should be dumped by cPickle with cPickle.dump(obj, file).
                If it is None (default), dataset information should extracted from training data set.
        :param args:
        :param kwargs:
        :exception: IOError
                And IOError is raised if param_load_from is not None and file path not exists. This is designed
                to monitor occasional error by user.
        '''
        if param_load_from is not None and (not os.path.exists(param_load_from)):
            warn('{0} not exist and cannot load parameters from that to initialize dataset!\n'
                 'Do not do testing before training!'.format(param_load_from))
        self._initialize(param_load_from)

    def get_train_stream(self, raw_dataset, it='shuffled'):
        '''Construct a fuel.stream object for training from the raw dataset.

        It should do some processings on the raw dataset, like mapping string type of words into integer
        representation, mapping true labels and predicted labels, extracting training data information,
        padding and shuffling samples and so on.

        :param raw_dataset: list or tuple
                    This stores the attributes of training samples. Every item corresponds an attribute and
                    the length of each attribute should be the same.
                    e.g., raw_dataset = [list_of_ids,list_of_ages,list_of_genders] and
                    len(list_of_ids) == len(list_of_ages) == len(list_of_genders)
        :param it: str
                Assign the iteration schema used in fuel.stream. If it is 'shuffled' (default), samples
                will be shuffled before feeding into training, this is necessary for trainig with Stochastic
                Gradient Descent (SGD) method.
                Optional, 'sequencial' is also supported, this option is usually used from testing or
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

    def get_valid_stream(self, raw_dataset, it='sequencial'):
        '''Construct a fuel.stream object for validation from the raw dataset.

        It should do some processings on the raw dataset, like mapping string type of words into integer
        representation, extracting training data information, padding and shuffling samples and so on.

        :param raw_dataset: list or tuple
                This stores the attributes of training samples. Every item corresponds an attribute and
                the length of each attribute should be the same.
                e.g., raw_dataset = [list_of_ids,list_of_ages,list_of_genders] and
                len(list_of_ids) == len(list_of_ages) == len(list_of_genders)
        :param it: str
                Assign the iteration schema used in fuel.stream. The default is 'sequencial' which will keep 
                the order of samples. If it is 'shuffled', samples will be shuffled before feeding into validation.
                This option is usually used during training with SGD.
                
        :return: fuel.stream
                This is typically feeded into the validation processing.
        '''
        if self.initialized:
            return self._get_stream(raw_dataset, it, for_type='valid')
        else:
            raise Exception('Cannot obtain validation stream for dataset has not been initialized!')

    def get_test_stream(self, raw_dataset, it='sequencial'):
        '''Construct a fuel.stream object for test from the raw dataset.
        
        It should do some processings on the raw dataset, like mapping string type of words into integer
        representation, extracting training data information, padding and shuffling samples and so on.

        :param raw_dataset: list or tuple
                    This stores the attributes of training samples. Every item corresponds an attribute and
                    the length of each attribute should be the same.
                    e.g., raw_dataset = [list_of_ids,list_of_ages,list_of_genders] and
                    len(list_of_ids) == len(list_of_ages) == len(list_of_genders)
        :param it: str
                Assign the iteration schema used in fuel.stream. The default is 'sequencial' which will keep
                the order of samples. If it is 'shuffled', samples will be shuffled before feeding into test.
                This option is usually used during training with SGD.

        :return: fuel.stream
                This is typically feeded into the test processing.
        '''
        if self.initialized:
            return self._get_stream(raw_dataset, it, for_type='test')
        else:
            raise Exception('Cannot obtain testing stream for dataset has not been initialized!')

    def _get_stream(self, raw_dataset, it='shuffled', for_type='train'):

        if not isinstance(raw_dataset, Iterable) or len(raw_dataset) == 0:
            raise ValueError('raw_dataset should be a non empty list or tuple!')
        l = len(raw_dataset[0])
        for attr in raw_dataset:
            if len(attr) != l:
                raise ValueError('The length of attributes of raw_dataset are not the same!')
            else:
                pass
        raw_dataset = self._process_before_mapping(raw_dataset, for_type)
        dataset = self._map(raw_dataset, for_type)
        dataset = self._process_after_mapping(dataset, for_type)
        dataset = self._construct_dataset(dataset)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset, for_type)
        elif it == 'sequencial':
            return self._construct_sequential_stream(dataset, for_type)
        else:
            raise ValueError('it should be "shuffled" or "sequencial"!')

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

        :param dataset: list
                List of numpy.ndarray or list. Each element of the list corresponds a field of the dataset
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

    def _construct_dataset(self, dataset):
        '''Construct an fuel indexable dataset.

        Every field corresponds to the name of self.provide_sources
        :param dataset: A tuple of data
        :return:
        '''
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))


class AbstractClssificationDataset(AbstractDataset):

    @abstractproperty
    def label_num(self):
        '''Define the number of labels for classification

        :return: int
            The number of labels. This is typical for classification problem.
        '''
        if hasattr(self, '_label_num'):
            if not isinstance(self._label_num, int):
                raise TypeError('_label_num should be an integer!')
            else:
                return self._label_num
        else:
            raise ValueError('_label_num is not defined in the dataset class')

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
        else:
            return None


class AbstractDocClassificationDataset(AbstractClssificationDataset):

    @abstractproperty
    def token_num(self):
        '''Define token number, usually used as feature dimension

        :return: int
            The number of tokens (word number, character number and so on)
        '''
        if hasattr(self, '_token_num'):
            return self._token_num
        else:
            raise ValueError('_token_num is not defined in the dataset class')


