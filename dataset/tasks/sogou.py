import os
import cPickle
from collections import OrderedDict
from abc import abstractproperty, abstractmethod

import theano
import numpy as np

from ..tasks import *
from ..tasks import _balanced_batch_helper, OutputNoise, MatrixPadding, BaggedQuerySample
from ..base import AbstractDocClassificationDataset


class BUTHD(AbstractDocClassificationDataset):
    def __init__(self, config):
        self.config = config
        self.provide_souces = ('user', 'text', 'hashtag')
        self.need_mask_sources = {'text': theano.config.floatX}
        self.compare_source = 'text'
        self._initialize()

    @abstractmethod
    def _initialize(self, model_path = None, *args, **kwargs):
        '''
        Initialize dataset information
        '''
        raise NotImplementedError

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict of{name:data,...}
        '''
        raise NotImplementedError

    def get_train_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='train')

    def get_valid_stream(self, raw_dataset, it='sequencial'):
        return self._get_stream(raw_dataset, it, for_type='valid')

    def get_test_stream(self, raw_dataset, it='sequencial'):
        return self._get_stream(raw_dataset, it, for_type='test')

    def _get_stream(self, raw_dataset, it='shuffled', for_type='train'):
        raw_dataset = self._update_before_transform(raw_dataset, for_type)
        dataset = self._map(raw_dataset, for_type)

        dataset = self._update_after_transform(dataset, for_type)
        dataset = self._construct_dataset(dataset)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset, for_type)
        elif it == 'sequencial':
            return self._construct_sequencial_stream(dataset, for_type)
        else:
            raise ValueError('it should be "shuffled" or "sequencial"!')

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        return raw_dataset

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    @abstractmethod
    def _map(self, raw_dataset, for_type='train'):
        '''
        Turn raw_dataset into index representation dataset.

        Note: Implement this function in subclass
        '''

        raise NotImplementedError

    def _construct_dataset(self, dataset):
        '''
        Construct an fule indexable dataset.
        Every data corresponds to the name of self.provide_sources
        :param dataset: A tuple of data
        :return:
        '''
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset, for_type = 'train'):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
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
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    def _construct_sequencial_stream(self, dataset, for_type = 'train'):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream


class UTHD(BUTHD):
    '''
    UTHD with only user, word and hashtag embeddings
    '''

    def __init__(self, config):
        super(UTHD, self).__init__(config)
        self.provide_souces = ('user', 'text', 'hashtag')

    @abstractmethod
    def _initialize(self, model_path = None, *args, **kwargs):
        if model_path is None:
            model_path = self.config.model_path
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.user2index = dataset_prarms['user2index']
                self.word2index = dataset_prarms['word2index']
                self.hashtag2index = dataset_prarms['hashtag2index']
                self.user2freq = dataset_prarms['user2freq']
                self.word2freq = dataset_prarms['word2freq']
                self.hashtag2freq = dataset_prarms['hashtag2freq']
                self.sparse_word_threshold = dataset_prarms['sparse_word_threshold']
                self.sparse_user_threshold = dataset_prarms['sparse_user_threshold']
                self.sparse_hashtag_threshold = dataset_prarms['sparse_hashtag_threshold']
                return dataset_prarms
        else:
            # Dictionary
            self.user2index = {'<unk>': 0}  # Deal with OV when testing
            self.hashtag2index = {'<unk>': 0}
            self.word2index = {'<unk>': 0}
            self.word2freq = {}
            self.user2freq = {}
            self.hashtag2freq = {}
            self.sparse_word_threshold = 0
            self.sparse_user_threshold = 0
            self.sparse_hashtag_threshold = 0
            return None

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        return OrderedDict(
            {'hashtag2index': self.hashtag2index, 'word2index': self.word2index, 'user2index': self.user2index,
             'user2freq': self.user2freq, 'word2freq': self.word2freq, 'hashtag2freq': self.hashtag2freq,
             'sparse_word_threshold': self.sparse_word_threshold, 'sparse_user_threshold': self.sparse_user_threshold,
             'sparse_hashtag_threshold': self.sparse_hashtag_threshold})

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        if for_type == 'train':
            fields = zip(*raw_dataset)
            self.word2freq = self._extract_word2freq(fields[self.config.text_index])
            self.user2freq = self._extract_user2freq(fields[self.config.user_index])
            self.hashtag2freq = self._extract_hashtag2freq(fields[self.config.hashtag_index])
            #region Define sparse item with percent
            # self.sparse_word_threshold = get_sparse_threshold(self.word2freq.values(), self.config.sparse_word_percent)
            # self.sparse_user_threshold = get_sparse_threshold(self.user2freq.values(), self.config.sparse_user_percent)
            # self.sparse_hashtag_threshold = get_sparse_threshold(self.hashtag2freq.values(),
            #                                                      self.config.sparse_hashtag_percent)
            #endregion
            self.sparse_hashtag_threshold = self.config.sparse_hashtag_freq
            self.sparse_user_threshold = self.config.sparse_user_freq
            self.sparse_word_threshold = self.config.sparse_word_freq
            return raw_dataset
            # Implement more updation
        elif for_type == 'valid' or for_type == 'test':
            return raw_dataset
        else:
            raise ValueError('for_type should be either "train" or "test"')

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    def _map(self, raw_dataset, for_type='train'):
        '''
        Turn string type user, words of context, hashtag representation into index representation.

        Note: Implement this function in subclass
        '''
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        users = np.array(
            [self._get_user_index(user, for_type) for user in fields[self.config.user_index]],
            dtype=self.config.int_type)
        hashtags = np.array([self._get_hashtag_index(hashtag, for_type) for hashtag in
                                fields[self.config.hashtag_index]],
                               dtype=self.config.int_type)

        texts = [np.array([self._get_word_index(word, for_type) for word in text],
                             dtype=self.config.int_type)
                 for text in fields[self.config.text_index]]
        return (users, texts, hashtags)

    def _extract_word2freq(self, texts):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        id, count = np.unique(np.concatenate(np.array(texts)), return_counts=True)
        return dict(zip(id, count))

    def _extract_user2freq(self, users):
        assert users is not None
        id, count = np.unique(np.array(users), return_counts=True)
        return dict(zip(id, count))

    def _extract_hashtag2freq(self, hashtag):
        assert hashtag is not None
        id, count = np.unique(np.array(hashtag), return_counts=True)
        return dict(zip(id, count))

    def _is_sparse_hashtag(self, hashtag):
        if hashtag in self.hashtag2freq and self.hashtag2freq[hashtag] >= self.sparse_hashtag_threshold:
            return False
        else:
            return True

    def _is_sparse_word(self, word):
        if word in self.word2freq and self.word2freq[word] >= self.sparse_word_threshold:
            return False
        else:
            return True

    def _is_sparse_user(self, user):
        if user in self.user2freq and self.user2freq[user] >= self.sparse_user_threshold:
            return False
        else:
            return True

    def _get_hashtag_index(self, hashtag, for_type='train'):
        if self._is_sparse_hashtag(hashtag):
            return self.hashtag2index['<unk>']
        else:
            return self._get_index(hashtag, self.hashtag2index, for_type)

    def _get_user_index(self, user, for_type='train'):
        if self._is_sparse_user(user):
            return self.user2index['<unk>']
        else:
            return self._get_index(user, self.user2index, for_type)

    def _get_word_index(self, word, for_type='train'):
        if self._is_sparse_word(word):
            return self.word2index['<unk>']
        else:
            return self._get_index(word, self.word2index, for_type)

    def _get_index(self, item, _dic, for_type='train'):
        if item not in _dic:
            if for_type == 'train':
                _dic[item] = len(_dic)
                return len(_dic) - 1
            else:
                raise Exception('Cannot get index of '+item)
        else:
            return _dic[item]


class AQD(UTHD):
    def __init__(self, config):
        super(AQD, self).__init__(config)
        self.provide_souces = ('age',  'age_mask', 'gender', 'gender_mask','edu', 'edu_mask','query')
        self.need_mask_sources = {'query':theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        if hasattr(self, '_label_num'):
            return self._label_num
        else:
            raise ValueError

    @property
    def token_num(self):
        if hasattr(self, '_word_num'):
            return self._word_num
        else:
            raise ValueError

    @property
    def true_label2pred_label(self):
        if hasattr(self, '_true_label2pred_label'):
            return self._true_label2pred_label
        else:
            labels = [(3, 2), (3, 3), (3, 4), (3, 5),
                      (4, 1), (4, 2), (4, 3), (4, 4),
                      (4, 5), (5, 1), (5, 2), (6, 1),
                      (6, 0)]
            self._true_label2pred_label = dict(zip(labels, range(len(labels))))
            self._pred_label2true_label = dict(zip(range(len(labels)), labels))
            self._label_num = len(labels)
            return self._true_label2pred_label

    @property
    def pred_label2true_label(self):
        if hasattr(self, '_pred_label2true_label'):
            return self._pred_label2true_label
        else:
            return {}

    @abstractmethod
    def _initialize(self, model_path = None, *args, **kwargs):
        if model_path is None:
            model_path = self.config.model_load_from
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                cPickle.load(f)
                dataset_prarms = cPickle.load(f)
                self.char2index = dataset_prarms['char2index']
                self.char2freq = dataset_prarms['char2freq']
                return dataset_prarms
        else:
            # Dictionary
            self.char2index = {'<unk>': 0}  # Deal with OV when testing
            self.char2freq = {}
            return None

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict
        '''
        return OrderedDict(
            {'char2index': self.char2index, 'char2freq':self.char2freq})

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        if for_type == 'train':
            fields = zip(*raw_dataset)
            self.char2freq = self._extract_char2freq(fields[self.config.query_index])
        return raw_dataset

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        if for_type == 'train':
            zip_dataset = zip(*dataset)
            ages = dataset[0]
            age, count = np.unique(ages, return_counts=True)
            age2freq = dict(zip(age, count))
            zip_dataset = self._balance_class(zip_dataset, ages, age2freq, self.config.age_up_sample_k,
                                              self.config.age_min_per)
            unzip_dataset = zip(*zip_dataset)

            genders = unzip_dataset[2]
            gender, count = np.unique(genders, return_counts=True)
            gender2freq = dict(zip(gender, count))
            zip_dataset = self._balance_class(zip_dataset, genders, gender2freq, self.config.gender_up_sample_k,
                                              self.config.gender_min_per)
            unzip_dataset = zip(*zip_dataset)

            edus = unzip_dataset[4]
            edu, count = np.unique(edus, return_counts=True)
            edu2freq = dict(zip(edu, count))
            zip_dataset = self._balance_class(zip_dataset, edus, edu2freq, self.config.edu_up_sample_k,
                                              self.config.edu_min_per)
            unzip_dataset = zip(*zip_dataset)

            ages = unzip_dataset[0]
            age, count = np.unique(ages, return_counts=True)
            self.age2freq = dict(zip(age, count))

            genders = unzip_dataset[2]
            gender, count = np.unique(genders, return_counts=True)
            self.gender2freq = dict(zip(gender, count))

            edus = unzip_dataset[4]
            edu, count = np.unique(edus, return_counts=True)
            self.edu2freq = dict(zip(edu, count))
            dataset = unzip_dataset
        return dataset

    def _extract_char2freq(self, all_user_queries):
        '''
        Count word frequency
        :param texts:
        :return:
        '''
        whole_queries = [np.concatenate(np.array(queries)).tolist() for queries in all_user_queries]
        id, count = np.unique(np.concatenate(np.array(whole_queries)), return_counts=True)
        return dict(zip(id, count))

    def _balance_class(self, dataset, labels, label2freq, up_sample_k, min_per, *args, **kwargs):
        assert len(dataset) == len(labels)
        labels = np.array(labels)
        max_freq = 1.*max(label2freq.values())
        label2sample_num = {}
        for label, freq in label2freq.items():
            label2sample_num[label] = max(int(freq* ((max_freq / freq) ** up_sample_k -1.)), int(max_freq * min_per))

        full_idxes = np.arange(len(labels))
        add_idxes = []
        for label in label2freq.keys():
            if label2sample_num[label] > 0:
                idxes = full_idxes[labels==label]
                rvs = np.random.randint(low=0, high=len(idxes), size = label2sample_num[label])
                add_idxes += idxes[rvs].tolist()
        added_dataset = [dataset[idx] for idx in add_idxes]
        return dataset + added_dataset

    def _map(self, raw_dataset, for_type='train'):
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        age, age_mask = self._add_mask(fields[self.config.age_index])
        gender, gender_mask = self._add_mask(fields[self.config.gender_index])
        edu, edu_mask = self._add_mask(fields[self.config.edu_index])
        queries_per_user = [[np.array(list(set([self._get_char_index(char, for_type) for char in query])), dtype=self.config.int_type)
                             for query in queries] for queries in fields[self.config.query_index]]
        return (age, age_mask, gender, gender_mask, edu, edu_mask, queries_per_user)

    def _add_mask(self, label, *args, **kwargs):
        '''
        Add mask on labels
        :param label:
        :param args:
        :param kwargs:
        :return:
        '''
        label = np.array(label, dtype=self.config.int_type) - 1
        label_mask = np.ones(label.shape, dtype=theano.config.floatX)
        idxes = label == -1
        label[idxes] = 0
        label_mask[idxes] = 0.
        return (label, label_mask)

    def _is_sparse_char(self, char):
        if not char in self.char2freq or self.char2freq[char] < self.config.sparse_char_freq:
            return True
        else:
            return False

    def _get_char_index(self, char, for_type='train'):
        if self._is_sparse_char(char):
            return self.char2index['<unk>']
        else:
            return self._get_index(char, self.char2index, for_type)

    def _construct_shuffled_stream(self, dataset, for_type='train'):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
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
                                   for_type = for_type)
        if for_type == 'train':
            stream = FeatureSample(stream, 'query_mask',
                                   self.config.word_sample_prob)
            stream = OutputNoise(stream, output_source='age',
                                 label2freq= self.age2freq,
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

    def _construct_sequencial_stream(self, dataset, for_type='train'):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = MatrixPadding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        stream = BaggedQuerySample(stream,
                                   sample_source='query',
                                   sample_prob=self.config.query_sample_prob,
                                   for_type = for_type)
        if for_type == 'train':
            stream = FeatureSample(stream, 'query_mask',
                                   self.config.word_sample_prob)
            stream = OutputNoise(stream, output_source='age',
                                 label2freq= self.age2freq,
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


class CombinedDataset(AQD):
    def __init__(self, config):
        super(CombinedDataset, self).__init__(config)
        self.provide_souces = (self.config.main_task_name, self.config.assist_task_name, 'combined_label', 'query')

    def _map(self, raw_dataset, for_type='train'):
        assert raw_dataset is not None or len(raw_dataset) > 0
        fields = zip(*raw_dataset)
        main_task_label = numpy.array(fields[self.config.main_task_label_index], dtype=self.config.int_type)
        assist_task_label = numpy.array(fields[self.config.assist_task_label_index], dtype=self.config.int_type)
        queries_per_user = numpy.array([[numpy.array(list(set([self._get_char_index(char, for_type) for char in query])),
                                         dtype=self.config.int_type)
                             for query in queries] for queries in fields[self.config.query_index]], dtype='O')
        ts = [(i, self.true_label2pred_label[(main_task_label[i],assist_task_label[i])])
                             for i in range(len(main_task_label)) if (main_task_label[i], assist_task_label[i])
                             in self.true_label2pred_label]
        self._filter_percent = 1.*len(ts) / (main_task_label > 0).sum()
        idxes, labels = zip(*ts)
        idxes = numpy.array(idxes)
        labels = numpy.array(labels, dtype=self.config.int_type)
        main_task_label = main_task_label[idxes]
        assist_task_label = assist_task_label[idxes]
        queries_per_user = queries_per_user[idxes]
        return (main_task_label, assist_task_label, labels, queries_per_user)

    @property
    def label_num(self):
        if hasattr(self, '_label_num'):
            return self._label_num
        else:
            raise ValueError

    @property
    def word_num(self):
        if hasattr(self, '_word_num'):
            return self._word_num
        else:
            raise ValueError

    @property
    def true_label2pred_label(self):
        if hasattr(self, '_true_label2pred_label'):
            return self._true_label2pred_label
        else:
            labels = [(3,2),(3,3),(3,4),(3,5),
                   (4,1),(4,2),(4,3),(4,4),
                   (4,5),(5,1),(5,2),(6,1),
                   (6,0)]
            self._true_label2pred_label = dict(zip(labels, range(len(labels))))
            self._pred_label2true_label = dict(zip(range(len(labels)), labels))
            self._label_num = len(labels)
            return self._true_label2pred_label

    @property
    def pred_label2true_label(self):
        if hasattr(self, '_pred_label2true_label'):
            return self._pred_label2true_label
        else:
            return {}

    @property
    def filter_percent(self):
        if hasattr(self, '_filter_percent'):
            return self._filter_percent
        else:
            return 1.

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        if for_type == 'train':
            zip_dataset = zip(*dataset)
            combined_labels = dataset[2]
            label, count = numpy.unique(combined_labels, return_counts=True)
            label2freq = dict(zip(label, count))
            zip_dataset = self._balance_class(zip_dataset, combined_labels, label2freq,
                                              self.config.main_task_up_sample_k, 0.)
            unzip_dataset = zip(*zip_dataset)
            label, count = numpy.unique(unzip_dataset[2], return_counts=True)
            self.combined_label2freq = dict(zip(label, count))
            dataset = unzip_dataset
            for i in range(self.label_num):
                self.combined_label2freq.setdefault(i, 0)
        self._word_num = len(self.char2index)
        return dataset

    def _construct_shuffled_stream(self, dataset, for_type='train'):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
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
                                   self.config.word_sample_prob)
            stream = OutputNoise(stream, output_source='combined_label',
                                 label2freq=self.combined_label2freq,
                                 max_noise_prob=self.config.main_task_max_noise,
                                 decay_rate=self.config.main_task_decay_rate)
        return stream

    def _construct_sequencial_stream(self, dataset, for_type='train'):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = MatrixPadding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        stream = BaggedQuerySample(stream,
                                   sample_source='query',
                                   sample_prob=self.config.query_sample_prob,
                                   for_type=for_type)
        if for_type == 'train':
            stream = FeatureSample(stream, 'query_mask',
                                   self.config.word_sample_prob)
            stream = OutputNoise(stream, output_source='combined_label',
                                 label2freq=self.combined_label2freq,
                                 max_noise_prob=self.config.main_task_max_noise,
                                 decay_rate=self.config.main_task_decay_rate)
        return stream


class SingleTaskDataset(AbstractDocClassificationDataset):
    def __init__(self, config, task_name, true_label2pred_label = None, *args, **kwargs):
        super(SingleTaskDataset, self).__init__(*args, **kwargs)
        self.config = config
        self.task_name = task_name
        self.true_label2pred_label = true_label2pred_label
        self.provide_souces = (task_name, 'query')
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'

    @property
    def label_num(self):
        self._label_num = len(self.true_label2pred_label)
        return super(SingleTaskDataset, self).label_num()

    @property
    def token_num(self):
        self._token_num = len(self.token2index)
        return super(SingleTaskDataset, self).token_num()

    @property
    def token2index(self):
        if hasattr(self, '_token2index'):
            return self._token2index.copy()
        else:
            raise ValueError('_token2index has not been defined!')
    
    @property
    def token2freq(self):
        if hasattr(self, '_token2freq'):
            return self._token2freq.copy()
        else:
            raise ValueError('_token2freq has not been defined!')

    @property
    def train_sample_num(self):
        '''Get number of training sample

        :return: int
        '''
        if hasattr(self, '_train_sample_num'):
            assert isinstance(self._train_sample_num, int)
            return self._train_sample_num
        else:
            raise ValueError('_train_sample_num is not defined!')

    @property
    def valid_sample_num(self):
        '''Get number of validation sample

        :return: int
        '''
        if hasattr(self, '_valid_sample_num'):
            assert isinstance(self._valid_sample_num, int)
            return self._valid_sample_num
        else:
            raise ValueError('_valid_sample_num is not defined!')

    @property
    def test_sample_num(self):
        '''Get number of testing sample

        :return: int
        '''
        if hasattr(self, '_test_sample_num'):
            assert isinstance(self._test_sample_num, int)
            return self._test_sample_num
        else:
            raise ValueError('_test_sample_num is not defined!')

    def _initialize(self, param_load_from=None, *args, **kwargs):
        '''Initialize dataset information
        '''
        if param_load_from is None:
            param_load_from = self.config.dataset_param_load_from
        if os.path.exists(param_load_from):
            with open(param_load_from, 'rb') as f:
                cPickle.load(f)
                dataset_params = cPickle.load(f)
                self._token2index = dataset_params['token2index']
                self._token2freq = dataset_params['token2freq']
                self._true_label2pred_label = dataset_params['true_label2pred_label']
                self._pred_label2true_label = dataset_params['pred_label2true_label']
                self.initialized = True
                return dataset_params
        else:
            # Dictionary
            self._token2index = {'<unk>': 0}  # Deal with OV when testing
            self._token2freq = {}
            self.initialized = False
            return None

    def get_parameter_to_save(self):
        '''Return parameters that need to be saved with model
        :return: OrderedDict of{name:data,...}
        '''
        return OrderedDict(
            {'token2index': self.token2index, 'token2freq':self.token2freq,
             'true_label2pred_label':self.true_label2pred_label,
             'pred_label2true_label':self.pred_label2true_label})

    def _process_before_mapping_for_train(self, raw_dataset):
        '''Extract frequency of tokens

         This method is designed to do pre-processing, e.g. statistic token frequency, on raw dataset
         on training raw dataset.
        :param raw_dataset: list or tuple
                    This stores the user ids, user labels of given task and queries posts by the user
        :return: list
                Original raw_dataset
        '''
        self._token2freq = self._extract_token2freq(raw_dataset[2])
        return raw_dataset
    
    def _extract_token2freq(self, all_user_queries):
        '''Count frequency of tokens
        :param all_user_queries: list of list
                Queries are stored in a list by user, and each user's queries are stored in a list with 
                each item of the list corresponds a query post by the user.
        :return: dict
                {token:frequency of token}
        '''
        whole_queries = [np.concatenate(np.array(queries)).tolist() for queries in all_user_queries]
        id, count = np.unique(np.concatenate(np.array(whole_queries)), return_counts=True)
        return dict(zip(id, count))
    
    def _map(self, raw_dataset, for_type='train'):
        '''Map raw dataset into integer representation dataset

        :param raw_dataset: list or tuple
                    This stores the user ids, user labels of given task and queries posts by the user
        :param for_type: str
                Indicator of the usage of this dataset: 'train','valid' or 'test'
        :return: list
                A list of np.ndarray or lists. Each element corresponds, in order, one field of the dataset defined in
                self.provide_sources.
        '''
        ids = np.array(raw_dataset[0])
        if self.true_label2pred_label is None:
            labels = np.array([self._convert_label(label) for label in raw_dataset[1]])
            idxes = np.arange(len(labels))
        else:
            idxes, labels = zip(*[(i, self.true_label2pred_label[label]) for i, label in enumerate(raw_dataset[1])
                               if label in self.true_label2pred_label])
            idxes = np.array(idxes)
            labels = np.array(labels)
        queries_per_user = np.array(
            [[np.array(list(set([self._get_token_index(char, for_type) for char in query])), # one hot
                          dtype=self.config.int_type)
              for query in queries] for queries in raw_dataset[2]], dtype='O')
        ids = ids[idxes]
        queries_per_user = queries_per_user[idxes]

        return (ids, labels, queries_per_user)

    def _convert_label(self, label):
        if self.true_label2pred_label is None:
            self.true_label2pred_label = {}
        return self.true_label2pred_label.setdefault(label, len(self.true_label2pred_label))

    def _is_sparse_token(self, token):
        if not token in self.token2freq or self.token2freq[token] < self.config.sparse_token_freq:
            return True
        else:
            return False

    def _get_token_index(self, token, for_type='train'):
        if self._is_sparse_token(token):
            return self.token2index['<unk>']
        else:
            return self._get_index(token, self.token2index, for_type)

    def _get_index(self, item, _dic, for_type='train'):
        if item not in _dic:
            if for_type == 'train':
                _dic[item] = len(_dic)
                return len(_dic) - 1
            else:
                raise Exception('Cannot get index of ' + item)
        else:
            return _dic[item]

    @abstractmethod
    def _process_after_mapping_for_train(self, dataset):
        '''Process mapped training dataset

        :return: list
                A new list of dataset.
        '''
        dataset = super(SingleTaskDataset, self)._process_after_mapping_for_valid(dataset)
        # Balance classes
        zip_dataset = zip(*dataset)
        ages = dataset[0]
        age, count = np.unique(ages, return_counts=True)
        age2freq = dict(zip(age, count))
        zip_dataset = self._balance_class(zip_dataset, ages, age2freq, self.config.age_up_sample_k,
                                          self.config.age_min_per)
        unzip_dataset = zip(*zip_dataset)

        genders = unzip_dataset[2]
        gender, count = np.unique(genders, return_counts=True)
        gender2freq = dict(zip(gender, count))
        zip_dataset = self._balance_class(zip_dataset, genders, gender2freq, self.config.gender_up_sample_k,
                                          self.config.gender_min_per)
        unzip_dataset = zip(*zip_dataset)

        edus = unzip_dataset[4]
        edu, count = np.unique(edus, return_counts=True)
        edu2freq = dict(zip(edu, count))
        zip_dataset = self._balance_class(zip_dataset, edus, edu2freq, self.config.edu_up_sample_k,
                                          self.config.edu_min_per)
        unzip_dataset = zip(*zip_dataset)

        ages = unzip_dataset[0]
        age, count = np.unique(ages, return_counts=True)
        self.age2freq = dict(zip(age, count))

        genders = unzip_dataset[2]
        gender, count = np.unique(genders, return_counts=True)
        self.gender2freq = dict(zip(gender, count))

        edus = unzip_dataset[4]
        edu, count = np.unique(edus, return_counts=True)
        self.edu2freq = dict(zip(edu, count))
        dataset = unzip_dataset

        self._train_sample_num = len(dataset[0])
        self._token_num = len(self.token2index)
        return dataset

    def _process_after_mapping_for_valid(self, dataset):
        dataset = super(SingleTaskDataset, self)._process_after_mapping_for_valid(dataset)
        self._valid_sample_num = len(dataset[0])
        return dataset

    def _process_after_mapping_for_test(self, dataset):
        dataset = super(SingleTaskDataset, self)._process_after_mapping_for_test(dataset)
        self._test_sample_num = len(dataset[0])
        return dataset

    def _balance_class(self, dataset, labels, label2freq, up_sample_k, min_per, *args, **kwargs):
        assert len(dataset) == len(labels)
        labels = np.array(labels)
        max_freq = 1. * max(label2freq.values())
        label2sample_num = {}
        for label, freq in label2freq.items():
            label2sample_num[label] = max(int(freq * ((max_freq / freq) ** up_sample_k - 1.)), int(max_freq * min_per))

        full_idxes = np.arange(len(labels))
        add_idxes = []
        for label in label2freq.keys():
            if label2sample_num[label] > 0:
                idxes = full_idxes[labels == label]
                rvs = np.random.randint(low=0, high=len(idxes), size=label2sample_num[label])
                add_idxes += idxes[rvs].tolist()
        added_dataset = [dataset[idx] for idx in add_idxes]
        return dataset + added_dataset

    @abstractmethod
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

    @abstractmethod
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


class UserProfile(object):
    def __init__(self, config,  *args, **kwargs):
        self.config = config
        self.provide_souces = ('age', 'age_mask', 'gender', 'gender_mask', 'edu', 'edu_mask', 'query')
        self.need_mask_sources = {'query': theano.config.floatX}
        self.compare_source = 'query'
        self._initialize()

    @abstractmethod
    def _initialize(self, model_path=None, *args, **kwargs):
        '''
        Initialize dataset information
        '''
        raise NotImplementedError

    @abstractmethod
    def get_parameter_to_save(self):
        '''
        Return parameters that need to be saved with model
        :return: OrderedDict of{name:data,...}
        '''
        raise NotImplementedError

    def get_train_stream(self, raw_dataset, it='shuffled'):
        return self._get_stream(raw_dataset, it, for_type='train')

    def get_valid_stream(self, raw_dataset, it='sequencial'):
        return self._get_stream(raw_dataset, it, for_type='valid')

    def get_test_stream(self, raw_dataset, it='sequencial'):
        return self._get_stream(raw_dataset, it, for_type='test')

    def _get_stream(self, raw_dataset, it='shuffled', for_type='train'):
        raw_dataset = self._update_before_transform(raw_dataset, for_type)
        dataset = self._map(raw_dataset, for_type)

        dataset = self._update_after_transform(dataset, for_type)
        dataset = self._construct_dataset(dataset)
        if it == 'shuffled':
            return self._construct_shuffled_stream(dataset, for_type)
        elif it == 'sequencial':
            return self._construct_sequencial_stream(dataset, for_type)
        else:
            raise ValueError('it should be "shuffled" or "sequencial"!')

    @abstractmethod
    def _update_before_transform(self, raw_dataset, for_type='train'):
        '''
        Do updation beform transform raw_dataset into index representation dataset
        :param raw_dataset:
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :return: a new raw_dataset
        '''
        return raw_dataset

    @abstractmethod
    def _update_after_transform(self, dataset, for_type='train'):
        '''
        Do updation after transform raw_dataset into index representation dataset
        :param for_type: 'train' if the data is used fof training or 'test' for testing
        :param dataset: tranformed dataset
        :return: a new transformed dataset
        '''
        return dataset

    @abstractmethod
    def _map(self, raw_dataset, for_type='train'):
        '''
        Turn raw_dataset into index representation dataset.

        Note: Implement this function in subclass
        '''

        raise NotImplementedError

    def _construct_dataset(self, dataset):
        '''
        Construct an fule indexable dataset.
        Every data corresponds to the name of self.provide_sources
        :param dataset: A tuple of data
        :return:
        '''
        return IndexableDataset(indexables=OrderedDict(zip(self.provide_souces, dataset)))

    def _construct_shuffled_stream(self, dataset, for_type='train'):
        '''
        Construc a shuffled stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel shuffled stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
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
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

    def _construct_sequencial_stream(self, dataset, for_type='train'):
        '''
        Construc a sequencial stream from given dataset
        :param dataset: fuel Indexable dataset
        :return: A fuel sequencial stream with basic transformations:
        1.Sort data by self.compare_source
        2.Batch dataset
        3.Add mask on self.need_mask_sources
        '''
        it = SequentialScheme(dataset.num_examples, self.config.batch_size)
        stream = DataStream(dataset, iteration_scheme=it)
        # Add mask
        for source in self.need_mask_sources.iteritems():
            stream = Padding(stream, mask_sources=[source[0]], mask_dtype=source[1])
        return stream

