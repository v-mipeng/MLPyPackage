'''Read raw dataset from disk

This module should provide source name and raw dataset
'''
import numpy as np


class AbstractReader(object):
    def __init__(self, read_from, sources):
        '''Read raw dataset from file

        :param read_from: str
                File path of the dataset
        :param sources: list of str
                List of names of the features of a sample. For a sample, it may contain
                multiple fields, like for a user may contain age, gender and some other
                fields.
        '''
        self._read_from = read_from
        self._sources = sources
        self._sample_num = 0
        self._raw_dataset = None

    @property
    def read_from(self):
        return self._read_from

    @read_from.setter
    def read_from(self, value):
        assert isinstance(value, str)
        self._read_from = value

    @property
    def sources(self):
        return self._sources

    @sources.setter
    def sources(self, value):
        assert isinstance(value, list) or isinstance(value, tuple)
        self._sources = value

    @property
    def raw_dataset(self):
        if self.raw_dataset is None:
            self.prepare()
        return self._raw_dataset

    @raw_dataset.setter
    def raw_dataset(self, value):
        self._raw_dataset = value

    @property
    def sample_num(self):
        return self._sample_num

    def prepare(self):
        '''Prepare dataset (load dataset from file)

        User can invoke this method explicitly to read data from file with path being
        self.read_from
        '''
        raw_dataset = self._read_dataset()
        if len(raw_dataset) != len(self.sources):
            raise Exception('Source name and value mismatch!')
        if not all([len(field)==raw_dataset[0] for field in raw_dataset]):
            raise Exception('Field dimension mismatch!')
        self._sample_num = len(raw_dataset[0])
        self.raw_dataset = dict(zip(self.sources, raw_dataset))

    def cross_validation(self, portion, shuffled=True, seed=None):
        '''Split dataset into train and validation part

        :param portion: float
                Validation portion
                size(validation) = portion * size(all dataset)
        :param shuffled:
        :param seed: int
                Seed for shuffled the dataset
        :return: tuple
                Training dataset and testing dataset
        '''
        valid_num = int(self.sample_num * portion)
        idxes = np.arange(self.sample_num)
        if shuffled:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(idxes)

        for i in range(self.sample_num / valid_num):
            train_raw_dataset = {}
            valid_raw_dataset = {}
            train_idxes = np.concatenate([idxes[:i * valid_num],
                                         idxes[(i + 1) * valid_num:]])
            valid_idxes = idxes[i*valid_num:(i+1)*valid_num]
            for key, value in self.raw_dataset.items():
                train_raw_dataset[key] = value[train_idxes]
                valid_raw_dataset[key] = value[valid_idxes]
            yield (train_raw_dataset, valid_raw_dataset)

    def _read_dataset(self):
        '''Read dataset from file defined in self.read_from

        :return: list of numpy.ndarray
        '''
        raise NotImplementedError