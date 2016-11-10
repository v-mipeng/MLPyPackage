# -*- coding: utf-8 -*-
import re
import cPickle
from abc import abstractmethod, abstractproperty
from warnings import warn

import numpy as np
import jieba



class AbstractPreprocessor(object):
    '''Do preprocessing on raw dataset

    Sub-class should implement the process() method which does preprocessing on the raw dataset.

    '''
    def __init__(self, preprocessor = None, *args, **kwargs):
        '''Pre-processor for raw dataset

        The pre-processing maybe some common operation on raw dataset like tokenization for
        chinese text, sparse word filtering, stop-word filtering, word stemming, characterization
        and so on.
        :param preprocessor: instance of AbstractPreprocessor
                The wrapped preprocessor which should be applied before current preprocessor.
        '''
        self.preprocessor = preprocessor

    def chattr(self, preprocess_cls, key, value):
        '''Change attribute of current and former pre-processor

        All the stacked pre-processors' attribute with the given name will be changed to the
        given value.
        :param preprocess_cls: preprocess class
                Define which preprocess the attribute changing should be applied.
        :param key: str
                Attribute name
        :param value: obj
                Attribute value
        '''
        if isinstance(self, preprocess_cls) and hasattr(self, key):
            self.key = value
        elif self.preprocessor is not None:
            self.preprocessor.chattr(preprocess_cls, key, value)
        else:
            raise ValueError('No preprocessor has define {0} attribute'.format(key))

    def apply(self, raw_dataset, *args, **kwargs):
        '''Do preprocessing on the given dataset

        The processing is applied on the outputs of former preprocessor defined in the constructor
        :param raw_dataset: list or tuple of numpy.ndarray
                This stores the attributes of training samples. Every item corresponds an attribute and
                the length of each attribute should be the same.
        :return: list or tuple
                Pre-processed raw dataset
        '''
        if self.preprocessor is not None:
            raw_dataset = self.preprocessor.apply(raw_dataset, *args, **kwargs)
        return self.process(raw_dataset, *args, **kwargs)

    def process(self, raw_dataset, *args, **kwargs):
        '''Do dataset pre-process

         :param raw_dataset: list or tuple of numpy.ndarray
                This stores the attributes of training samples. Every item corresponds an attribute and
                the length of each attribute should be the same.
        :return: list or tuple
                Pre-processed raw dataset
        '''
        return raw_dataset


class ChineseTokenizer(AbstractPreprocessor):
    def __init__(self, source_index, *args, **kwargs):
        super(ChineseTokenizer, self).__init__(*args, **kwargs)
        self.source_index = source_index

    def process(self, raw_dataset, *args, **kwargs):
        '''Tokenize chinese text into tokens with jieba in accurate mode with HMM

        Refer 'https://github.com/fxsjy/jieba' for more detail information about jieba
        :param raw_dataset: list or tuple of numpy.ndarray
                raw_dataset[self.text_index] should store the text field of dataset
        :return: list
                A new list with text in text filed being replaced by tokenized text.
        '''
        texts = raw_dataset[self.source_index]
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(jieba.lcut(text, cut_all=False, HMM=True))
        new_raw_dataset = list(raw_dataset)
        new_raw_dataset[self.source_index] = np.array(tokenized_texts)
        return new_raw_dataset


class ChinsesCharacterizer(AbstractPreprocessor):
    '''Split chinese text into list of characters

    Continuous digitals, english characters are combined and treated as one character.
    '''
    def __init__(self, source_index, *args, **kwargs):
        super(ChinsesCharacterizer, self).__init__(*args, **kwargs)
        self.source_index = source_index

    def process(self, raw_dataset, *args, **kwargs):
        '''Split chinese text into list of characters


        :param raw_dataset: list or tuple of numpy.ndarray
                raw_dataset[self.text_index] should store the text field of dataset
        :return: list
                A new list with text in text filed being replaced by its composed characters.
        '''
        texts = raw_dataset[self.source_index]
        text_characters = []
        for text in texts:
            chars = []
            s = []
            for char in text.decode('utf-8'):
                if not self._is_ascii(char):
                    if len(s) > 0:
                        chars.append(''.join(s))
                        s = []
                    chars.append(char)
                else:
                    s.append(char)
            if len(s)>0:
                chars.append(''.join(s))
            text_characters.append(chars)
        new_raw_dataset = list(raw_dataset)
        new_raw_dataset[self.source_index] = np.array(text_characters)
        return new_raw_dataset

    def _is_ascii(self, char):
        if ord(char) < 128:
            return True
        else:
            return False


class SparseTokenFilter(AbstractPreprocessor):
    '''Filter out sparse token, e.g., word, character

    Filter out tokens which occur less than given times. If the 'backup_token' attribute given in the constructor,
    the sparse token will be stemmed to the backup token otherwise (default), they will be deleted
    directly.

    For training: extract token2freq informaiton, do filtering, save token2freq information to given file
    For testing: load token2freq information (is training dataset has not been dealt within current runing),
                 do filtering
    '''
    def __init__(self, source_index,
                 sparse_threshold,
                 load_from = None,
                 save_to = None,
                 backup_token = None,
                 *args, **kwargs):
        '''
        :param source_index: int
                The index of the token field to be filtered, i.e., raw_dataset[source_index]
        :param sparse_threshold: int (>=0)
                Tokens with f(token) <= sparse_threshold are defined as sparse token.
        :param load_from: str
                File path to load token2freq information with cPickle.load. This information is usually extracted from
                training dataset. When dealing with validation or testing dataset, you should usually
                offer this path if you have not dealed with training dataset before on current run.
        :param save_to: str
                File path to save token2freq information with cPickle.dump. When you filter training dataset, you are
                recommended to offer this path to store this information for future usage on validation
                dataset or testing dataset. Otherwise, you can only do the consistent processing on training
                and testing dataset on the run.
        :param backup_token: str
                Token to replace the sparse token, e.g., '<unk>'
        '''
        super(SparseTokenFilter, self).__init__(*args, **kwargs)
        self.source_index = source_index
        self.sparse_threshold = sparse_threshold
        self.backup_token = backup_token
        self.load_from = load_from
        self.save_to = save_to
        self._token2freq = None

    @property
    def token2freq(self):
        '''Get a copy of token2freq informaiton

        :return: dict
                Token frequency dictionary
        '''
        if self._token2freq is not None:
            return self._token2freq.copy()
        else:
            return None

    def process(self, raw_dataset, *args, **kwargs):
        texts = raw_dataset[self.source_index]
        # Generate token2freq information
        if self._token2freq is None:
            if self.load_from is not None:
                self._load_token2freq()
            else:
                tokens = np.concatenate(texts, axis=0)
                unique_tokens, counts = np.unique(tokens, return_counts=True)
                self._token2freq = dict(zip(unique_tokens, counts))
                if self.save_to is not None:
                    self._save_token2freq()
        # Filter out sparse token
        filtered_texts = []
        idxes = []
        for idx, text in enumerate(texts):
            new_text = []
            for token in text:
                if self._is_sparse_token(token):
                    if self.backup_token is not None:
                        new_text.append(self.backup_token)
                else:
                    new_text.append(token)
            if len(new_text) == 0:  # All the tokens of a text are sparse
                continue
            else:
                filtered_texts.append(new_text)
                idxes.append(idx)
        idxes = np.array(idxes)
        for i in range(len(raw_dataset)):
            if i != self.source_index:
                raw_dataset[i] = raw_dataset[i][idxes]
        new_raw_dataset = list(raw_dataset)
        new_raw_dataset[self.source_index] = np.array(filtered_texts)
        return new_raw_dataset

    def _load_token2freq(self):
        with open(self.load_from, 'rb') as reader:
            self._token2freq = cPickle.load(reader)

    def _save_token2freq(self):
        if self._token2freq is not None:
            with open(self.save_to, 'wb+') as writer:
                cPickle.dump(self._token2freq, writer)
        else:
            raise ValueError('Cannot save token2freq for it is none!')

    def _is_sparse_token(self, token):
        if self._token2freq.get(token, -1) <= self.sparse_threshold:
            return True
        else:
            return False


class KeywordFilter(AbstractPreprocessor):
    '''Filter out none keywords

    Filter out tokens which are not keywords defined in external source. If the 'backup_token' attribute given
    in the constructor, the none keyword tokens will be stemmed to the backup token, otherwise (default),
    they will be deleted directly.

    '''

    def __init__(self, source_index,
                 load_from=None,
                 backup_token=None,
                 *args, **kwargs):
        '''
        :param source_index: int
                The index of the token field to be filtered, i.e., raw_dataset[source_index]
        :param load_from: str
                File path to load keywords. The keywords should be stored in a set and pickled by cPickle.dump.
                It will be loaded by cPickle.load(open(load_from, 'rb'))
        :param backup_token: str
                Token to replace the sparse token, e.g., '<unk>'
        '''
        super(KeywordFilter, self).__init__(*args, **kwargs)
        self.source_index = source_index
        self.backup_token = backup_token
        self.load_from = load_from
        self._keywords = None

    @property
    def keywords(self):
        if self._keywords is None:
            if self.load_from is None:
                raise Exception('Keywords are not provided!')
            else:
                self._load_keywords()
        return self._keywords

    @keywords.setter
    def keywords(self, value):
        if isinstance(value, set):
            self._keywords = value
        elif isinstance(value, list) or isinstance(value, tuple):
            self._keywords = set(value)
        else:
            raise ValueError('Keywords should be stored in a list, tuple or set!')

    def process(self, raw_dataset, *args, **kwargs):
        texts = raw_dataset[self.source_index]
        # Filter out none keyword tokens
        idxes = []
        filtered_texts = []
        for idx, text in enumerate(texts):
            new_text = []
            for token in text:
                if token not in self.keywords:
                    if self.backup_token is not None:
                        new_text.append(self.backup_token)
                else:
                    new_text.append(token)
            if len(new_text) == 0:
                continue
            else:
                idxes.append(idx)
                filtered_texts.append(new_text)
        idxes = np.array(idxes)
        for i in range(len(raw_dataset)):
            if i != self.source_index:
                raw_dataset[i] = raw_dataset[i][idxes]
        new_raw_dataset = list(raw_dataset)
        new_raw_dataset[self.source_index] = np.array(filtered_texts)
        return new_raw_dataset

    def _load_keywords(self):
        with open(self.load_from, 'rb') as reader:
            self._keywords = cPickle.load(reader)


if __name__ == '__main__':
    characterizer = ChinsesCharacterizer(0)
    raw_dataset = [np.array(['我在中国good', 'American是个非常好的place我很666',
                            '中国是个很有意思的地方','我来自中国','我在读书','hahaha']), np.array([1,2,3,4,5,6])]
    # raw_dataset = characterizer.apply(raw_dataset)
    tokenizer = ChineseTokenizer(0)
    filter = SparseTokenFilter(0, 1, backup_token=None, preprocessor=tokenizer)
    # filter = KeywordFilter(0, preprocessor=tokenizer)
    # filter.keywords = {u'中国',u'good',u'American'}
    raw_dataset = filter.apply(raw_dataset)
    print(raw_dataset)