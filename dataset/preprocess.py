# -*- coding: utf-8 -*-
import cPickle

import numpy as np


class AbstractPreprocessor(object):
    '''Process DatasetContainer by pipeline

    Given an instance of pml.dataset.base.DatasetContainer, the processor does some work on given fields
    of the raw dataset and generates new fields corresponding the processed results.

    By default, with allow_replace set to be False, the generated fields should have unique names,
    otherwise an ValueError indicating name conflict will be raised. With allow_replace set to be
    True, using existing name means replacing values of the field with that name. The later mode is
    usually used for memory saving.

    Define specific processing:
        Implement _process() method to do user specific processing. You are recommended to access data
        fields by names. And you should note that the processing is pipelined, current processor is
        applied on the result of former processors.

    Do processing on dataset:
        Invoke apply() method on instance of pml.dataset.base.DatasetContainer, this will return the processed
        raw dataset with fields added or update.
    '''
    def __init__(self, name=None, preprocessor=None):
        '''Pre-processor for raw dataset

        The pre-processing maybe some common operation on raw dataset like tokenization for
        chinese text, sparse word filtering, stop-word filtering, word stemming, characterization
        and so on.
        :param preprocessor: instance of AbstractPreprocessor
                The wrapped preprocessor which should be applied before current preprocessor.
        :param allow_replace: bool

        '''
        if name is None:
            name = self.__class__
        self.name = name
        self.preprocessor = preprocessor

    def __add__(self, other):
        if not isinstance(other, AbstractPreprocessor):
            raise TypeError('"+" operation is not applied for type {0}'.format(type(other)))
        else:
            other.preprocessor = self
            return other

    @property
    def allow_replace(self):
        '''Indicator if processor can update fields of dataset. If False (default), you can only
           insert new fields into the dataset, otherwise, you can change the values of existing fields.
        '''
        if hasattr(self, '_allow_replace'):
            return self._allow_replace
        else:
            return False

    @allow_replace.setter
    def allow_replace(self, value):
        '''Indicator if processor can update fields of dataset. If False (default), you can only
           insert new fields into the dataset, otherwise, you can change the values of existing fields.
         '''
        if not isinstance(value, bool):
            raise TypeError('allow_replace should be a bool value')
        else:
            self._allow_replace = value
            if self.preprocessor is not None:
                self.preprocessor.allow_replace = value

    @property
    def appended_sources(self):
        if self.preprocessor is not None:
            return self.preprocessor.appended_sources
        else:
            return set()

    def _check_name_conflict(self, sources):
        intersection = self.appended_sources.intersection(sources)
        if len(intersection) != 0:
            error_str = 'Name conflict for:{0}'.format(','.join(map(str, intersection)))
            raise ValueError(error_str)

    def apply(self, raw_dataset, *args, **kwargs):
        '''Do preprocessing on the given dataset

        The processing is applied on the outputs of former preprocessor defined in the constructor
        :param raw_dataset: instance of pml.dataset.base.DatasetContainer

        :return: instance of pml.dataset.base.DatasetContainer
                The preprocessed result is added to the passed DatasetContainer object
        '''
        # Check name conflict
        if not self.allow_replace:
            self._check_name_conflict(raw_dataset.sources)
        if self.preprocessor is not None:
            raw_dataset = self.preprocessor.apply(raw_dataset, *args, **kwargs)
        return self._process(raw_dataset)

    def _process(self, raw_dataset):
        '''Do dataset pre-process

        :return: instance of pml.dataset.base.DatasetContainer
                Pre-processed raw dataset
        '''
        return raw_dataset


class SinglePreprocessor(AbstractPreprocessor):
    def __init__(self, source_name, result_source_name, **kwargs):
        super(SinglePreprocessor, self).__init__(**kwargs)
        self.source_name = source_name
        self.result_source_name = result_source_name

    @property
    def appended_sources(self):
        if self.preprocessor is not None:
            return self.preprocessor.appended_sources.union({self.result_source_name})
        else:
            return {self.result_source_name}

    def _is_name_conflict(self):
        if self.preprocessor is not None and self.result_source_name in self.preprocessor.appended_sources:
            return True
        else:
            return False


class ChineseTokenizer(SinglePreprocessor):
    def __init__(self, **kwargs):
        '''Tokenize chinese text into words

        The tokenized text will be added into the raw dataset.
        :param source_name: str
                The name of the field on which the tokenization applied
        :param result_source_name: str
                The name of tokenized text field. If it not given or set to be None (default),
                source_name+'_tokenized' will be applied.
        '''
        kwargs.setdefault('result_source_name', kwargs['source_name']+'_tokenized')
        super(ChineseTokenizer, self).__init__(**kwargs)
        if self._is_name_conflict():
            raise ValueError('Name conflict! The name {0} for tokenized text has been used!'
                             .format(self.result_source_name))

    def _process(self, raw_dataset):
        '''Tokenize chinese text into tokens with jieba in accurate mode with HMM

        Refer 'https://github.com/fxsjy/jieba' for more detail information about jieba
        :param raw_dataset: dict
                raw_dataset[self.source_name] should store the text field of dataset
        :return: list
                A new list with text in text filed being replaced by tokenized text.
        '''
        import jieba
        texts = raw_dataset[self.source_name]
        tokenized_texts = []
        for text in texts:
            tokenized_texts.append(jieba.lcut(text, cut_all=False, HMM=True))
        raw_dataset[self.result_source_name] = np.array(tokenized_texts)
        return raw_dataset


class ChineseCharacterizer(SinglePreprocessor):
    '''Split chinese text into list of characters

    Continuous digits, english characters are combined and treated as one character.
    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('result_source_name', kwargs['source_name']+'_characterized')
        super(ChineseCharacterizer, self).__init__(**kwargs)
        if self._is_name_conflict():
            raise ValueError('Name conflict! The name {0} for characterized text has been used!'
                             .format(self.result_source_name))

    def _process(self, raw_dataset):
        '''Split chinese text into list of characters


        :param raw_dataset: list or tuple of numpy.ndarray
                raw_dataset[self.source_name] should store the text field of dataset
        :return: list
                A new list with text in text filed being replaced by its composed characters.
        '''
        texts = raw_dataset[self.source_name]
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
        raw_dataset[self.result_source_name] = np.array(text_characters)
        return raw_dataset

    def _is_ascii(self, char):
        if ord(char) < 128:
            return True
        else:
            return False


class SparseTokenFilter(SinglePreprocessor):
    '''Filter out sparse token, e.g., word, character

    Filter out tokens which occur less than given times. If the 'backup_token' attribute given in the constructor,
    the sparse token will be stemmed to the backup token otherwise (default), they will be deleted
    directly.

    For training: extract token2freq informaiton, do filtering, save token2freq information to given file
    For testing: load token2freq information (is training dataset has not been dealt within current runing),
                 do filtering
    '''
    def __init__(self,
                 sparse_threshold,
                 load_from=None,
                 save_to=None,
                 backup_token=None,
                 remove_empty=False,
                 **kwargs):
        '''
        :param source_name: str
                The name of the token field to be filtered, i.e., raw_dataset[source_name]
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
        :param remove_empty: bool
                Indicate if removing empty text after the filtering.
        '''
        kwargs.setdefault('result_source_name', kwargs['source_name']+'_freq_filtered')
        super(SparseTokenFilter, self).__init__(**kwargs)
        if self._is_name_conflict():
            raise ValueError('Name conflict! The name {0} for token text filtered by frequency has been used!'
                             .format(self.result_source_name))
        self.sparse_threshold = sparse_threshold
        self.backup_token = backup_token
        self.load_from = load_from
        self.save_to = save_to
        self.remove_empty = remove_empty
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

    def _process(self, raw_dataset):
        texts = raw_dataset[self.source_name]
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
        for text in texts:
            new_text = []
            for token in text:
                if self._is_sparse_token(token):
                    if self.backup_token is not None:
                        new_text.append(self.backup_token)
                else:
                    new_text.append(token)
            filtered_texts.append(new_text)
        raw_dataset[self.result_source_name] = np.array(filtered_texts)
        if self.remove_empty:
            raw_dataset = self._trim_empty(self.result_source_name, raw_dataset)
        return raw_dataset

    @classmethod
    def _trim_empty(cls, result_source_name, raw_dataset):
        pass
        return raw_dataset

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


class KeywordFilter(SinglePreprocessor):
    '''Filter out none keywords

    Filter out tokens which are not keywords defined in external source. If the 'backup_token' attribute given
    in the constructor, the none keyword tokens will be stemmed to the backup token, otherwise (default),
    they will be deleted directly.

    '''
    def __init__(self,
                 load_from=None,
                 backup_token=None,
                 remove_empty=False,
                 **kwargs):
        '''
        :param source_name: int
                The name of the token field to be filtered, i.e., raw_dataset[source_name]
        :param load_from: str
                File path to load keywords. The keywords should be stored in a set and pickled by cPickle.dump.
                It will be loaded by cPickle.load(open(load_from, 'rb'))
        :param backup_token: str
                Token to replace the sparse token, e.g., '<unk>'
        '''
        kwargs.setdefault('result_source_name', kwargs['source_name']+'_keyword_filtered')
        super(KeywordFilter, self).__init__(**kwargs)
        if self._is_name_conflict():
            raise ValueError('Name conflict! The name {0} for text filtered by keywords has been used!'
                             .format(self.result_source_name))
        self.backup_token = backup_token
        self.load_from = load_from
        self.remove_empty = remove_empty
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

    def _process(self, raw_dataset):
        texts = raw_dataset[self.source_name]
        # Filter out none keyword tokens
        filtered_texts = []
        for text in texts:
            new_text = []
            for token in text:
                if token not in self.keywords:
                    if self.backup_token is not None:
                        new_text.append(self.backup_token)
                else:
                    new_text.append(token)
            filtered_texts.append(new_text)
        raw_dataset[self.result_source_name] = np.array(filtered_texts)
        if self.remove_empty:
            raw_dataset = SparseTokenFilter._trim_empty(self.result_source_name, raw_dataset)
        return raw_dataset

    def _trim_empty(self, raw_dataset):
        idxes = []
        for idx, text in enumerate(raw_dataset[self.result_source_name]):
            if len(text) != 0:
                idxes.append(idx)
        idxes = np.array(idxes)
        if len(idxes) != len(raw_dataset):
            for source in raw_dataset.sources:
                raw_dataset[source] = raw_dataset[source][idxes]
        return raw_dataset

    def _load_keywords(self):
        with open(self.load_from, 'rb') as reader:
            self._keywords = cPickle.load(reader)


def test_pre_process():
    from pml.dataset.base import DatasetContainer
    raw_dataset = DatasetContainer({'doc': np.array(['我在中国good', 'American是个非常好的place我很666',
                                               '中国是个很有意思的地方', '我来自中国', '我在读书', 'hahaha']),
                              'idx': np.array([1, 2, 3, 4, 5, 6])})
    tokenizer = ChineseTokenizer(source_name='doc', result_source_name='doc_tokenized')
    freq_filter = SparseTokenFilter(source_name='doc_tokenized', result_source_name='doc_sparse_filtered',
                                    sparse_threshold=1, backup_token=None, remove_empty=True)
    keyword_filter = KeywordFilter(source_name='doc_sparse_filtered', result_source_name='doc_keyword_filtered',
                                   remove_empty=True)
    keyword_filter.keywords = {u'中国', u'good', u'American'}
    preprocessor = tokenizer + freq_filter + keyword_filter
    preprocessor.allow_replace = True
    raw_dataset = preprocessor.apply(raw_dataset)
    print(raw_dataset.values())


if __name__ == '__main__':

    test_pre_process()