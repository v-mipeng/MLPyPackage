#-*- coding: utf-8 -*-

import os
from collections import OrderedDict

import numpy as np

from pml.dataset.preprocess import (ChineseTokenizer, SparseTokenFilter, KeywordFilter, SinglePreprocessor)
from pml.dataset.base import DatasetContainer
from pml.dataset.readwrite import AbstractDatasetReaderWriter
from pml.tasks.sogou.readwrite import SogouTrainRawDatasetReaderWriter, SogouPredictRawDatasetReaderWriter

class SogouDigitFilter(SinglePreprocessor):
    '''Stem digits into 0

    Example: 1996 --> 0, 124345 --> 0

    Operate on query field. It consists of a list of users with each user containing a list of
    queries and each query being a string.

    The format of the result field is the same as the original one.

    Note:
        The modified queries will take place of the original queries with field name 'query'

    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('source_name', 'query')
        kwargs.setdefault('result_source_name', 'query')
        super(SogouDigitFilter, self).__init__(**kwargs)
        self.allow_replace = True

    def _process(self, raw_dataset):
        import re
        queries = raw_dataset[self.source_name]
        regex = re.compile(r'\d+')
        stemed_queries = []
        for queries_per_user in queries:
            stemed_queries_per_user = []
            for query in queries_per_user:
                stemed_query = re.sub(regex, '0', query)
                stemed_queries_per_user.append(stemed_query)
            stemed_queries.append(np.array(stemed_queries_per_user, dtype='O'))
        raw_dataset[self.result_source_name] = np.array(stemed_queries, dtype='O')
        return raw_dataset


class SogouTokenizer(ChineseTokenizer):
    '''Tokenize queries
    Example:  中国是个好地方--> 中国 是 个 好 地方

    Operate on query field. It consists of a list of users with each user containing a list of
    queries and each query being a string.

    The result field consists of a list of users with each user containing a list of queries and
    each query being a list of strings.

    Note:
        The modified queries will take place of the original queries with field name 'query'


    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('source_name', 'query')
        kwargs.setdefault('result_source_name', 'query')
        super(SogouTokenizer, self).__init__(**kwargs)
        self.allow_replace = True

    def _process(self, raw_dataset):
        import jieba
        queries = raw_dataset[self.source_name]
        seg_queries = []
        for queries_per_user in queries:
            seg_queries_per_user = []
            for query in queries_per_user:
                    seg_query = jieba.lcut(query, True)
                    seg_queries_per_user.append(seg_query)
            seg_queries.append(np.array(seg_queries_per_user, dtype='O'))
        raw_dataset[self.result_source_name] = np.array(seg_queries, dtype='O')
        return raw_dataset


class SogouSparseTokenFilter(SparseTokenFilter):
    '''Filter out sparse tokens

    Filter out saprse tokens which occur less than given times (default 20) in training dataset.

    Operate on query field with its input and output format being the same as the result of SogouTokenizer.

    Note:
        By default, the query which is empty after the filtering will be deleted and the sparse token
        will be replaced by a special token '<unk>'
        The modified queries will take place of the original queries with field name 'query'

    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('source_name', 'query')
        kwargs.setdefault('result_source_name', 'query')
        kwargs.setdefault('remove_empty', True)
        kwargs.setdefault('sparse_threshold', 20)
        super(SogouSparseTokenFilter, self).__init__(**kwargs)
        self.allow_replace = True

    def _process(self, raw_dataset):
        queries = raw_dataset[self.source_name]
        # Generate token2freq information
        if self._token2freq is None:
            if self.load_from is not None:
                self._load_token2freq()
            else:
                whole_queries = [np.concatenate(np.array(queries_per_user)).tolist() for queries_per_user in queries]
                tokens = np.concatenate(np.array(whole_queries))
                unique_tokens, counts = np.unique(tokens, return_counts=True)
                self._token2freq = dict(zip(unique_tokens, counts))
                if self.save_to is not None:
                    self._save_token2freq()
        # Filter out sparse token
        filtered_queries = []
        for queries_per_user in queries:
            filtered_queries_per_user = []
            for query in queries_per_user:
                new_query = []
                for token in query:
                    if self._is_sparse_token(token):
                        if self.backup_token is not None:
                            new_query.append(self.backup_token)
                    else:
                        new_query.append(token)
                if len(new_query) > 0:
                    filtered_queries_per_user.append(new_query)
            filtered_queries.append(np.array(filtered_queries_per_user, dtype='O'))
        raw_dataset[self.result_source_name] = np.array(filtered_queries, dtype='O')
        if self.remove_empty:
            raw_dataset = self._trim_empty(self.result_source_name, raw_dataset)
        return raw_dataset


class SogouKeywordFilter(KeywordFilter):
    '''Filter out non keywords

    This filter out non keywords of queries. The keywords are obtained by statistic methods like mutual informaiton
    or machine learning, e.g., selected with logical regression model.

    The source and result formats are the same as that of SogouSparseTokenFilter

    '''
    def __init__(self, **kwargs):
        kwargs.setdefault('source_name', 'query')
        kwargs.setdefault('result_source_name', 'query')
        kwargs.setdefault('remove_empty', True)
        super(SogouKeywordFilter, self).__init__(**kwargs)
        self.allow_replace = True

    def _process(self, raw_dataset):
        queries = raw_dataset[self.source_name]
        filtered_queries = []
        for queries_per_user in queries:
            filtered_queries_per_user = []
            for query in queries_per_user:
                new_query = []
                for token in query:
                    if token not in self.keywords:
                        if self.backup_token is not None:
                            new_query.append(self.backup_token)
                    else:
                        new_query.append(token)
                if len(new_query) > 0:
                    filtered_queries_per_user.append(new_query)
            filtered_queries.append(np.array(filtered_queries_per_user, dtype='O'))
        raw_dataset[self.result_source_name] = np.array(filtered_queries, dtype='O')
        if self.remove_empty:
            raw_dataset = SparseTokenFilter._trim_empty(self.result_source_name, raw_dataset)
        return raw_dataset


class SogouCharacterize(SinglePreprocessor):
    def __init__(self, **kwargs):
        kwargs.setdefault('source_name', 'query')
        kwargs.setdefault('result_source_name', 'query')
        super(SogouCharacterize, self).__init__(**kwargs)
        self.allow_replace = True

    def _process(self, raw_dataset):
        queries = raw_dataset[self.source_name]
        characterized_queries = []
        for queries_per_user in queries:
            characterized_queries_per_user = []
            for query in queries_per_user:
                new_query = []
                buffer = []
                for token in query.decode('utf-8'):
                    if ord(token) < 128:
                        buffer.append(token)
                    else:
                        if len(buffer) > 0:
                            new_query.append(''.join(buffer))
                            buffer = []
                        new_query.append(token)
                if len(buffer) > 0:
                    new_query.append(''.join(buffer))
                characterized_queries_per_user.append(new_query)
            characterized_queries.append(np.array(characterized_queries_per_user, dtype='O'))
        raw_dataset[self.result_source_name] = np.array(characterized_queries, dtype='O')
        return raw_dataset



def test_preprocess():
    raw_dataset = DatasetContainer({'query': np.array([['我在中国good', 'American是个非常好的place我很666'],
                                               ['中国是个很有意思的地方', '我来自中国', '我在读书', 'hahaha']]),
                              'idx': np.array([1, 6])})
    preprocessor = SogouDigitFilter()
    preprocessor += SogouTokenizer()
    preprocessor += SogouSparseTokenFilter(sparse_threshold=1, backup_token=None, remove_empty=True)
    keyword_filter = SogouKeywordFilter(remove_empty=True)
    keyword_filter.keywords = {u'中国', u'good', u'American'}
    preprocessor += keyword_filter
    preprocessor.allow_replace = True
    raw_dataset = preprocessor.apply(raw_dataset)
    for queries in raw_dataset['query']:
        for query in queries:
            print(' '.join(query))


def test_reader_writer():
    cur_path = os.path.abspath(__file__)
    project_dir = cur_path[0:cur_path.index('source\pml')]
    preprocessor = SogouCharacterize() + SogouSparseTokenFilter()
    preprocessor.allow_replace = True

    # Test train dataset reade writer
    read_from = os.path.join(project_dir, 'data/debug/train.txt')
    save_to = os.path.join(project_dir, 'data/debug/processed_train.txt')
    train_reader_witer = SogouTrainRawDatasetReaderWriter(read_from=read_from, save_to=save_to)
    raw_train_dataset = train_reader_witer.read_dataset()
    processed_train_dataset = preprocessor.apply(raw_train_dataset)
    train_reader_witer.write_dataset(processed_train_dataset)
    
    # Test predict dataset reader witer
    read_from = os.path.join(project_dir, 'data/debug/test.txt')
    save_to = os.path.join(project_dir, 'data/debug/processed_test.txt')
    predict_reader_witer = SogouPredictRawDatasetReaderWriter(read_from=read_from, save_to=save_to)
    raw_predict_dataset = predict_reader_witer.read_dataset() 
    processed_predict_dataset = preprocessor.apply(raw_predict_dataset)
    predict_reader_witer.write_dataset(processed_predict_dataset)


if __name__ == '__main__':
    test_reader_writer()


