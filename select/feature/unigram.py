from collections import Iterable
import numpy as np
import math
import warnings
from pml.vectorizer import *
from threading import Thread

class KeywordSelector(object):
    '''
    Select keyword for classification with given measurement:
    MiTrans: information transformation;
            I(p;q) = H(p)-H(p|q) = sum_x_y r(x,y)log_2(r(x,y)/p(x)/q(y))
            where r(x,y) is the joint distribution of random variable x and y
            reference: Pattern Classification P508
    GiniDrop: drop of gini impurity.
            reference: Pattern Classification P322
    '''
    def __init__(self, docs = None, labels = None, *args, **kwargs):
        self.reset(docs, labels)

    def reset(self, docs, labels, *args, **kwargs):
        '''
        Reset this object for new dataset.
        '''
        self._unique_labels = None
        self._mitrans_dist = None
        self._square_dist = None
        self._gini_drop_dist = None
        self._vocabulary = None
        self.docs = docs
        self.labels = labels
        self.thread_num = 10

    @property
    def thread_num(self):
        return self._thread_num

    @thread_num.setter
    def thread_num(self, value):
        self._thread_num = value

    @property
    def docs(self):
        '''
        The dataset
        :return:
        '''
        return self._docs

    @docs.setter
    def docs(self, value):
        if value is not None:
            if not isinstance(value, Iterable) and not isinstance(value, np.array):
                raise Exception('Input value should be iterable or a numpy array!')
            self._docs = np.array(self._padding(value))

    @property
    def labels(self):
        '''
        Category of each document
        :return:
        '''
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is not None:
            if not isinstance(value, Iterable) and not isinstance(value, np.array):
                raise Exception('Input value should be iterable or a numpy array!')
            self._labels = np.array(value)

    @property
    def unique_labels(self):
        if self._unique_labels is None:
            self._unique_labels = set(self.labels)
        return self._unique_labels

    @property
    def vocabulary(self):
        if self._vocabulary is not None:
            return self._vocabulary
        else:
            self._vocabulary = self._vectorizer.word2index.keys()
            return self._vocabulary

    @property
    def mitrans_distance(self):
        if self._mitrans_dist is not None:
            return self._mitrans_dist
        else:
            self._mitrans_dist = self._get_mitrans_distances()
            return self._mitrans_dist

    @property
    def square_distance(self):
        if self._square_distance is not None:
            return self._square_distance
        else:
            self._square_distance = self._get_square_distances()
            return self._square_distance

    @property
    def gini_drop_distance(self):
        if self._gini_drop_dist is not None:
            return self._gini_drop_dist
        else:
            self._gini_drop_dist = self._get_gini_drop_distances()
            return self._gini_drop_dist

    def sort_word_by_distance(self, distance='MiTrans', *args, **kwargs):
        if self.docs is None:
            raise Exception('Dataset is not initialized!')
        if distance == 'MiTrans':
            return self.mitrans_distance
        elif distance == 'GiniDrop':
            return self.gini_drop_distance
        else:
            raise Exception('{0} distance is not supported!'.format(distance))

    def _padding(self, docs, *args, **kwargs):
        print('Padding dataset...')
        self._vectorizer = Vectorizer(docs)
        vectorized_docs = self._vectorizer.get_vectorized_docs()
        # padding with -1
        lens = [len(vectorized_doc) for vectorized_doc in vectorized_docs]
        max_len = max(lens)
        for vectorized_doc, l in zip(vectorized_docs, lens):
            vectorized_doc.extend([-1]*(max_len-l))
        print('Padding done!')
        return vectorized_docs

    def _get_mitrans_distances(self, *args, **kwargs):
        return self._get_distances(self._get_mitrans_distance)

    def _get_square_distances(self, *args, **kwargs):
        return self._get_distances(self._get_square_distance)

    def _get_gini_drop_distances(self, *args, **kwargs):
        _, counts = np.unique(self.labels, return_counts=True)
        N = len(self.labels)
        self.global_gini = 1-(1.*(counts**2)/(N**2)).sum()
        return self._get_distances(self._get_gini_drop_distance)

    def _get_distances(self, func,  *args, **kwargs):
        whole_tuples = [()]*self.thread_num
        whole_idxes = self._vectorizer.index2word.keys()
        ave_len = len(whole_idxes)/self.thread_num
        threads = []
        for i in range(self.thread_num):
            if i == self.thread_num-1:
                idxes = whole_idxes[i*ave_len:]
            else:
                idxes = whole_idxes[i*ave_len:min((i+1)*ave_len, len(whole_idxes))]
            thread = Thread(target=self._get_distances_with_one_thread, args=(func, idxes, whole_tuples, i))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        tuples = []
        for tuple in whole_tuples:
            tuples += tuple
        words, distances = zip(*tuples)
        return self._sort(words, distances)

    def _get_distances_with_one_thread(self, func, idxes, tuples, i):
        words = []
        distances = []
        for idx in idxes:
            feature_values = (self.docs == idx).sum(axis=1) > 0
            words.append(self._vectorizer.index2word[idx])
            distances.append(func(feature_values))
        tuples[i] = zip(words, distances)

    def _get_mitrans_distance(self, feature_values, *args, **kwargs):
        assert len(feature_values) == len(self.labels)
        feature_values = np.array(feature_values)
        unique_feature_values = set(feature_values)
        mitrans_distance = 0.
        N = 1.*len(feature_values)
        for feature_value in unique_feature_values:
            for label in self.unique_labels:
                x = feature_values == feature_value
                y = self.labels == label
                x_and_y = np.logical_and(x, y)
                x_and_y_num = x_and_y.sum()
                if x_and_y_num != 0:
                    mitrans_distance += x_and_y_num/N*math.log(x_and_y_num*N/x.sum()/y.sum())
        return mitrans_distance

    def _get_square_distance(self, feature_values, *args, **kwargs):
        pass
    
    def _get_gini_drop_distance(self, feature_values, *args, **kwargs):
        assert len(feature_values) == len(self.labels)
        feature_values = np.array(feature_values)
        unique_feature_values = set(feature_values)
        gini_drop_distance = 0
        N = 1.*len(feature_values)
        for feature_value in unique_feature_values:
            y_given_x = self.labels[feature_values == feature_value]
            _, counts = np.unique(y_given_x, return_counts=True)
            y_N = len(y_given_x)
            gini_drop_distance += -(1.-(1.*(counts**2)/(y_N**2)).sum())*y_N/N
        return self.global_gini + gini_drop_distance
    
    def _sort(self, words, distances, *args, **kwargs):
        words = np.array(words)
        distances = np.array(distances)
        idxes = np.argsort(distances)[::-1]
        return zip(words[idxes], distances[idxes])


#region Sampling code
# selector = KeywordSelector()
# docs = [['a','b','d'],['b','d','c'],['a','e','b'],['a','c','a'],['b','a','a'],['b','d','d'], ['a','c','d']]
# labels = [1,0,0,1,1,0,0]
# selector.reset(docs, labels)
# print(selector.sort_word_by_distance())
# print(selector.sort_word_by_distance('GiniDrop'))
#endregion