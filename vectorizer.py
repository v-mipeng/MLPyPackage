import math
import numpy as np

class TFIDF(object):
    def __init__(self, docs):
        self.reset(docs)

    def reset(self, docs):
        '''
        Reset this object for new dataset.

        '''
        self.docs = docs
        self.idf = None
        self.tfidf_docs = None

    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            raise Exception('Input value should be list or tuple!')
        self._docs = value

    def get_tfidf_doc(self, doc, *args, **kwargs):
        '''
        Get doc's tfidf based on self.docs
        For a given item w, its tfidf is:
        tfidf(w) = (0.5+0.5(tf(w)/max_w(tf(w)))*log(N/df(w))
        :param doc: an iterable document
        :param args:
        :param kwargs:
        :return: a dict mapping item of doc to its tfidf value

        '''
        if self.idf is None:
            self._get_idf()
        item2tf = self._get_tf(doc)
        for key in item2tf.keys():
            if key in self.idf:
                item2tf[key] *= self.idf[key]
        return item2tf

    def get_tfidf_docs(self, *args, **kwargs):
        '''
        Get docs' tfidf representation
        :param args:
        :param kwargs:
        :return: a list of dicts, each dict is the tfidf representation of a doc
        '''
        if self.tfidf_docs is None:
            self.tfidf_docs = []
            for doc in self.docs:
                self.tfidf_docs.append(self.get_tfidf_doc(doc))
        return self.tfidf_docs

    def _get_idf(self, *args, **kwargs):
        '''
        Get idf of each item in docs
        :param docs: two dimensional iterable list or tuple
        :param args:
        :param kwargs:
        :return: a dic mapping each unique item to its idf value
        '''
        if self.docs is None:
            raise Exception('docs is empty, cannot get idf!')
        N = len(self.docs)*1.
        item2freq = {}
        for doc in self.docs:
            words = set(doc)
            for word in words:
                item2freq[word] = item2freq.get(word, 0) + 1
        for key in item2freq.keys():
            item2freq[key] = math.log(N/item2freq[key])
        self.idf = item2freq

    def _get_tf(self, doc, *args, **kwargs):
        '''
        Get tf of each item in a doc
        :param docs: one dimensional iterable list or tuple
        :param args:
        :param kwargs:
        :return: a dic mapping each item to its tf value
        '''
        if not isinstance(doc, list) and not isinstance(doc, tuple):
            raise Exception('Input docs should be list or tuple!')
        item, count = np.unique(np.array(doc), return_counts=True)
        count = 0.5+0.5*count/count.max()
        return dict(zip(item, count))


class Vectorizer(object):
    '''
    Vectorize dataset
    '''
    def __init__(self, docs):
        self.reset(docs)

    def reset(self, docs):
        '''
        Reset this object for new dataset.

        '''
        self.docs = docs
        self._word2index = None
        self._index2word = None
        self.vector_docs = None

    @property
    def docs(self):
        return self._docs

    @docs.setter
    def docs(self, value):
        if not isinstance(value, list) and not isinstance(value, tuple):
            raise Exception('Input value should be list or tuple!')
        self._docs = value

    @property
    def word2index(self):
        if self._word2index is None:
            self._get_word2index()
        return self._word2index

    @property
    def index2word(self):
        if self._index2word is not None:
            return self._index2word
        else:
            if self.word2index is None:
                return None
            else:
                words, idxes = zip(*self.word2index.items())
                self._index2word = dict(zip(idxes, words))
                return self._index2word

    def get_vectorized_doc(self, doc, *args, **kwargs):
        '''
        Get vectorized doc based on self.docs
        Words that not occur in self.docs are deleted
        :param doc: an iterable document
        :param args:
        :param kwargs:
        :return: a list of integers. Each integer corresponds a word in self.docs
        '''
        if self.docs is None:
            raise Exception('Dataset has not been initialized!')
        return [self.word2index[word] for word in doc if word in self.word2index]

    def get_vectorized_docs(self, *args, **kwargs):
        '''
        Get vectorized docs of self.docs
        :param args:
        :param kwargs:
        :return: two dimensional list. Each list represents a document
                and each item of the list represents an integer representation of word.
        '''
        if self.docs is None:
            raise Exception('Dataset has not been initialized!')
        if self.vector_docs is None:
            self.vector_docs = []
            for doc in self.docs:
                self.vector_docs.append(self.get_vectorized_doc(doc))
        return self.vector_docs

    def _get_word2index(self, *args, **kwargs):
        '''
        Get idf of each item in docs
        :param docs: two dimensional iterable list or tuple
        :param args:
        :param kwargs:
        :return: a dic mapping each unique item to its idf value
        '''
        if self.docs is None:
            raise Exception('docs is empty, cannot get idf!')
        self._word2index = {}
        for doc in self.docs:
            words = set(doc)
            for word in words:
                if word not in self.word2index:
                    self.word2index[word] = len(self.word2index)


#region Sample code

train_docs = []
vectorizer = Vectorizer(train_docs)
train_vector_docs = vectorizer.get_vectorized_docs()
tfidfer = TFIDF(train_vector_docs)
train_tfidf_docs = tfidfer.get_tfidf_docs()
test_docs = []
test_vector_docs = [vectorizer.get_vectorized_doc(doc) for doc in test_docs]
test_tfidf_docs = [tfidfer.get_tfidf_doc(vector_doc) for vector_doc in test_vector_docs]

#endregion