import numpy as np
import numpy.linalg as la

class EmbeddingVisual(object):
    '''
    Similarity is measured in Euclid space
    '''
    def __init__(self, embeddings, *args, **kwargs):
        self._initialize(embeddings)

    def _initialize(self, embeddings, *args, **kwargs):
        self.key2index = {}
        keys = []
        embeds = []
        for key, vec in embeddings.iteritems():
            self.key2index[key] = len(self.key2index)
            keys.append(key)
            embeds.append(vec)
        self.embeddings = np.array(embeds)
        self.keys = np.array(keys, 'O')

    def get_norm(self, key, norm_type='l2_norm'):
        '''

        :param key:
        :param norm_type: l2_norm; max_norm: return max value of the embedding vector
        :return:
        '''
        if norm_type == 'l2_norm':
            return la.norm(self.embeddings[self.key2index[key]], 2)
        elif norm_type == 'max_norm':
            return self.embeddings[self.key2index[key]].max()
        else:
            raise Exception('Given norm type is not supported!')

    def get_embedding(self, key, *args, **kwargs):
        return self.embeddings[self.key2index[key]]

    def get_similar_by_key(self, key, topn=10, *args, **kwargs):
        '''
        Get keys whose embeddings are closest to that of the given key
        :param key:
        :param topn:
        :param args:
        :param kwargs:
        :return:
        '''
        orignal_vec = self.embeddings[self.key2index[key]]
        return self.get_similar_by_vec(orignal_vec)

    def get_similar_by_vec(self, vec, topn = 10, *args, **kwargs):
        vec = np.array(vec)
        sims = ((self.embeddings - vec)**2).sum(axis=1)
        orders = np.argsort(sims)
        keys = self.keys[orders[0:topn]]
        values = sims[orders[0:topn]]
        return zip(keys, values)

    def get_distance(self, key_one, key_two, *args, **kwargs):
        '''
        Get distance of the embedding of give two keys in Euclid Space
        :param key_one:
        :param key_two:
        :param args:
        :param kwargs:
        :return:
        '''
        vec_one = self.embeddings[self.key2index[key_one]]
        vec_two = self.embeddings[self.key2index[key_two]]
        distance = np.sqrt(((vec_one-vec_two)**2).sum())
        return distance


