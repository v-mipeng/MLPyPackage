import cPickle

import numpy as np
import theano

from pml.model.extensions.saveload import AbstractModelSaverLoader, FullModelSaverLoader, PartialModelSaverLoader
from pml.blocks.graph.model import Model


class SogouModelSaveLoader(FullModelSaverLoader):
    def __init__(self, config, **kwargs):
        kwargs.setdefault('load_from', config.model_load_from)
        kwargs.setdefault('save_to', config.model_save_to)
        super(SogouModelSaveLoader, self).__init__(**kwargs)
        self.config = config

    def load_model(self, model=None, load_from=None):
        if model is None:
            model = self.model
        if load_from is None:
            load_from = self.load_from
        try:
            with open(self.load_from, 'rb') as f:
                print('Loading parameters from %s...' % load_from)
                model_params = cPickle.load(f)
            model.set_parameter_values(model_params)
            print('Done!')
        except Exception as e:
            print('Cannot load model from {0} for {1}.'.format(load_from, e.message))
            try:
                print('Initialize with pre-trained word2vec...')
                self._initialize_with_embedding(model)
                print('Initialization done!')
            except Exception as e:
                print('Cannot find pre-trained word2vec!')
                raise e

    def _initialize_with_embedding(self, model):
        word2vec = self._load_pretrain_word2vec(self.config.word2vec_load_from)
        model_params = model.get_parameter_values()
        model_token2index = self.config.dataset.token2index
        token_embed = model_params['/token_embed.W']
        if len(token_embed[0]) != len(word2vec.values()[0]):
            raise ValueError('Embedding dimension mismatch. Model embedding dimension {0} which pre-trained '
                             'embedding dimension {1}'.format(len(token_embed[0]), len(word2vec.values()[0])))
        for token, index in model_token2index.iteritems():
            if token in word2vec:
                token_embed[index] = word2vec[token]
        model.set_parameter_values(model_params)

    def _load_pretrain_word2vec(self, load_from):
        word2vec = {}
        with open(load_from, 'r') as reader:
            for line in reader:
                array = line.strip().split(' ')
                word2vec[array[0]] = np.array(map(float, array[1:]), dtype=theano.config.floatX)
        return word2vec
