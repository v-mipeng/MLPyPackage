'''Define classes for saving and loading models.'''

import cPickle
import logging
import os

logger = logging.getLogger('model.extensions.saveload')


class AbstractModelSaverLoader(object):
    '''Save and load model parameters'''
    def __init__(self, load_from, save_to):
        self.load_from = load_from
        self.save_to = save_to

    def save_model(self, save_to=None):
        raise NotImplementedError

    def load_model(self, load_from=None):
        raise NotImplementedError


class FullModelSaverLoader(AbstractModelSaverLoader):
    def __init__(self, model, **kwargs):
        '''
        :param model: blocks.model.Model
        '''
        super(FullModelSaverLoader, self).__init__(**kwargs)
        self.model = model

    def save_model(self, save_to=None):
        if save_to is None:
            save_to = self.save_to
        if not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        with open(save_to, 'wb+') as f:
            logger.info('Saving parameters to %s...' % save_to)
            cPickle.dump(self.model.get_parameter_values(), f) # Get step rule parameters to continue training processing

    def load_model(self, load_from=None):
        '''Load model from disk and initialize the parameters of current model

        Only model parameters are loaded but not those built by step rule, so you are not recommended to
        resume training basing on the load parameters except the you use scale step rule.
        Besides, loaded model parameters and those of given model loaded should completely match both in
        name and value shape
        '''
        if load_from is None:
            load_from = self.load_from
        logger.info('Load model parameters from {0}'.format(load_from))
        if not os.path.exists(load_from):
            raise ValueError('Model file does not exist!')
        else:
            model_params = cPickle.load(open(load_from, 'rb'))
            # Set model parameters.
            self.model.set_parameter_values(model_params)


class ResumeModelSaverLoader(FullModelSaverLoader):
    '''Save parameters of model and those of step rule.

    '''
    def __init__(self, algorithm, **kwargs):
        '''
        :param algorithm: blocks.algorithms.GradientDescent
            To step rule parameters
        '''
        super(ResumeModelSaverLoader, self).__init__(*kwargs)
        self.algorithm = algorithm

    def save_model(self, save_to=None):
        '''Save training processing including parameters of model and step rule.

        This is designed to save model for futural training.
        '''
        if save_to is None:
            save_to = self.save_to
        if not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        with open(save_to, 'wb+') as f:
            logger.info('Saving parameters to %s...' % save_to)

            cPickle.dump(self.model.get_parameter_values(), f)
            # Save step rule parameters in order of model.parameters
            # Note for some step rule like AdaDelta, one model parameter corresponds
            # multiple step rule parameters
            cPickle.dump([update[0].get_value() for update in self.algorithm.step_rule_updates], f)

    def load_model(self, load_from=None):
        '''Load parameters of current model and step rule

        All the parameters of the model and step rule should match exactly
        '''
        if load_from is None:
            load_from = self.load_from
        logger.info('Load model parameters from {0}'.format(load_from))
        if not os.path.exists(load_from):
            raise ValueError('Model file does not exist!')
        else:
            with open(load_from, 'rb') as f:
                model_params = cPickle.load(f)
                step_rule_params = cPickle.load(f)
            # Set model parameters
            self.model.set_parameter_values(model_params)
            # Set step rule parameters
            for update, value in zip(self.algorithm.step_rule_updates, step_rule_params):
                update[0].set_value(value)


class PartialModelSaverLoader(FullModelSaverLoader):
    def __init__(self, initialize_sources, **kwargs):
        '''
        :param initialize_sources: list of str or dict
            For list:
                List of names of model parameters that need to be initialized
                with those of loaded model.
                It will be converted into a dict with key and value being the same
            For dict:
                Name pairs with key being the name of model parameter needed to be initialized and
                value being the name of the loaded model parameter used to initialize that.

        :param kwargs: Parameters for FullModelSaverLoader
        '''
        super(PartialModelSaverLoader, self).__init__(**kwargs)
        if isinstance(initialize_sources, list):
            initialize_sources = dict(zip(initialize_sources, initialize_sources))
        assert isinstance(initialize_sources, dict)
        self.initialize_sources = initialize_sources

    def load_model(self, load_from=None):
        if load_from is None:
            load_from = self.load_from
        logger.info('Load model parameters from {0}'.format(load_from))
        if not os.path.exists(load_from):
            raise ValueError('Model file does not exist!')
        else:
            old_model_params = cPickle.load(open(load_from, 'rb'))
            cur_model_params = self.model.get_parameter_values()
            for old_name, cur_name in self.initialize_sources.iteritems():
                cur_model_params[cur_name] = old_model_params[old_name]
            # Set model parameters.
            self.model.set_parameter_values(cur_model_params)


class PartialResumeModelSaverLoader(ResumeModelSaverLoader):
    def __init__(self, initialize_sources, **kwargs):
        super(PartialResumeModelSaverLoader, self).__init__(**kwargs)
        if isinstance(initialize_sources, list):
            initialize_sources = dict(zip(initialize_sources, initialize_sources))
        assert isinstance(initialize_sources, dict)
        self.initialize_sources = initialize_sources

    def load_model(self, load_from=None):
        if load_from is None:
            load_from = self.load_from
        logger.info('Load model parameters from {0}'.format(load_from))
        if not os.path.exists(load_from):
            raise ValueError('Model file does not exist!')
        else:
            with open(load_from, 'rb') as f:
                old_model_params = cPickle.load(f)
                old_step_rule_params = cPickle.load(f)
            cur_model_params = self.model.get_parameter_values()
            cur_step_rule_params = self.algorithm.step_rule_updates
            span = len(old_step_rule_params) / len(old_model_params)
            old_keys = old_model_params.keys()
            cur_keys = cur_model_params.keys()
            for old_name, cur_name in self.initialize_sources.items():
                cur_model_params[cur_name] = old_model_params[old_name]
                # Update step rule parameters
                old_idx_start = old_keys.index(old_name) * span
                cur_idx_start = cur_keys.index(cur_name) * span
                for old_idx, cur_idx in zip(range(old_idx_start, old_idx_start+span), range(cur_idx_start, cur_idx_start+span)):
                    cur_step_rule_params[cur_idx][0].set_value(old_step_rule_params[old_idx])


class EmbedInitializer(PartialModelSaverLoader):
    def __init__(self, external_token2index, model_token2index,
                 embed_param_name, external_embeds=None,
                 **kwargs):
        '''Initialize embedding of tokens by token name.

        Commonly, user will pre-train a word2vec with large external dataset. The the trained word2vecs
        will be used to initialize those of the training model. In this case, user should indicate the mapping
        relation between word and its vector.
        In the model, token is represented by an encoded integer while in the external word2vecs, token is usually
        represented by its string format or integers in different encode style. In order to initialize correctly
        user has to offer token encoded style in the two different systems.

        :param external_token2index: dict
                Mapping token to the index of external embedding matrix.
        :param model_token2index: dict
                Mapping token to the index of model embedding matrix
        :param embed_param_name: str
                Name of embedding brick in the model. This is used to access to the embedding matrix of the model
        :param external_embeds: 2D np.ndarray of list of np.ndarray
                External embedding matrix
        :param kwargs: parameters for PartialModelSaverLoader
        '''
        super(EmbedInitializer, self).__init__(**kwargs)
        self.external_token2index = external_token2index
        self.model_token2index = model_token2index
        self.embed_param_name = embed_param_name
        self.external_embeds = external_embeds

    def load_model(self, load_from=None):
        if load_from is None:
            load_from = self.load_from
        super(EmbedInitializer, self).load_model(load_from)
        if self.external_embeds is None:
            with open(load_from, 'rb') as f:
                old_model_params = cPickle.load(f)
                logger.info('Load embeddings from {0}'.format(load_from))
                self.external_embeds = old_model_params[self.embed_param_name]
        embed_var = self.model.get_parameter_dict()[self.embed_param_name]
        cur_embeds = embed_var.get_value()
        initialized_num = 0
        for token, cur_idx in self.model_token2index.iteritems():
            old_idx = self.external_token2index.get(token, None)
            if old_idx is not None:
                initialized_num += 1
                cur_embeds[cur_idx] = self.external_embeds[old_idx]
        embed_var.set_value(cur_embeds)
        logger.info('Initialized {0} words among {1} words.'.format(initialized_num, len(cur_embeds)))