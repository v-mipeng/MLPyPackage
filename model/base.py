from abc import abstractmethod

import theano
from theano import tensor


class AbstractModel(object):
    def __init__(self, name=None):
        '''Build model

        This class should provide computing graph generator for training, validation and testing. Besides,
        it should also offer some monitors for the three processes
        :param config: pml.config.base.BasicConfig
                System configuration object
        :param dataset: pml.dataset.base.AbstractDataset
                The dataset providing training, validation and testing streams
        '''
        if name is None:
            name = self.__class__
        self.name = name
        self._initialized = False
        self._consider_constant = []

    @property
    def initialized(self):
        if self._initialized:
            return True
        else:
            return False

    def build_model(self):
        self._define_inputs()
        self._build_bricks()
        self._get_cost()
        self._initialized = True

    @property
    def train_cg_generator(self):
        '''Get computing graph generator for training

        :return: TheanoVariable.scalar
                Usually it is the training cost
        '''
        if hasattr(self, '_train_cg_generator'):
            return self._train_cg_generator
        else:
            raise NotImplementedError('train_cg_generator is not defined!')

    @property
    def valid_cg_generator(self):
        '''In case you want to do validation independently
        '''
        if hasattr(self, '_valid_cg_generator'):
            return self._valid_cg_generator
        else:
            return self.train_cg_generator

    @property
    def test_cg_generator(self):
        
        if hasattr(self, '_test_cg_generator'):
            return self._test_cg_generator
        else:
            return self.valid_cg_generator
    
    @property
    def predict_cg_generator(self):
        '''Get computing graph generator for testing
        
        Usually you should use this generator to build a computing graph and initialize it with trained
        parameters. Then you use predict monitors to build prediction functions.
        :return: TheanoVariable.scalar
        '''
        if hasattr(self, '_predict_cg_generator'):
            return self._predict_cg_generator
        else:
            return self.train_cg_generator
    
    @property
    def consider_constant(self):
        '''Get parameters that should not be updated.

        :return: list
                List of parameters
        '''
        if hasattr(self, '_consider_constant'):
            return self._consider_constant
        else:
            return []

    @property
    def train_monitors(self):
        '''Get variables for monitoring training process

        :return: List of ~class: TheanoVariable.scalar
            Usually the cost and accuracy
        '''
        if hasattr(self, '_train_monitors'):
            return self._train_monitors
        else:
            raise NotImplementedError('Train monitors are not defined!')

    @property
    def valid_monitors(self):
        '''Get symbol variables to build validation functions to valid trained model

        :return: List of class: TheanoVariable.scalar
            Usually, the cost and accuracy
        '''
        if hasattr(self, '_valid_monitors'):
            return self._valid_monitors
        else:
            raise NotImplementedError('valid_monitors are not defined!')

    @property
    def test_monitors(self):
        '''Get symbolic variables to build functions for prediction on testing dataset

        :return: List of class: TheanoVariable.scalar
                Commonly, the most likely prediction of given input
        '''
        if hasattr(self, '_test_monitors'):
            return self._test_monitors
        else:
            raise NotImplementedError('test_monitors are not defined!')

    @property
    def predict_monitors(self):
        '''Get symbolic variables to build functions for prediction on predicting dataset

        :return: List of class: TheanoVariable.scalar
                Commonly, the most likely prediction of given input
        '''
        if hasattr(self, '_predict_monitors'):
            return self._predict_monitors
        else:
            raise NotImplementedError('predict_monitors are not defined!')

    @abstractmethod
    def _define_inputs(self, *args, **kwargs):
        '''Define the input of this model'''
        raise NotImplementedError('_define_inputs is not defined!')

    @abstractmethod
    def _build_bricks(self, *args, **kwargs):
        '''Define bricks of this model'''
        raise NotImplementedError('_build_bricks is not defined')

    @abstractmethod
    def _get_train_cost(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _get_valid_cost(self, *args, **kwargs):
        pass

    def _get_test_cost(self, *args, **kwargs):
        self._get_valid_cost(*args, **kwargs)

    @abstractmethod
    def _get_predict_cost(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _apply_reg(self):
        '''Add regularization, e.g. l2 norm, on parameters

        This should add regularization on system lost and update training computing graph
        '''
        pass

    @abstractmethod
    def _apply_noise(self):
        '''Add noise on parameters like dropout'''
        pass

    def _get_cost(self):
        self._get_train_cost()
        self._get_valid_cost()
        self._get_test_cost()
        self._apply_reg()
        self._apply_noise()

    def log_probabilities(self, input_):
        """Normalize log-probabilities.

        Converts unnormalized log-probabilities (exponents of which do not
        sum to one) into actual log-probabilities (exponents of which sum
        to one).

        Parameters
        ----------
        input_ : :class:`~theano.Variable`
            A matrix, each row contains unnormalized log-probabilities of a
            distribution.

        Returns
        -------
        output : :class:`~theano.Variable`
            A matrix with normalized log-probabilities in each row for each
            distribution from `input_`.

        """
        shifted = input_ - input_.max(axis=1, keepdims=True)
        return shifted - tensor.log(
            tensor.exp(shifted).sum(axis=1, keepdims=True))

    def categorical_cross_entropy(self, y, x):
        """Computationally stable cross-entropy for pre-softmax values.

        Parameters
        ----------
        y : :class:`~tensor.TensorVariable`
            In the case of a matrix argument, each row represents a
            probabilility distribution. In the vector case, each element
            represents a distribution by specifying the position of 1 in a
            1-hot vector.
        x : :class:`~tensor.TensorVariable`
            A matrix, each row contains unnormalized probabilities of a
            distribution.

        Returns
        -------
        cost : :class:`~tensor.TensorVariable`
            A vector of cross-entropies between respective distributions
            from y and x.

        """
        x = self.log_probabilities(x)
        if y.ndim == x.ndim - 1:
            indices = tensor.arange(y.shape[0]) * x.shape[1] + y
            cost = -x.flatten()[indices]
        elif y.ndim == x.ndim:
            cost = -(x * y).sum(axis=1)
        else:
            raise TypeError('rank mismatch between x and y')
        return cost


class AbstractEmbedModel(AbstractModel):
    '''Abstract model which used embedding.

    This is designed to only calculate gradients on input related embeddings and

    '''
    pass
