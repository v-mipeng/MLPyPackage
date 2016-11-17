from abc import abstractproperty

from pml.config.base import BasicConfig
from ..base import AbstractDataset


class AbstractTaskDataset(AbstractDataset):
    '''Convert preprocessed raw dataset into stream and extract necessary dataset information

    Initialize dataset:
        1. from file: invoke Initialize() method given file path to load dataset information
        2. pass training dataset first: invoke get_train_datastream() method given training
           raw dataset

        Note: self.initialized is an indicator of weather the dataset have been initialized.
        The above two methods will set it to be True if initializing successfully.
        You cannot process validation or testing dataset without initialization. The

    Save dataset:
        Invoke save() method given file path to save.
        This method will save all the attributes of current instance into given file with cPickle.dump.

    Reset dataset:
        By default, some dataset information will not be erased even you deal with another training dataset
        by invoking get_train_stream() method again. To make sure you can extract exactly new dataset information,
        you can make a new instance or invoke reset() method given attribute names you want to preserve. This will
        erase all the attributes of current instance except those you want to preserve.

    Process dataset:
        It offer get_train_stream(), get_valid_stream() and get_test_stream() methods for dealing with training,
        validation and testing dataset. You should define the sources provided by training or validation stream in
        self.provide_train_sources and those by testing stream in self.provide_test_sources.
        Note: you should make sure the dimension of each field of the raw dataset to be the same, otherwise an
        error will raise.

    What subclass should implement:

        1. Implement _map[_for_<train|valid|test>] methods to convert feature representation. Do common work for
        training, valid and testing dataset in _map method and do specific work in corresponding methods
        2. If you want to do some work before converting feature representation,
        invoke: _process_before_mapping[_for_<train|valid|test>]. All the type of dataset will be processed by
        _process_before_mapping() method
        3. If you want to do some work after converting feature representation,
        invoke: _process_after_mapping[_for_<train|valid|test>]. All the type of dataset will be processed by
        _process_after_mapping() method
        4. Add some transformation to the stream in _construct_shuffled_stream method and _construct_sequential_stream
        methods

    '''
    def __init__(self, config=None):
        '''Build training, validation and testing streams

        :param config: pml.config.base.BasicConfig
        '''
        super(AbstractTaskDataset, self).__init__(self.config.__dict__.get('name', None))
        self.initialized = False
        if config is not None and not isinstance(config, BasicConfig):
            raise TypeError('config should be object of pml.config.base.BasicConfig!')
        self.config = config

    @property
    def config(self):
        if self._config is not None:
            return self._config.copy()
        else:
            return None

    @config.setter
    def config(self, value):
        self._config = value


class AbstractTaskClassificationDataset(AbstractTaskDataset):

    @abstractproperty
    def label_num(self):
        '''Define the number of labels for classification

        :return: int
            The number of labels. This is typical for classification problem.
        '''
        return len(self.true_label2pred_label)

    @property
    def true_label2pred_label(self):
        '''Get the mapping from predicted label to true label

        Sometimes, the true labels are in symbolic representation or not start from zero. For classification
        task, these labels should be mapped into integers starting with 0. User can manually define the mapping
        rules here. Make sure your mapped label should start with 0 and be continuous.

        :return: dict
            A dictionary mapping the true label to predicted label
        '''
        if hasattr(self, '_true_label2pred_label'):
            return self._true_label2pred_label
        else:
            return None

    @true_label2pred_label.setter
    def true_label2pred_label(self, value):
        self._true_label2pred_label = value

    @property
    def pred_label2true_label(self):
        '''Get the mapping from predicted label to true label

        Sometimes, the true labels are in symbolic representation or not start from zero. For classification
        task, these labels should be mapped into integers starting with 0. In order to make the prediction
        of system easy to understand, it should map the predicted labels back to the true labels. This provide
        the mapping relationship from predicted label to true label.

        :return: dict
            A dictionary mapping the predicted label to true label
        '''
        if hasattr(self, '_pred_label2true_label'):
            return self._pred_label2true_label
        elif self.true_label2pred_label is not None:
            key, value = self.true_label2pred_label.items()
            self._pred_label2true_label = dict(zip(value, key))
            return self._pred_label2true_label.copy()
        else:
            return None


class AbstractTaskDocClassificationDataset(AbstractTaskClassificationDataset):

    @abstractproperty
    def token_num(self):
        '''Define token number, usually used as feature dimension

        :return: int
            The number of tokens (word number, character number and so on)
        '''
        raise NotImplementedError
