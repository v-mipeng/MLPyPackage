import os

from blocks.algorithms import BasicMomentum, AdaDelta, AdaGrad, RMSProp, Adam, CompositeRule, StepClipping, Momentum, Scale


class BasicConfig:
    '''Basic Config'''
    def __init__(self):
        # Running mode: debug or run
        self.mode = "debug"
    
        #region raw dataset control parameters
        cur_path = os.path.abspath(__file__)
        self.project_dir = cur_path[0:cur_path.index('source/pml')]
    
        # GPU: "int32"; CPU: "int64"
        self.int_type = "int32"
    
        self.batch_size = 32
        self.sort_batch_count = 20
    
        # Step rule
        self.step_rule = AdaDelta()
        
        self.print_freq = 100           # Measured by batches
        self.valid_freq = 0.5           # Measured by epoch
        # Wait wait_times of valid frequency for better valid result
        self.wait_times = 20

        self.valid_proportion = 0.2

    @property
    def Model(self):
        '''Get model class

        :return: Instance of subclass: pml.model.AbstractModel
        '''
        if not hasattr(self, '_Model'):
            raise NotImplementedError
        return self._Model
    
    @Model.setter
    def Model(self, value):
        self._Model = value

    @property
    def Dataset(self):
        '''Get dataset class

        :return: instance of subclass:AbstractDataset
        '''
        if not hasattr(self, '_Dataset'):
            raise NotImplementedError
        return self._Dataset

    @Dataset.setter
    def Dataset(self, value):
        self._Dataset = value

    @property
    def ModelSaverLoader(self):
        '''Get model saver loader class

        :return: instance of subclass: pml.model.AbstractModelSaverLoader
        '''
        if not hasattr(self, '_ModelSaverLoader'):
            raise NotImplementedError
        return self._ModelSaverLoader

    @ModelSaverLoader.setter
    def ModelSaverLoader(self, value):
        self._ModelSaverLoader = value

    @property
    def model_load_from(self):
        '''Get file path to load model
        
        :return: str
        '''
        if not hasattr(self, '_model_load_from'):
            raise NotImplementedError
        return self._model_load_from
    
    @model_load_from.setter
    def model_load_from(self, value):
        self._model_load_from = value

    @property
    def model_save_to(self):
        '''Get file path to save model
        
        :return: str
        '''
        if not hasattr(self, '_model_save_to'):
            raise NotImplementedError
        return self._model_save_to

    @model_save_to.setter
    def model_save_to(self, value):
        self._model_save_to = value

    @property
    def dataset_param_load_from(self):
        '''Get file path to load dataset parameters
        
        :return: str
        '''
        if not hasattr(self, '_dataset_param_load_from'):
            raise NotImplementedError
        return self._dataset_param_load_from

    @dataset_param_load_from.setter
    def dataset_param_load_from(self, value):
        self._dataset_param_load_from = value

    @property
    def dataset_param_save_to(self):
        '''Get file path to save dataset parameters
        
        :return: str
        '''
        if not hasattr(self, '_dataset_param_save_to'):
            raise NotImplementedError
        return self._dataset_param_save_to

    @dataset_param_save_to.setter
    def dataset_param_save_to(self, value):
        self._dataset_param_save_to = value

    @property
    def train_data_load_from(self):
        '''Get file path to load training dataset
        
        :return: str
        '''
        if not hasattr(self, '_train_data_load_from'):
            raise NotImplementedError
        return self._train_data_load_from

    @train_data_load_from.setter
    def train_data_load_from(self, value):
        self._train_data_load_from = value

    @property
    def test_data_load_from(self):
        '''Get file path to load testing dataset
        
        :return: str
        '''
        if not hasattr(self, '_test_data_load_from'):
            raise NotImplementedError
        return self._test_data_load_from

    @test_data_load_from.setter
    def test_data_load_from(self, value):
        self._test_data_load_from = value