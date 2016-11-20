import os

from pml.blocks.algorithms import *


class BasicConfig(object):
    '''Basic Config'''
    def __init__(self):
        # Running mode: debug or run
        self.mode = "debug"
    
        #region raw dataset control parameters
        cur_path = os.path.abspath(__file__)
        self.project_dir = cur_path[0:cur_path.index('source\pml')]
    
        # GPU: "int32"; CPU: "int64"
        self.int_type = "int32"
    
        self.batch_size = 32
        self.sort_batch_count = 20
    
        # Step rule
        self.step_rule = AdaDelta()
        
        self.print_freq = 100           # Measured by batches
        self.valid_freq = 0.5           # Measured by epoch
        # Wait wait_times of valid frequency for better valid result
        self.tolerate_times = 20

        self.valid_proportion = 0.2
        self.test_proportion = 0.2

        # Random seed used for random value generation
        self.seed = 1234

    def get_train_dataset_reader_writer(self):
        '''Get dataset reader writer object

        This should provide a dataset_reader_writer object which implements the interfaces of
        pml.dataset.readwrite.AbstractDatasetReaderWriter.

        '''

        raise NotImplementedError

    def get_predict_dataset_reader_writer(self):
        '''Get dataset reader writer object

        This should provide a dataset_reader_writer object which implements the interfaces of
        pml.dataset.readwrite.AbstractDatasetReaderWriter.

        '''
        raise NotImplementedError

    def get_train_preprocessor(self):
        '''Get a piped preprocessor

                This should provide a piped preprocessor object which implement the interfaces of
                pml.dataset.preprocessor.AbstractPreprocessor

                :return: pml.dataset.preprocessor.AbstractPreprocessor
                        Default None
                '''
        # Example
        # preprocessor = Tokenizer(source_name='text', result_source_name='tokenized_text') +
        #                SparseTokenFilter(source_name='tokenized_text',
        #                                   result_source_name='doc_sparse_filtered',
        #                                   sparse_threshold=1,
        #                                   backup_token=None,
        #                                   remove_empty=True)
        # return preprocessor
        #
        return None

    def get_valid_preprocessor(self):
        return self.get_train_preprocessor()

    def get_predict_preprocessor(self):
        return self.get_valid_preprocessor()

    def get_dataset(self):
        '''Get dataset object

        This should provide a dataset object which implements the interfaces of pml.dataset.base.AbstractDataset.

        '''
        raise NotImplementedError

    def get_train_transformer(self):
        '''Get dataset transformer

        Transformer is used to transform the dataset format like padding batch, add noise on input and output,
        sample words and so on.
        '''
        raise NotImplementedError

    def get_valid_transformer(self):
        return self.get_train_transformer()

    def get_predict_transformer(self):
        return self.get_valid_transformer()

    def get_model(self):
        '''Get model object

        This should provide a model object which implements the interfaces of pml.model.AbstractModel.
        The object is constructed with the self.attributions

        '''
        raise NotImplementedError

    def get_model_saver_loader(self):
        '''Get model saver loader object

        This should provide a model_saver_loader object which implements the interfaces of
        pml.model.saveload.AbstractModelSaverLoader.

        '''
        raise NotImplementedError

    @property
    def model_load_from(self):
        '''Get file path to load model
        
        :return: str
        '''
        if not hasattr(self, '_model_load_from'):
            return self.model_save_to
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
            return None
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
            return self.dataset_param_save_to
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
            return None
        return self._dataset_param_save_to

    @dataset_param_save_to.setter
    def dataset_param_save_to(self, value):
        self._dataset_param_save_to = value

    @property
    def train_data_read_from(self):
        '''Get file path to load training dataset
        
        :return: str
        '''
        if not hasattr(self, '_train_data_load_from'):
            return None
        return self._train_data_load_from

    @train_data_read_from.setter
    def train_data_read_from(self, value):
        self._train_data_load_from = value

    @property
    def predict_data_read_from(self):
        '''Get file path to read testing dataset
        
        :return: str
        '''
        if not hasattr(self, '_predict_data_read_from'):
            return None
        return self._predict_data_read_from

    @predict_data_read_from.setter
    def predict_data_read_from(self, value):
        self._predict_data_read_from = value