import os

from ..sogou.readwrite import *
from ..sogou.preprocess import *
from ..sogou.dataset import *
from ..sogou.transform import *
from ..sogou.model import *
from ..sogou.saveload import *
from pml.config.base import BasicConfig


class SogouBaseConfig(BasicConfig):
    def __init__(self):
        super(SogouBaseConfig, self).__init__()
        
        # Do validation every 0.5 epoch
        self.valid_freq = 0.5
        # Print training result every 100 batches
        self.print_freq = 5
        self.tolerate_times = 10
        
        # Sample queries and words
        self.query_sample_prob = 0.5
        self.query_sample_num = 50
        self.token_sample_prob = 0.5
        
        # Output noise
        self.age_max_noise = 0.15
        self.age_decay_rate = 2.
        self.gender_max_noise = 0.10
        self.gender_decay_rate = 2.
        self.edu_max_noise = 0.15
        self.edu_decay_rate = 2.
        
        # Class balance
        self.age_up_sample_k = 0.250
        self.gender_up_sample_k = 0.
        self.edu_up_sample_k = 0.250

        # Disk exchange
        self.model_load_from = os.path.join(self.project_dir,
                                       "output/model/none/drop_words_consistently_05_noise_015_010_015_20.pkl")

        self.model_save_to = os.path.join(self.project_dir,
                                     "output/model/multi_task/drop_word_output_noise_with_init/drop_05_noise_015_010_015_20.pkl")

        self.word2vec_load_from = os.path.join(self.project_dir, 'data/word2vec.vec')

        self.train_data_read_from = os.path.join(self.project_dir, "data/debug/train_tok.txt")

        self.predict_data_read_from = os.path.join(self.project_dir, "data/debug/test_tok.txt")

        self.dataset_param_save_to = os.path.join(self.project_dir, 'data/dataset_params.pkl')

        self.predict_result_save_to = os.path.join(self.project_dir, "output/result/drop_word_output_noise_with_init/result.csv")

        # Split dataset
        self.valid_proportion = 0.2
        self.test_proportion = 0.

        # Model dimension
        self.token_embed_dim = 15

        self.query_encode_dim = 15

        self.age_transform_dim = 40

        self.gender_transform_dim = 15

        self.edu_transform_dim = 50
        
        # Regularization
        self.l2_norm_embed = 1e-6   # l2_norm for embedding
        self.l2_norm_other = 1e-6   # l2_norm for other parameters
        self.dropout_embed = 0.6    # dropout probability for embedding
        self.dropout_other = 0.5    # dropout probability for other parameters


class SogouSingleTaskConfig(SogouBaseConfig):
    def __init__(self):
        super(SogouSingleTaskConfig, self).__init__()
        self.task_name = 'age'

    def get_train_dataset_reader_writer(self):
        return SogouTrainDatasetReaderWriter(self)

    def get_predict_dataset_reader_writer(self):
        return SogouPredictDatasetReaderWriter(self)

    def get_dataset(self):
        self.dataset = SogouSingleTaskDataset(self)
        return self.dataset

    def get_train_transformer(self):
        self.train_transformer = SogouSingleTaskTrainTransformer(self)
        return self.train_transformer

    def get_valid_transformer(self):
        return SogouValidTransformer(self)

    def get_predict_transformer(self):
        return SogouPredictTransformer(self)

    def get_model(self):
        self.model = SogouSingleTaskModel(self)
        return self.model
    
    def get_model_saver_loader(self):
        self.model_saver_loader = SogouModelSaveLoader(self)
        return self.model_saver_loader


class SogouMultiTaskConfig(SogouSingleTaskConfig):
    def __init__(self):
        super(SogouMultiTaskConfig, self).__init__()

    def get_train_dataset_reader_writer(self):
        return SogouTrainDatasetReaderWriter(self)
    
    def get_dataset(self):
        self.dataset = SogouMultiTaskDataset(self)
        return self.dataset
    
    def get_train_transformer(self):
        self.train_transformer = SogouMultiTaskTrainTransformer(self)
        return self.train_transformer

    def get_valid_transformer(self):
        return SogouValidTransformer(self)
    
    def get_predict_transformer(self):
        return SogouPredictTransformer(self)
    
    def get_model(self):
        self.model = SogouMultiTaskModel(self)
        return self.model