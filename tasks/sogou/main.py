import sys

from pml.entrance import TypicalEntrance
from pml.tasks.sogou.config import *
from pml.tasks.sogou.entrance import SogouEntrance
from pml.tasks.sogou.preprocess import *


def preprocess():
    cur_path = os.path.abspath(__file__)
    project_dir = cur_path[0:cur_path.index('source')]
    preprocessor = SogouTokenizer() + SogouSparseTokenFilter(sparse_threshold=5, backup_token='<unk>')
    preprocessor.allow_replace = True

    # Process training dataset
    read_from = os.path.join(project_dir, 'data/debug/train.txt')           # original data path
    save_to = os.path.join(project_dir, 'data/debug/processed_train.txt')   # processed data path
    train_reader_witer = SogouTrainRawDatasetReaderWriter(read_from=read_from, save_to=save_to)
    raw_train_dataset = train_reader_witer.read_dataset()
    processed_train_dataset = preprocessor.apply(raw_train_dataset)
    train_reader_witer.write_dataset(processed_train_dataset)

    # Process prediction dataset
    read_from = os.path.join(project_dir, 'data/debug/test.txt')
    save_to = os.path.join(project_dir, 'data/debug/processed_test.txt')
    predict_reader_writer = SogouPredictRawDatasetReaderWriter(read_from=read_from, save_to=save_to)
    raw_predict_dataset = predict_reader_writer.read_dataset()
    processed_predict_dataset = preprocessor.apply(raw_predict_dataset)
    predict_reader_writer.write_dataset(processed_predict_dataset)


if __name__ == '__main__':
    # You should do pre-processing first if your data have not been processed.
    # Refer above function preprocess for more detail

    config = SogouMultiTaskCharacterConfig()
    entrance = SogouEntrance(config)
    entrance.train()
    print('Train Done!')
    name, ext = os.path.splitext(config.predict_result_save_to)
    for i in range(10):
        predict_result_save_to = '{0}_{1}.{2}'.format(name, i+1, ext)
        entrance.predict(model_load_from=config.model_save_to,
                         result_save_to=predict_result_save_to)
