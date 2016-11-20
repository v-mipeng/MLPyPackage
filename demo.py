import os
import sys
from multiprocessing import Process

from pml.config.base import BasicConfig
from pml.entrance import TypicalEntrance


#region Demo usage


def simple_train():
    config = BasicConfig()
    entrance = TypicalEntrance(config)
    # Training
    log = entrance.train()
    best_result = log.status['best_valid_result']
    print('Best result of hyper-parameter group {0} on validation dataset:{1}'.format(best_result))


def simple_test():
    config = BasicConfig()
    entrance = TypicalEntrance(config)
    # Only after tuning hyper-parameters
    entrance.test()


def tune_model():
    '''Tune model parameters automatically'''
    config = BasicConfig()
    entrance = TypicalEntrance(config)
    model_options = {'l2_norm': [1e-5, 1e-4], 'dropout_prob': [0.5, 0.4]}
    results = []
    for i in range(len(model_options[model_options.keys()[0]])):
        for key in model_options.keys():
            setattr(config, key, model_options[key][i])
        # With reset, you avoid reload reconstruct data stream.
        entrance.reset(reset_what='model')
        log = entrance.train()
        best_result = log.status['best_valid_result']
        results.append(best_result)
        print('Best result of hyper-parameter group {0} on validation dataset:{1}'.format(i, best_result))


def tune_dataset_transformation():
    '''Tune hyper-parameters on dataset

    Sometimes you may want to change the hyper-parameters of data stream generation. This is a sample for this purpose.
    '''
    config = BasicConfig()
    entrance = TypicalEntrance(config)
    # Hype-parameters for transformation
    dataset_options = {'query_sample_num': [50, 40, 30], 'word_sample_prob': [0.3, 0.4, 0.5]}
    results = []
    for i in range(len(dataset_options[dataset_options.keys()[0]])):
        for key in dataset_options.keys():
            setattr(entrance.dataset, key, dataset_options[key][i])
        # Note: the dataset is unchanged while the transformation operation is changed by the hyper-parameters
        log = entrance.train()
        best_result = log.status['best_valid_result']
        results.append(best_result)
        print('Best result of hyper-parameter group {0} on validation dataset:{1}'.format(i, best_result))


def tune_model_multi_processing():
    '''Invoke single processing to train model'''
    '''Tune model parameters automatically'''
    model_options = {'l2_norm': [1e-5, 1e-4], 'dropout_prob': [0.5, 0.4]}
    processes = []
    shared_cmd = "THEANO_FLAGS='device={0}' nohup python pml/demo.py {1} > output/log/tune_model_{1}.log &"
    for i in range(len(model_options[model_options.keys()[0]])):
        if i in range(5):
            cmd = shared_cmd.format('gpu0', i + 1)
        elif i in range(5, 10):
            cmd = shared_cmd.format('gpu1', i + 1)
        else:
            cmd = shared_cmd.format('gpu2', i + 1)
        process = Process(target=os.system, args=(cmd,))
        processes.append(process)
    for process in processes:
        process.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    process = int(sys.argv[1])
    config = BasicConfig()
    model_options = {'l2_norm': [1e-5, 1e-4], 'dropout_prob': [0.5, 0.4]}
    for key in model_options.keys():
        setattr(config, key, model_options[key][process])
    # record hyper-parameters with file name
    name, ext = os.path.splitext(config.model_save_to)
    config.model_save_to = '{0}_{1}{2}'.format(name, process, ext)
    entrance = TypicalEntrance(config)
    entrance.train()
    # Training
    log = entrance.train()
    best_result = log.status['best_valid_result']
    print('Best result of hyper-parameter group {0} on validation dataset:{1}'.format(best_result))

#endregion