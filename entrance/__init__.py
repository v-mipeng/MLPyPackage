import logging

from blocks.algorithms import GradientDescent
from blocks.extensions import Printing, ProgressBar
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model

from pml.config.base import BasicConfig
from pml.model.extensions.monitoring import PredictDataMonitor, TestingDataMonitor, EarlyStopMonitor


logger = logging.getLogger('pml.entrance')


'''
Handle parameters to dataset objects
'''


class AbstractEntrance(object):
    def __init__(self, config):
        if not isinstance(config, BasicConfig):
            raise TypeError('config should be object of pml.config.base.BasicConfig!')
        self.config = config
        self._initialize()

    @property
    def raw_train_dataset(self):
        if not hasattr(self, '_raw_train_dataset'):
            return None
        return self._raw_train_dataset

    @raw_train_dataset.setter
    def raw_train_dataset(self, value):
        self._raw_train_dataset = value

    @property
    def raw_valid_dataset(self):
        if not hasattr(self, '_raw_valid_dataset'):
            return None
        return self._raw_valid_dataset

    @raw_valid_dataset.setter
    def raw_valid_dataset(self, value):
        self._raw_valid_dataset = value

    @property
    def raw_test_dataset(self):
        if not hasattr(self, '_raw_test_dataset'):
            return None
        return self._raw_test_dataset

    @raw_test_dataset.setter
    def raw_test_dataset(self, value):
        self._raw_test_dataset = value

    @property
    def raw_predict_dataset(self):
        if not hasattr(self, '_raw_predict_dataset'):
            return None
        return self._raw_predict_dataset

    @raw_predict_dataset.setter
    def raw_predict_dataset(self, value):
        self._raw_predict_dataset = value

    def reset(self, reset_what='model'):
        '''Reset entrance
    
        Sometimes you will change hyper-parameters of model, dataset and so on and want to update them in the entrance,
        for this purpose, you can invoke this method. It will fetch new instance of model, dataset and so on which is
        constructed when fetching.
        This typical usage follows: Change a hyper-parameter of model in config --> get new model object --> train again
    
        :return:
        '''
        if reset_what == 'model':
            self.model = self.config.get_model()
        elif reset_what == 'dataset':
            self.model = self.config.get_dataset()
            self.raw_train_dataset = None
            self.raw_train_dataset = None
            self.raw_test_dataset = None
            self.raw_predict_dataset = None
        elif reset_what == 'model_saver_loader':
            self.model_saver_loader = self.config.get_model_saver_loader()
        elif reset_what == 'dataset_reader_writer':
            self.train_dataset_reader_writer = self.config.get_train_dataset_reader_writer()
            self.predict_dataset_reader_writer = self.config.get_predict_dataset_reader_writer()
        elif reset_what == 'preprocessor':
            self.preprocessor = self.config.get_preprocessor()
        elif reset_what == 'all':
            self._initialize()
        else:
            raise ValueError('{0} is not supported to be reset.'.format(reset_what))

    def _initialize(self):
        self.dataset = self.config.get_dataset()
        self.model = self.config.get_model()
        self.model_saver_loader = self.config.get_model_saver_loader()
        self.train_dataset_reader_writer = self.config.get_train_dataset_reader_writer()
        self.predict_dataset_reader_writer = self.config.get_train_dataset_reader_writer()
        self.preprocessor = self.config.get_preprocessor()

    def train(self):
        '''Training model

        This should start the training process, do validation, save model, save dataset information
        '''
        raise NotImplementedError

    def test(self):
        '''Test model with best tuned parameters on validation dataset

        Note:
        You should only do testing after tuning hyper-parameters on validation dataset.
        It is recommended you restart the program for doing testing. This clumsy step
        is aim to remind you not to do testing when tuning hyper-parameters.

        :return: dict
                Key is the tested variable name and value is its result calculated on testing dataset.
        '''
        raise NotImplementedError

    def predict(self):
        '''Do prediction as a whole

        :return:
        '''
        pass


class TypicalEntrance(AbstractEntrance):
    def __init__(self, *args, **kwargs):
        super(TypicalEntrance, self).__init__(*args, **kwargs)

    def train(self):
        '''Train model

        :return: MainLoop
                The log of training process
        '''
        if self.raw_train_dataset is None or self.raw_valid_dataset is None:  # In case user set training dataset outside the entrance.
            self._prepare_data()
        if self.dataset.train_dataset is None:
            train_stream = self.dataset.get_train_stream(self.raw_train_dataset)
        else:
            train_stream = self.dataset.get_train_stream()
        if self.dataset.valid_dataset is None:
            valid_stream = self.dataset.get_valid_stream(self.raw_valid_dataset)
        else:
            valid_stream = self.dataset.get_valid_stream()
        self.dataset.save(self.config.dataset_param_save_to)
        print("Train on {0} samples".format(self.raw_train_dataset.sample_num))
        # Train model
        logging.info("Training model...")
        main_loop = self._train_model(train_stream, valid_stream)
        logger.info("Training model finished!")
        return main_loop

    def test(self):
        '''Test model with best tuned parameters on validation dataset

        Note:
        You should only do testing after tuning hyper-parameters on validation dataset.
        Once model parameters done, change the config.model_load_from to config.model_save_to.
        This clumsy step is designed to remind you not to do testing when tuning hyper-parameters.

        '''
        if self.raw_test_dataset is None:
            self._prepare_data()
        if not self.dataset.initialized:
            self.dataset.initialize(param_load_from=self.config.dataset_param_load_from)
        test_stream = self.dataset.get_test_stream(self.raw_test_dataset)
        if not self.model.built:
            self.model.build_model()
        self.model_saver_loader.load_model(self.config.model_load_from)
        test_monitor = TestingDataMonitor(variables=self.model.test_monitors, data_stream=test_stream)
        return test_monitor.test()

    def predict(self):
        if self.raw_test_dataset is None:
            self._prepare_data()
        if not self.dataset.initialized:
            self.dataset.initialize(param_load_from=self.config.dataset_param_load_from)
        predict_stream = self.dataset.get_predict_stream(self.raw_predict_dataset)
        if not self.model.built:
            self.model.build_model()
        self.model_saver_loader.load_model(self.config.model_load_from)
        predictor = PredictDataMonitor(variables=self.model.predict_monitors, data_stream=predict_stream)
        predicted_results = predictor.predict()
        for key, value in predicted_results:
            self.raw_predict_dataset[key] = value
        # Save predicted result
        self.predict_dataset_reader_writer.write_dataset(self.raw_predict_dataset)

    def _prepare_data(self):
        '''Define how to split training, validation and testing dataset
        '''
        raw_dataset = self.train_dataset_reader_writer.read_dataset()
        raw_train_dataset, raw_valid_dataset = raw_dataset.split(proportion=self.config.valid_proportion,
                                                                 shuffled=True,
                                                                 seed=self.config.seed)
        raw_train_dataset, raw_test_dataset = raw_train_dataset.split(proportion=self.config.test_proportion,
                                                                      shuffled=False)
        if self.preprocessor is not None:
            self.raw_train_dataset = self.preprocessor.apply(raw_train_dataset)
            self.raw_valid_dataset = self.preprocessor.apply(raw_valid_dataset)
            self.raw_test_dataset = self.preprocessor.apply(raw_test_dataset)
        else:
            self.raw_train_dataset = raw_train_dataset
            self.raw_valid_dataset = raw_valid_dataset
            self.raw_test_dataset = raw_test_dataset

    def _prepare_predict_data(self):
        raw_dataset = self.predict_dataset_reader_writer.read_dataset()
        if self.preprocessor is not None:
            self.raw_predict_dataset = self.preprocessor.apply(raw_dataset)
        else:
            self.raw_predict_dataset = raw_dataset

    def _train_model(self, train_stream, valid_stream):
        if not self.model.built:
            self.model.build_model()

        cg = Model(self.model.train_cg_generator)

        algorithm = GradientDescent(cost=self.model.train_cg_generator,
                                    consider_constant=self.model.consider_constant,
                                    step_rule=self.config.step_rule,
                                    parameters=cg.parameters,
                                    on_unused_sources='ignore')

        # Build extensions
        extensions = self._build_extension(valid_stream)

        main_loop = MainLoop(
            model=cg,
            data_stream=train_stream,
            algorithm=algorithm,
            extensions=extensions
        )
        # Run the model!
        main_loop.run()
        return main_loop

    def _build_extension(self, valid_stream):
        extensions = [
            TrainingDataMonitoring(
                [v for v in self.model.train_monitors],
                prefix='train',
                every_n_batches=self.config.print_freq)
        ]
        n_batches = self.raw_train_dataset.sample_num / self.config.batch_size

        # Initialize model
        try:
            self.model_saver_loader.load_model()
        except:
            print('Cannot initialize model! Train from the beginning!')
        extensions += [EarlyStopMonitor(data_stream=valid_stream,
                                        variables=self.model.valid_monitors,
                                        stop_variable=self.model.valid_monitors[0],
                                        model_saver=self.model_saver_loader,
                                        tolerate_times=self.config.wait_times,
                                        every_n_batches=int(self.config.valid_freq * n_batches))]
        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]
        return extensions
