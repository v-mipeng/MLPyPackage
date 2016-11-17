import logging

import numpy
import theano
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.model import Model
from blocks.monitoring.evaluators import DatasetEvaluator


logger = logging.getLogger('model.extensions.monitors')


class ValidationDataMonitor(DataStreamMonitoring):
    pass


class TestingDataMonitor(object):
    """Monitors Theano variables on test data stream.

        It is recommended to monitor on testing dataset only after hyper-parameters tuned.

        Parameters
        ----------
        variables : list of :class:`~tensor.TensorVariable` and
            :class:`MonitoredQuantity`
            The variables to monitor. The variable names are used as record
            names in the logs.
        data_stream : instance of :class:`.DataStream`
            The data stream to monitor on. A data epoch is requested
            each time monitoring is done.

        """
    def __init__(self, variables, data_stream):
        self._evaluator = DatasetEvaluator(variables)
        self.data_stream = data_stream

    def test(self):
        """Test on testing dataset and print the results.

        :return value_dict: dict
                A dictionary mapping variable names to its aggregated values.
        """
        logger.info("Monitoring on testing data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        for key, value in value_dict.iteritems():
            print('test_{0}:{1}'.format(key, value))
        logger.info("Monitoring on testing data finished")
        return value_dict


class PredictDataMonitor(object):
    def __init__(self, variables, data_stream):
        self.variables = variables
        self.data_stream = data_stream

    def _complie(self):
        self.inputs = []
        for variable in self.variables:
            cg = Model(variable)
            inputs = cg.inputs
            self.inputs.append(inputs)
        self.func = theano.function(self.inputs, self.variables, on_unused_input='ignore')

    def predict(self):
        all_predictions = []
        for batch in self.data_stream.get_epoch_iterator():
            input_batch = (batch[self.data_stream.sources.index(input.name)] for input in self.inputs)
            predictions = self.func(input_batch)
            all_predictions.append([prediction.tolist() for prediction in predictions])
        outputs = zip(*all_predictions)
        return dict(zip([variable.name for variable in self.variables], outputs))

class EarlyStopMonitor(ValidationDataMonitor):
    '''Do early stop with specific measurement on validation dataset


    :param stop_variable: theano.TensorVariable
            Do early stop with this variable on validation dataset
    :param model_saver: pml.model.AbstractModelSaverLoader
            Invoke save_model() method of it to save model parameters with which
            current best result obtained on validation dataset
    :param tolerate_time: int
            Validation times before new best result obtained. If no better result obtained
            after tolerate_time times validation, stop training.

    '''
    def __init__(self, stop_variable, model_saver, tolerate_time=20, **kwargs):
        variables = kwargs.get('variables', [])
        names = [variable.name for variable in variables]
        if stop_variable.name not in names:
            variables.append(stop_variable)
        kwargs['variables'] = variables
        super(EarlyStopMonitor, self).__init__(**kwargs)
        self.stop_variable = stop_variable
        self.model_saver = model_saver
        self.tolerate_time = tolerate_time
        self._best_result = -numpy.inf
        self.wait_time = 0
        self.tolerate_time = tolerate_time

    @property
    def best_result(self):
        return self.best_result

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log and do early stop

        Compare current validation result (the larger the better) with former best result one, if current
        result is of new best, save the model parameters into file and update the best
        result to the current one and reset the wait time to be zero. Otherwise, increment
        wait time by one and if it is over tolerate time, stop training process.

        Note:
            Only the parameters with which the best result obtained on validation dataset are saved.

        """
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        self.check_stop(value_dict[self.stop_variable.name])
        logger.info("Monitoring on auxiliary data finished")

    def check_stop(self, stop_value):
        result = stop_value
        if result > self._best_result:
            self.wait_time = 0
            self._best_result = result
            # Update saved model parameters
            if self.model_saver is not None:
                self.model_saver.save_model()
        else:
            self.wait_time += 1
        if self.wait_time > self.tolerate_time:
            # Log best result on stop_variable on validation dataset
            self.main_loop.status['best_valid_result'.format(self.stop_variable.name)] = self.best_result
            self.main_loop.status['batch_interrupt_received'] = True
            self.main_loop.status['epoch_interrupt_received'] = True
            self.main_loop.current_row['training_finished'] = True