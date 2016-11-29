import numpy as np

from pml.model.extensions.monitoring import TrainingDataMonitor, EarlyStopMonitor

# Deprecated


class SogouTrainingDataMonitor(TrainingDataMonitor):
    def do(self, callback_name, *args):
        """Initializes the buffer or commits the values to the log.

        What this method does depends on from what callback it is called.
        When called within `before_training`, it initializes the
        aggregation buffer and instructs the training algorithm what
        additional computations should be carried at each step by adding
        corresponding updates to it. In all other cases it writes
        aggregated values of the monitored variables to the log.

        """
        if callback_name == 'before_training':
            self.main_loop.algorithm.add_updates(
                self._buffer.accumulation_updates)
            self._buffer.initialize_aggregators()
        else:
            if (self.main_loop.status['iterations_done'] ==
                    self._last_time_called):
                raise Exception("TrainingDataMonitoring.do should be invoked"
                                " no more than once per iteration")
            self._last_time_called = self.main_loop.status['iterations_done']
            dic = self._buffer.get_aggregated_values()
            self.add_records(self.main_loop.log, dic.items())
            self.main_loop.log.status['train_accuracy'] = dic['accuracy']
            self._buffer.initialize_aggregators()


class SogouValidDataMonitor(EarlyStopMonitor):

    def check_stop(self, stop_value):
        train_accuracy = self.main_loop.log.status['train_accuracy']
        result = 0.9*stop_value + 0.1*train_accuracy
        super(SogouValidDataMonitor, self).check_stop(result)


