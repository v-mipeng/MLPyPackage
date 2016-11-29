import logging

from blocks.extensions import Printing, ProgressBar

from pml.entrance import TypicalEntrance
from pml.blocks.graph.model import Model
from pml.tasks.sogou.entension import SogouTrainingDataMonitor, SogouValidDataMonitor

logger = logging.getLogger('pml.tasks.sogou.entrance')


class SogouEntrance(TypicalEntrance):

    def _build_extension(self, valid_stream):
        extensions = [
            SogouTrainingDataMonitor(
                [v for v in self.model.train_monitors],
                prefix='train',
                every_n_batches=self.config.print_freq)
        ]
        n_batches = self.preprocessed_raw_train_dataset.sample_num / self.config.batch_size

        # Initialize model
        try:
            self.model_saver_loader.load_model(Model(self.model.train_cg_generator))
        except:
            print('Cannot initialize model! Train from the beginning!')
        extensions += [SogouValidDataMonitor(data_stream=valid_stream,
                                            variables=self.model.valid_monitors,
                                            stop_variable=self.model.valid_monitors[0],
                                            model_saver=self.model_saver_loader,
                                            tolerate_times=self.config.tolerate_times,
                                            start_monitor_epoch=self.config.start_valid_epoch,
                                            every_n_batches=int(self.config.valid_freq * n_batches))]
        extensions += [
            Printing(every_n_batches=self.config.print_freq, after_epoch=True),
            ProgressBar()
        ]
        return extensions