import logging

from blocks.extensions import SimpleExtension


logger = logging.getLogger('model.extensions')


class EpochClip(SimpleExtension):
    '''Set most epoch number to train'''
    def __init__(self, max_epoch, **kwargs):
        super(EpochClip, self).__init__(after_epoch=True, **kwargs)
        self.cur_epoch = 0
        self.max_epoch = max_epoch

    def do(self, which_callback, *args):
        if which_callback == "after_epoch":
            self.cur_epoch += 1
            if self.cur_epoch >= self.max_epoch:
                self.main_loop.status['epoch_interrupt_received'] = True

                self.main_loop.current_row['training_finished'] = True
                s = 'after_training'
