"""Some of the simplest individual bricks."""
import logging

from blocks.bricks import Bias


logger = logging.getLogger(__name__)


class Vector(Bias):
    '''Construct a float type shared vector
    '''
    def __init__(self, *args, **kwargs):
        super(Vector, self).__init__(*args, **kwargs)

    @property
    def W(self):
        b, = self.parameters
        return b

