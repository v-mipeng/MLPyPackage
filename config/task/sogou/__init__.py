from pml.config.base import BasicConfig
from pml.dataset.tasks.sogou.dataset import SingleTaskDataset, MultiTaskDataset

class BaseConfig(BasicConfig):
    def __init__(self):
        super(BaseConfig, self).__init__()


class SingleTaskConfig(BaseConfig):
    def __init__(self):
        super(SingleTaskConfig, self).__init__()

        source_names = ['id', 'age', 'gender', 'edu', 'query']

        self.task_name = 'age'

        self.Dataset = SingleTaskDataset


class MultiTaskConfig(BaseConfig):
    def __init__(self):
        super(MultiTaskConfig, self).__init__()
