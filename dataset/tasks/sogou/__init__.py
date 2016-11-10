from fuel.datasets import IndexableDataset
from fuel.schemes import ConstantScheme, ShuffledExampleScheme, SequentialScheme, SequentialExampleScheme
from fuel.streams import DataStream
from fuel.transformers import Batch, Mapping, SortMapping, Unpack, Padding

from pml.dataset.tasks.sogou.transform import (QuerySample, QueryMerge, TokenSample, OutputNoise, MatrixPadding)
from pml.fuel.transform import _balanced_batch_helper