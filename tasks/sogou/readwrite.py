import os
from collections import OrderedDict

import numpy
import numpy as np

from pml.dataset.readwrite import AbstractDatasetReaderWriter
from pml.dataset.base import DatasetContainer


#region Reader writer for pre-processing
class SogouTrainRawDatasetReaderWriter(AbstractDatasetReaderWriter):

    def read_dataset(self, read_from=None):
        if read_from is None:
            read_from = self.read_from
        print('Reading data from {0}'.format(read_from))
        names = ['id', 'age', 'gender', 'edu', 'query']
        ids = []
        ages = []
        genders = []
        edus = []
        queries_per_user = []
        with open(read_from, 'r') as f:
            for line in f:
                array = line.strip().split('\t')
                ids.append(array[0])
                ages.append(int(array[1]))
                genders.append(int(array[2]))
                edus.append(int(array[3]))
                queries_per_user.append(array[4:])
        ids = np.array(ids)
        query_by_users = np.asarray(queries_per_user, dtype='O')
        ages = np.array(ages)
        genders = np.array(genders)
        edus = np.array(edus)
        print('Done!')
        return DatasetContainer(OrderedDict(zip(names, [ids, ages, genders, edus, query_by_users])))

    def write_dataset(self, dataset, save_to=None):
        '''Save preprocessed dataset into file

        :param dataset: pml.dataset.base.DatasetContainer
        '''
        if save_to is None:
            save_to = self.save_to
        if not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        if len(dataset.sources) > 2:
            iter_order = ['id', 'age', 'gender', 'edu', 'query']
        else:
            iter_order = ['id', 'query']
        print('Save processed dataset into {0} with fields:{1}'.format(save_to, ' '.join(iter_order)))
        dataset.iter_order = iter_order
        with open(save_to, 'w+') as writer:
            for output_tuple in dataset:
                writer.write('{0}'.format('\t'.join(map(str, output_tuple[:-1]))))
                writer.write('\t{0}\n'.format('\t'.join([' '.join([word.encode('utf-8') for word in query])
                                                         for query in output_tuple[-1]])))
        print('Done!')


class SogouPredictRawDatasetReaderWriter(SogouTrainRawDatasetReaderWriter):

    def read_dataset(self, read_from=None):
        if read_from is None:
            read_from = self.read_from
        print('Reading data from {0}'.format(read_from))
        names = ['id', 'query']
        ids = []
        queries_per_user = []
        with open(read_from, 'r') as f:
            for line in f:
                array = line.strip().split('\t')
                ids.append(array[0])
                queries_per_user.append(array[1:])
        ids = np.array(ids)
        query_by_users = np.asarray(queries_per_user, dtype='O')
        print('Done!')
        return DatasetContainer(OrderedDict(zip(names, [ids, query_by_users])))

#endregion


class SogouTrainDatasetReaderWriter(SogouTrainRawDatasetReaderWriter):
    def __init__(self, config, *args, **kwargs):
        kwargs.setdefault('read_from', config.train_data_read_from)
        try:
            kwargs.setdefault('save_to', config.train_data_save_to)
        except:
            pass
        super(SogouTrainDatasetReaderWriter, self).__init__(*args, **kwargs)


class SogouPredictDatasetReaderWriter(SogouPredictRawDatasetReaderWriter):
    def __init__(self, config, *args, **kwargs):
        kwargs.setdefault('read_from', config.predict_data_read_from)
        try:
            kwargs.setdefault('save_to', config.predict_result_save_to)
        except:
            pass
        super(SogouPredictDatasetReaderWriter, self).__init__(*args, **kwargs)
        self.config = config

    def write_dataset(self, dataset, save_to=None):
        if save_to is None:
            save_to = self.save_to
        if len(dataset.sources) > 2:
            iter_order = ['id', 'age', 'gender', 'edu']
        else:
            source = [source for source in dataset.sources if source != 'id'][0]
            iter_order = ['id', source]
        print('Save prediction results with fields:{0}'.format(' '.join(iter_order)))
        dataset.iter_order = iter_order
        pred_label2true_labels = []
        for task_name in iter_order[1:]:
            pred_label2true_labels.append(self.config.dataset.get_pred_label2true_label(task_name))
        if not os.path.exists(os.path.dirname(save_to)):
            os.makedirs(os.path.dirname(save_to))
        with open(save_to, 'w+') as writer:
            for output_tuple in dataset:
                outputs = list(output_tuple)
                for i in range(1, len(output_tuple)):
                    outputs[i] = pred_label2true_labels[i-1][output_tuple[i]]
                writer.write('{0}\n'.format(' '.join(map(str, outputs))))
        print('Done!')
