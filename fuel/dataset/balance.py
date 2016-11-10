import numpy as np


class DataBalancer(object):
    
    def __init__(self, dataset = None, label_index = 0, sample_size = 5, sample_times = 10, *args, **kwargs):
        self.reset(dataset, label_index, sample_size, sample_times, *args, **kwargs)
        
    def reset(self, dataset, label_index = 0, sample_size = 5, sample_times = 10, *args, **kwargs):
        self.label_index = label_index
        self.sample_size = sample_size
        self.sample_times = sample_times
        self.dataset = dataset
        
    @property
    def label_index(self):
        return self._label_index
    
    @label_index.setter
    def label_index(self, value):
        self._label_index = value

    @property
    def sample_size(self):
        return self._sample_size

    @sample_size.setter
    def sample_size(self, value):
        self._sample_size = value

    @property
    def sample_times(self):
        return self._sample_times

    @sample_times.setter
    def sample_times(self, value):
        self._sample_times = value

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = np.asarray(value, dtype='O')

    def get_balanced_datasets(self, random_seed = 1):
        fields = zip(*self.dataset)
        lables = np.array(fields[self.label_index])
        unique_labels, counts = np.unique(lables, return_counts=True)
        idxes = np.arange(len(lables))
        min_count = counts.min()
        datasets = []
        for i in range(self.sample_times):
            dataset = []
            j = 1
            for label, count in zip(unique_labels,counts):
                # Set random seed for re-generating
                np.random.seed((i+1)*j*random_seed)
                j += 1
                idxes_of_label = idxes[lables==label]
                if len(idxes_of_label) <= min_count*self.sample_size:
                    dataset += self.dataset[idxes_of_label].tolist()
                else:
                    rand_idxes = np.random.randint(low=0, high=len(idxes_of_label), size=min_count*self.sample_size)
                    sampled_idxes_of_label = idxes_of_label[rand_idxes]
                    dataset += self.dataset[sampled_idxes_of_label].tolist()
            datasets.append(dataset)
        return datasets