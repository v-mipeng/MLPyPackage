
import theano
import numpy
import numpy as np

from pml.dataset.transform import *


class _balanced_batch_helper(object):
    def __init__(self, key):
        self.key = key
    def __call__(self, data):
        return len(data[self.key])


class NegativeSample(Transformer):

    def __init__(self, data_stream, dist_tables, sample_sources, sample_sizes, **kwargs):
        # produces_examples = False: invoke transform_batch() otherwise transform_example()
        super(NegativeSample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.dist_tables = dist_tables
        self.sample_sources = sample_sources
        self.sample_sizes = sample_sizes
        self._check_dist_table()

    def _check_dist_table(self):
        for i in range(len(self.dist_tables)):
            _,count = self.dist_tables[i]
            if not isinstance(count, numpy.ndarray):
                count = numpy.array(count)
            if sum(count == count.sum()) > 0:
                raise ValueError('Cannot apply negtive sampling for the probability of one element is 1.0')

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.sample_sources:
                sources.append(source + '_negtive_sample')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_samplings = []
        i = 0
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source not in self.sample_sources:
                batch_with_samplings.append(source_batch)
                continue

            neg_samples = []
            for source_example in source_batch:
                neg_sample = []
                while len(neg_sample) < self.sample_sizes[i]:
                    ids = self.sample_id(self.dist_tables[i], self.sample_sizes[i])
                    for id in ids:
                        if len(numpy.where(source_example == id)[0]) == 0:
                            neg_sample.append(id)
                            if len(neg_sample) == self.sample_sizes[i]:
                                break
                neg_samples.append(neg_sample)
            neg_samples = numpy.array(neg_samples, dtype= source_batch.dtype)
            batch_with_samplings.append(source_batch)
            batch_with_samplings.append(neg_samples)
        i = i+1
        return tuple(batch_with_samplings)


    def sample_id(self, num_by_id, sample_size = 1):
        # bisect search
        def bisect_search(sorted_na, value):
            '''
            Do bisect search
            :param sorted_na: cumulated sum array
            :param value: random value
            :return: the index that sorted_na[index-1]<=value<sorted_na[index] with defining sorted_na[-1] = -1
            '''
            if len(sorted_na) == 1:
                return 0
            left_index = 0
            right_index = len(sorted_na)-1

            while right_index-left_index > 1:
                mid_index = (left_index + right_index) / 2
                # in right part
                if value > sorted_na[mid_index]:
                    left_index = mid_index
                elif value < sorted_na[mid_index]:
                    right_index = mid_index
                else:
                    return min(mid_index+1,right_index)
            return right_index
        id, num = num_by_id
        cum_num = num.cumsum()
        rvs = numpy.random.uniform(low = 0.0, high = cum_num[-1], size=(sample_size,))
        ids = []
        for rv in rvs:
            if len(id) < 20000: # This value is obtained by test
                index = numpy.argmin(numpy.abs(cum_num-rv))
                if rv >= cum_num[index]:
                    index += 1
                else:
                    pass
            else:
                index = bisect_search(cum_num, rv)
            ids.append(id[index])
        return ids


class SparseIndex(Transformer):
    def __init__(self, data_stream, sparse_pairs, **kwargs):
        # produces_examples = False: invoke transform_batch() otherwise transform_example()
        super(SparseIndex, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sparse_sources, self.sparse_idxes = zip(*sparse_pairs)

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.sparse_sources:
                sources.append(source+"_sparse_mask")
            if source in self.sparse_idxes:
                sources.append(source + '_left_idx')
                sources.append(source + '_right_idx')
        return tuple(sources)

    def transform_batch(self, batch):
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.sparse_sources:
                # turn list of ndarray to one ndarray
                tmp = numpy.concatenate(source_batch, axis=0)
                if len(tmp) > 0:
                    mask = numpy.ones(len(tmp), dtype=theano.config.floatX)
                else:
                    tmp = numpy.array([0], dtype='int32') #TDDO: check here
                    mask = numpy.zeros(1, dtype=theano.config.floatX)
                new_batch.append(tmp)
                new_batch.append(mask)
            elif source in self.sparse_idxes:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i]*len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                if len(left_idxes) == 0:
                    left_idxes=[0]
                    right_idxes=[0]
                new_batch.append(numpy.array(left_idxes, dtype=source_batch[0].dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch[0].dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)


class CharEmbedding(Transformer):
    def __init__(self, data_stream, char_source, char_idx_source, **kwargs):
        super(CharEmbedding, self).__init__(data_stream=data_stream, produces_examples = False, **kwargs)
        self.char_source = char_source
        self.char_idx_source = char_idx_source

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.char_source:
                sources.append(source + '_mask')
                sources.append(source + '_sparse_mask')
            elif source in self.char_idx_source:
                sources.append(source + '_left_idx')
                sources.append(source + '_right_idx')
            else:
                pass
        return tuple(sources)

    def transform_batch(self, batch):
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.char_source:
                # turn list of ndarray to one ndarray
                char_batch = []
                for item in source_batch:
                    try:
                        char_batch += list(item)
                    except Exception as e:
                        print(type(item))
                if len(char_batch) == 0:
                    padded_batch = numpy.array([[0,0]], dtype="int32")
                    batch_mask = numpy.array([[1.,1.]], dtype=theano.config.floatX)
                    mask = numpy.zeros(1, dtype=theano.config.floatX)
                else:
                    padded_batch, batch_mask = self._padding(numpy.asarray(char_batch))
                    mask = numpy.ones(len(padded_batch), dtype=theano.config.floatX)
                new_batch.append(padded_batch)
                new_batch.append(batch_mask)
                new_batch.append(mask)
            elif source in self.char_idx_source:
                new_batch.append(source_batch)
                i = 0
                left_idxes = []
                right_idxes = []
                for idxes in source_batch:
                    left_idxes += [i] * len(idxes)
                    right_idxes += idxes.tolist()
                    i += 1
                if len(left_idxes) == 0:
                    left_idxes = [0]
                    right_idxes = [0]
                new_batch.append(numpy.array(left_idxes, dtype=source_batch[0].dtype))
                new_batch.append(numpy.array(right_idxes, dtype=source_batch[0].dtype))
            else:
                new_batch.append(source_batch)
        return tuple(new_batch)

    def _padding(self, source_batch):
        shapes = [numpy.asarray(sample).shape for sample in source_batch]
        lengths = [shape[0] for shape in shapes]
        max_sequence_length = max(lengths)
        rest_shape = shapes[0][1:]
        if not all([shape[1:] == rest_shape for shape in shapes]):
            raise ValueError("All dimensions except length must be equal")
        dtype = numpy.asarray(source_batch[0]).dtype

        padded_batch = numpy.zeros(
            (len(source_batch), max_sequence_length) + rest_shape,
            dtype=dtype)
        for i, sample in enumerate(source_batch):
            padded_batch[i, :len(sample)] = sample

        mask = numpy.zeros((len(source_batch), max_sequence_length),
                           dtype = theano.config.floatX)
        for i, sequence_length in enumerate(lengths):
            mask[i, :sequence_length] = 1
        return padded_batch, mask


class FeatureSample(Transformer):
    def __init__(self, data_stream, sample_source, sample_prob, sample_exp = 1., *args, **kwargs):
        '''
        For a given example, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(FeatureSample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sample_source = sample_source
        self.sample_prob = sample_prob
        self.sample_exp = sample_exp
        self._initialize()

    def _initialize(self):
        lens = []
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            lens += batch[self.sample_source].sum(axis=1).tolist()
        lens = numpy.array(lens)
        self.ave_len = lens.mean()

    def transform_batch(self, batch):
        batch_with_samplings = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source:
                batch_with_samplings.append(source_batch)
                continue
            rvs = numpy.random.rand(*source_batch.shape)
            # sample_prob may be larger than 1.0, this mean no sampling applied.
            # sample_prob = ((self.ave_len / source_batch.sum(axis=1))**self.sample_exp) * self.sample_prob
            source_batch[rvs > self.sample_prob] *= 0.
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class TokenSample(Transformer):
    def __init__(self, data_stream, sample_prob, sample_source='query_mask', **kwargs):
        '''For a given sample, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :return:
        '''
        super(TokenSample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sample_source = sample_source
        self.sample_prob = sample_prob
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_with_samplings = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source:
                batch_with_samplings.append(source_batch)
                continue
            rvs = numpy.random.rand(*source_batch.shape)
            source_batch *= rvs < self.sample_prob
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class TokenSampleByFrequency(Transformer):
    def __init__(self,  data_stream, sample_source, neg_prob_source, sample_exp = 1., *args, **kwargs):
        super(TokenSampleByFrequency, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sample_source = sample_source
        self.neg_prob_source = neg_prob_source
        self.sample_exp = sample_exp
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_with_samplings = []
        dic = dict(zip(self.data_stream.sources, batch))
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source:
                batch_with_samplings.append(source_batch)
                continue
            neg_sample_probs = dic[self.neg_prob_source]
            rvs = numpy.random.rand(*source_batch.shape)
            source_batch *= rvs > neg_sample_probs
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class QuerySample(Transformer):
    def __init__(self, sample_prob, sample_source='query_mask', seed=1234, *args, **kwargs):
        '''For a given user, sample its queries.

        :param data_stream: fuel.Datastream
        :param sample_sources: str
                Name of the sampled source.
        :param sample_prob: float (0.,1.)
                Sample out sample_prob * size(queries) queries
        :param seed: int
                Int seed for np.random. The seed will increment one every time the transform applied.
        '''
        super(QuerySample, self).__init__(produces_examples=False, **kwargs)
        self.seed = seed
        self.sample_source = sample_source
        self.sample_prob = sample_prob
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_with_samplings = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source:
                batch_with_samplings.append(source_batch)
                continue
            np.random.seed(self.seed)
            rvs = np.random.rand(*source_batch.shape[0:2])
            self.seed += 1
            source_batch *= (rvs < self.sample_prob)[:,:,None]
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class QueryMerge(Transformer):
    '''Merge queries of a user into bag of words'''
    def __init__(self, merge_source='query', *args, **kwargs):
        '''For a given user, sample its queries.

        :param data_stream: fuel.Datastream
        :param sample_sources: str
                Name of the sampled source.
        :param sample_prob: float (0.,1.)
                Sample out sample_prob * size(queries) queries
        :param seed: int
                Int seed for np.random. The seed will increment one every time the transform applied.
        '''
        super(QueryMerge, self).__init__(produces_examples=False, **kwargs)
        self.merge_source = merge_source
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_merged = []
        new_mask = None
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.merge_source:
                batch_merged.append(source_batch)
                continue
            queries = []
            masks = batch[self.data_stream.sources.index(self.merge_source+'_mask')]
            max_len = 0
            for i in range(len(source_batch)):
                words = np.unique(source_batch[i][masks[i] > 0.])
                if len(words) > max_len:
                    max_len = len(words)
                queries.append(words)
            new_batch = np.zeros((len(source_batch), max_len), dtype=source_batch.dtype)
            new_mask = np.zeros((len(source_batch), max_len), dtype=masks.dtype)
            for i in range(len(source_batch)):
                new_batch[i][0:len(queries[i])] = queries[i]
                new_mask[i][0:len(queries[i])] = 1.
            batch_merged.append(new_batch)
        batch_merged[self.data_stream.sources.index(self.merge_source+'_mask')] = new_mask
        return tuple(batch_merged)


class BaggedQuerySample(Transformer):
    def __init__(self, sample_source, sample_prob, sample_source_mask = None, **kwargs):
        '''
        For a given user, sample its queries and bag words of sampled queries
        :param data_stream:
        :param sample_sources: name of query mask
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(BaggedQuerySample, self).__init__(**kwargs)
        self.sample_source = sample_source
        if sample_source_mask is None:
            sample_source_mask = sample_source + '_mask'
        self.sample_source_mask = sample_source_mask
        self.sample_prob = sample_prob
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_with_samplings = []
        new_mask = None
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source:
                batch_with_samplings.append(source_batch)
                continue
            query_mask = batch[self.data_stream.sources.index(self.sample_source_mask)]
            rvs = numpy.random.rand(*source_batch.shape[0:2])
            masks = query_mask * (rvs < self.sample_prob)[:,:,None]
            queries = []
            max_len = 0
            for i in range(len(source_batch)):
                words = numpy.unique(source_batch[i][masks[i] > 0.])
                if len(words) > max_len:
                    max_len = len(words)
                queries.append(words)
            new_batch = numpy.zeros((len(source_batch), max_len), dtype='int32')
            new_mask = numpy.zeros((len(source_batch), max_len), dtype=theano.config.floatX)
            for i in range(len(source_batch)):
                new_batch[i][0:len(queries[i])] = queries[i]
                new_mask[i][0:len(queries[i])] = 1.
            batch_with_samplings.append(new_batch)
        batch_with_samplings[self.data_stream.sources.index(self.sample_source_mask)] = new_mask
        return tuple(batch_with_samplings)


class OutputNoise(Transformer):
    def __init__(self, output_source, max_noise_prob, label2freq, decay_rate=1., **kwargs):
        '''
        For a given example, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(OutputNoise, self).__init__(produces_examples=False, **kwargs)
        self.output_source = output_source
        self.max_noise_prob = max_noise_prob
        self.label2freq = label2freq
        self.decay_rate = decay_rate
        self._initialize()

    def _initialize(self):
        noise_probs = []
        for i in range(len(self.label2freq)):
            noise_probs.append(self.label2freq.get(i, 0))
        noise_probs = numpy.array(noise_probs, dtype=theano.config.floatX)
        label_nums = noise_probs.copy()
        noise_probs /= noise_probs.max()
        self.noise_probs = noise_probs * self.max_noise_prob

        idxes = numpy.arange(len(self.label2freq))
        self.cum_lable_nums = []
        for label in range(len(self.label2freq)):
            label_num = label_nums.copy()
            label_num[label] = 0.
            label_num /= numpy.exp(self.decay_rate * numpy.abs(idxes - label))
            label_num /= label_num.sum()
            label_num *= self.noise_probs[label]
            self.cum_lable_nums.append(label_num.cumsum())

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source == self.output_source:
                sources.append(source + '_noised_label')
            else:
                pass
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_noise = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            batch_with_noise.append(source_batch)
            if source != self.output_source:
                continue

            noised_batch = []
            rvs = numpy.random.rand(*source_batch.shape)
            for rv, label in zip(rvs, source_batch):
                noised_label = numpy.zeros(len(self.label2freq))
                if rv < self.noise_probs[label]:
                    for i in range(len(self.cum_lable_nums[label])):
                        if rv < self.cum_lable_nums[label][i]:
                            noised_label[i] = self.noise_probs[label]
                            noised_label[label] = 1.-self.noise_probs[label]
                            noised_batch.append(noised_label)
                            break
                else:
                    noised_label[label] = 1.
                    noised_batch.append(noised_label)
            batch_with_noise.append(numpy.array(noised_batch, dtype=theano.config.floatX))
        return tuple(batch_with_noise)


class MatrixPadding(Padding):
    """Adds padding to variable-size matrixes.

    When your batches consist of variable-size matrixes, use this class
    to equalize sizes by adding zero-padding. To distinguish between
    data and padding masks can be produced. For each data source that is
    masked, a new source will be added. This source will have the name of
    the original source with the suffix ``_mask`` (e.g. ``features_mask``).

    Element of incoming batches will be treated as a list or numpy type of matrix.
    Element of a matrix is a list or numpy array.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap
    mask_sources : tuple of strings, optional
        The sources for which we need to add a mask. If not provided, a
        mask will be created for all data sources
    mask_dtype: str, optional
        data type of masks. If not provided, floatX from config will
        be used.

    """

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

            shapes = [self._get_matrix_shape(sample) for sample in source_batch]
            rows, cols = zip(*shapes)
            max_rows = max(rows)
            max_cols = max(cols)
            dtype = numpy.asarray(source_batch[0][0]).dtype

            padded_batch = numpy.zeros((len(source_batch), max_rows, max_cols), dtype=dtype)
            mask = numpy.zeros((len(source_batch), max_rows, max_cols), dtype=self.mask_dtype)
            for i, sample in enumerate(source_batch):
                for j, row in enumerate(sample):
                    padded_batch[i, j, :len(row)] = numpy.asarray(row)
                    mask[i, j, :len(row)] = 1
            batch_with_masks.append(padded_batch)
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)

    def _get_matrix_shape(self, matrix):
        row = len(matrix)
        cols = [len(item) for item in matrix]
        max_col = max(cols)
        return (row, max_col)
