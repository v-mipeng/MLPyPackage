import theano
import numpy as np
from fuel.transformers import Transformer, Padding


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
            rvs = np.random.rand(*source_batch.shape)
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
            rvs = np.random.rand(*source_batch.shape)
            source_batch *= rvs > neg_sample_probs
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class QuerySample(Transformer):
    def __init__(self, data_stream, sample_prob, sample_source='query_mask', seed=1234, *args, **kwargs):
        '''For a given user, sample its queries.

        :param data_stream: fuel.Datastream
        :param sample_sources: str
                Name of the sampled source.
        :param sample_prob: float (0.,1.)
                Sample out sample_prob * size(queries) queries
        :param seed: int
                Int seed for np.random. The seed will increment one every time the transform applied.
        '''
        super(QuerySample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
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


class ConQuerySample(Transformer):
    def __init__(self, data_stream, sample_num, sample_source_mask='query_mask',
                 *args, **kwargs):
        '''
        For a given user, sample its queries and bag words of sampled queries
        :param data_stream:
        :param sample_sources: name of query mask
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(ConQuerySample, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.sample_source_mask = sample_source_mask
        self.sample_num = sample_num
        self._initialize()

    def _initialize(self):
        pass

    def transform_batch(self, batch):
        batch_with_samplings = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source != self.sample_source_mask:
                batch_with_samplings.append(source_batch)
                continue
            query_mask = source_batch
            full_idxes = numpy.arange(len(query_mask[0]))
            masks = []
            for mask in query_mask:
                idxes = full_idxes[mask[:, 0] > 0.]
                numpy.random.shuffle(idxes)
                mask[idxes[min(len(idxes), self.sample_num):], :] = 0.
                masks.append(mask)
            batch_with_samplings.append(numpy.array(masks, dtype=query_mask.dtype))
        return tuple(batch_with_samplings)


class QueryMerge(Transformer):
    '''Merge queries of a user into bag of words'''
    def __init__(self, data_stream, merge_source='query', *args, **kwargs):
        '''For a given user, sample its queries.

        :param data_stream: fuel.Datastream
        :param sample_sources: str
                Name of the sampled source.
        :param sample_prob: float (0.,1.)
                Sample out sample_prob * size(queries) queries
        :param seed: int
                Int seed for np.random. The seed will increment one every time the transform applied.
        '''
        super(QueryMerge, self).__init__(
            data_stream, produces_examples=False, **kwargs)
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


class OutputNoise(Transformer):
    def __init__(self, data_stream, output_source, max_noise_prob, label2freq, decay_rate=1., **kwargs):
        '''
        For a given example, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(OutputNoise, self).__init__(
            data_stream, produces_examples=False, **kwargs)
        self.output_source = output_source
        self.max_noise_prob = max_noise_prob
        self.label2freq = label2freq
        self.decay_rate = decay_rate
        self._initialize()

    def _initialize(self):
        noise_probs = []
        for i in range(len(self.label2freq)):
            noise_probs.append(self.label2freq.get(i, 0))
        noise_probs = np.array(noise_probs, dtype=theano.config.floatX)
        label_nums = noise_probs.copy()
        noise_probs /= noise_probs.max()
        self.noise_probs = noise_probs * self.max_noise_prob

        idxes = np.arange(len(self.label2freq))
        self.cum_lable_nums = []
        for label in range(len(self.label2freq)):
            label_num = label_nums.copy()
            label_num[label] = 0.
            label_num /= np.exp(self.decay_rate * np.abs(idxes - label))
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
            rvs = np.random.rand(*source_batch.shape)
            for rv, label in zip(rvs, source_batch):
                noised_label = np.zeros(len(self.label2freq))
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
            batch_with_noise.append(np.array(noised_batch, dtype=theano.config.floatX))
        return tuple(batch_with_noise)


class MatrixPadding(Padding):
    """Adds padding to variable-size matrixes.

    When your batches consist of variable-size matrixes, use this class
    to equalize sizes by adding zero-padding. To distinguish between
    data and padding masks can be produced. For each data source that is
    masked, a new source will be added. This source will have the name of
    the original source with the suffix ``_mask`` (e.g. ``features_mask``).

    Element of incoming batches will be treated as a list or np type of matrix.
    Element of a matrix is a list or np array.

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
            dtype = np.asarray(source_batch[0][0]).dtype

            padded_batch = np.zeros((len(source_batch), max_rows, max_cols), dtype=dtype)
            mask = np.zeros((len(source_batch), max_rows, max_cols), dtype=self.mask_dtype)
            for i, sample in enumerate(source_batch):
                for j, row in enumerate(sample):
                    padded_batch[i, j, :len(row)] = np.asarray(row)
                    mask[i, j, :len(row)] = 1
            batch_with_masks.append(padded_batch)
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)

    def _get_matrix_shape(self, matrix):
        row = len(matrix)
        cols = [len(item) for item in matrix]
        max_col = max(cols)
        return (row, max_col)
