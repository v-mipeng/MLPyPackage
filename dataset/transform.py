import logging
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from multiprocessing import Process, Queue
from warnings import warn

import numpy
from fuel import config
from fuel.exceptions import AxisLabelsMismatchError
from fuel.schemes import BatchSizeScheme
from fuel.streams import AbstractDataStream
from picklable_itertools import chain, ifilter, izip
from six import add_metaclass, iteritems

log = logging.getLogger(__name__)


class ExpectsAxisLabels(object):
    """Mixin for transformers, used to verify axis labels.

    Notes
    -----
    Provides a method :meth:`verify_axis_labels` that should be called
    with the expected and actual values for an axis labels tuple. If
    `actual` is `None`, a warning is logged; if it is non-`None` and does
    not match `expected`, a :class:`AxisLabelsMismatchError` is raised.

    The check is only performed on the first call; if the call succeeds,
    an attribute is written to skip further checks, in the interest of
    speed.

    """
    def verify_axis_labels(self, expected, actual, source_name):
        """Verify that axis labels for a given source are as expected.

        Parameters
        ----------
        expected : tuple
            A tuple of strings representing the expected axis labels.
        actual : tuple or None
            A tuple of strings representing the actual axis labels, or
            `None` if they could not be determined.
        source_name : str
            The name of the source being checked. Used for caching the
            results of checks so that the check is only performed once.

        Notes
        -----
        Logs a warning in case of `actual=None`, raises an error on
        other mismatches.

        """
        if not getattr(self, '_checked_axis_labels', False):
            self._checked_axis_labels = defaultdict(bool)
        if not self._checked_axis_labels[source_name]:
            if actual is None:
                log.warning("%s instance could not verify (missing) axis "
                            "expected %s, got None",
                            self.__class__.__name__, expected)
            else:
                if expected != actual:
                    raise AxisLabelsMismatchError("{} expected axis labels "
                                                  "{}, got {} instead".format(
                                                      self.__class__.__name__,
                                                      expected, actual))
            self._checked_axis_labels[source_name] = True


@add_metaclass(ABCMeta)
class Transformer(AbstractDataStream):
    """A data stream that wraps another data stream.

    Subclasses must define a `transform_batch` method (to act on batches),
    a `transform_example` method (to act on individual examples), or
    both methods.

    Typically (using the interface mentioned above), the transformer
    is expected to have the same output type (example or batch) as its
    input type.  If the transformer subclass is going from batches to
    examples or vice versa, it should override `get_data` instead.
    Overriding `get_data` is also necessary when access to `request` is
    necessary (e.g. for the :class:`Cache` transformer).

    Attributes
    ----------
    child_epoch_iterator : iterator type
        When a new epoch iterator is requested, a new epoch creator is
        automatically requested from the wrapped data stream and stored in
        this attribute. Use it to access data from the wrapped data stream
        by calling ``next(self.child_epoch_iterator)``.
    produces_examples : bool
        Whether this transformer produces examples (as opposed to batches
        of examples).

    """
    def __init__(self, name=None, produces_examples=None, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        if name is None:
            name = self.__class__
        self.name = name
        self._data_stream = None
        self.produces_examples = produces_examples
        self.former_transformer = None

    @property
    def data_stream(self):
        return self._data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value

    @property
    def produces_examples(self):
        if self._produces_examples is None:
            return self.data_stream.produces_examples
        else:
            return self._produces_examples

    @produces_examples.setter
    def produces_examples(self, value):
        self._produces_examples = value

    def __add__(self, other):
        if other.former_transformer is None:
            other.former_transformer = self
        else:
            self + other.former_transformer
        return other

    def apply(self, data_stream):
        if self.former_transformer is None:
            self.data_stream = data_stream
        else:
            self.former_transformer.apply(data_stream)
            self.data_stream = self.former_transformer
        return self

    def remove(self, remove_which):
        '''Remove sub-transformer from piped transformers

        :param transformer_name: str
                Transformer name
        :return: pml.transform.Transformer
                Piped transformers with given transformer removed.
        '''
        if remove_which == self.name:
            # If remove the top transformer
            former_transformer = self.former_transformer
            self.former_transformer = None
            return former_transformer
        else:
            if self.former_transformer is None:
                warn('Cannot find transformer:{0}!'.format(remove_which))
            elif remove_which == self.former_transformer.name:
                tmp = self.former_transformer.former_transformer
                self.former_transformer.former_transformer = None
                self.former_transformer = tmp
            else:
                self.former_transformer.remove(remove_which)
            return self

    def insert(self, transformer, before_which=None):
        '''Insert transformer into current piped transformers

        Given transformer may also be piped transformers. Current piped transformer will be changed
        internally

        :param transformer: pml.transform.Transformer
                Transformer to be inserted into current piped transformers.
        :param before_which: str
                Name of transformer before which to insert the given transformer.
        '''
        assert isinstance(transformer, Transformer)
        if before_which == self.name:
            tmp = transformer
            while tmp.former_transformer is not None:
                tmp = tmp.former_transformer
            tmp.former_transformer = self.former_transformer
            self.former_transformer = transformer
        elif self.former_transformer is None:
            warn('Cannot find transformer:{0}!'.format(before_which))
        else:
            self.former_transformer.insert(transformer, before_which)

    @property
    def sources(self):
        if hasattr(self, '_sources'):
            return self._sources
        return self.data_stream.sources

    @sources.setter
    def sources(self, value):
        self._sources = value

    def close(self):
        self.data_stream.close()

    def reset(self):
        self.data_stream.reset()

    def next_epoch(self):
        self.data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        """Get an epoch iterator for the wrapped data set.

        Notes
        -----
        This default implementation assumes that the epochs of the wrapped
        data stream are less or equal in length to the original data
        stream. Implementations for which this is not true should request
        new epoch iterators from the child data set when necessary.

        """
        self.child_epoch_iterator = self.data_stream.get_epoch_iterator()
        return super(Transformer, self).get_epoch_iterator(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)

        if self.produces_examples != self.data_stream.produces_examples:
            types = {True: 'examples', False: 'batches'}
            raise NotImplementedError(
                "the wrapped data stream produces {} while the {} transformer "
                "produces {}, which it does not support.".format(
                    types[self.data_stream.produces_examples],
                    self.__class__.__name__,
                    types[self.produces_examples]))
        elif self.produces_examples:
            return self.transform_example(data)
        else:
            return self.transform_batch(data)

    def transform_example(self, example):
        """Transforms a single example."""
        raise NotImplementedError(
            "`{}` does not support examples as input, but the wrapped data "
            "stream produces examples.".format(self.__class__.__name__))

    def transform_batch(self, batch):
        """Transforms a batch of examples."""
        raise NotImplementedError(
            "`{}` does not support batches as input, but the wrapped data "
            "stream produces batches.".format(self.__class__.__name__))


@add_metaclass(ABCMeta)
class AgnosticTransformer(Transformer):
    """A transformer that operates the same on examples or batches.

    Subclasses must implement the `transform_any` method, which is to be
    applied to both examples and batches. This is useful when the example
    and batch implementation of a transformation are the same.

    """
    def __init__(self, *args, **kwargs):
        super(AgnosticTransformer, self).__init__(*args, **kwargs)

    @property
    def data_stream(self):
        return self._data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value

    @abstractmethod
    def transform_any(self, data):
        """Transforms the input, which can either be an example or a batch."""

    def transform_example(self, example):
        return self.transform_any(example)

    def transform_batch(self, batch):
        return self.transform_any(batch)


class Mapping(Transformer):
    """Applies a mapping to the data of the wrapped data stream.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    mapping : callable
        The mapping to be applied.
    add_sources : tuple of str, optional
        When given, the data produced by the mapping is added to original
        data under source names `add_sources`.

    """
    def __init__(self, mapping, add_sources=None, **kwargs):
        kwargs.setdefault('name', 'Mapping')
        super(Mapping, self).__init__(**kwargs)
        self.mapping = mapping
        self.add_sources = add_sources

    @property
    def sources(self):
        return self.data_stream.sources + (self.add_sources
                                           if self.add_sources else ())

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        data = next(self.child_epoch_iterator)
        image = self.mapping(data)
        if not self.add_sources:
            return image
        return data + image


@add_metaclass(ABCMeta)
class SourcewiseTransformer(Transformer):
    """Applies a transformation sourcewise.

    Subclasses must define `transform_source_example` (to transform
    examples), `transform_source_batch` (to transform batches) or
    both.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The wrapped data stream.
    which_sources : tuple of str, optional
        Which sources to apply the mapping to. Defaults to `None`, in
        which case the mapping is applied to all sources.

    """
    def __init__(self, which_sources=None, **kwargs):
        super(SourcewiseTransformer, self).__init__(**kwargs)
        self.which_sources = which_sources

    @property
    def data_stream(self):
        return self._data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value

    @property
    def which_sources(self):
        if self._which_sources is None:
            return self.data_stream.sources
        else:
            return self._which_sources

    @which_sources.setter
    def which_sources(self, value):
        self._which_sources = value

    def _apply_sourcewise_transformation(self, data, method):
        data = list(data)
        for i, source_name in enumerate(self.data_stream.sources):
            if source_name in self.which_sources:
                data[i] = method(data[i], source_name)
        return tuple(data)

    def transform_source_example(self, source_example, source_name):
        """Applies a transformation to an example from a source.

        Parameters
        ----------
        source_example : :class:`numpy.ndarray`
            An example from a source.
        source_name : str
            The name of the source being operated upon.

        """
        raise NotImplementedError(
            "`{}` does not support examples as input, but the wrapped data "
            "stream produces examples.".format(self.__class__.__name__))

    def transform_source_batch(self, source_batch, source_name):
        """Applies a transformation to a batch from a source.

        Parameters
        ----------
        source_batch : :class:`numpy.ndarray`
            A batch of examples from a source.
        source_name : str
            The name of the source being operated upon.

        """
        raise NotImplementedError(
            "`{}` does not support batches as input, but the wrapped data "
            "stream produces batches.".format(self.__class__.__name__))

    def transform_example(self, example):
        return self._apply_sourcewise_transformation(
            data=example, method=self.transform_source_example)

    def transform_batch(self, batch):
        return self._apply_sourcewise_transformation(
            data=batch, method=self.transform_source_batch)


@add_metaclass(ABCMeta)
class AgnosticSourcewiseTransformer(AgnosticTransformer,
                                    SourcewiseTransformer):
    """A sourcewise transformer that operates the same on examples or batches.

    Subclasses must implement the `transform_any_source` method, which is
    to be applied to both examples and batches. This is useful when the
    example and batch implementation of a sourcewise transformation are
    the same.

    """
    def __init__(self, *args, **kwargs):
        super(AgnosticSourcewiseTransformer, self).__init__(*args, **kwargs)

    def transform_any(self, data):
        return self._apply_sourcewise_transformation(
            data=data, method=self.transform_any_source)

    @abstractmethod
    def transform_any_source(self, source_data, source_name):
        """Applies a transformation to a source.

        The data can either be an example or a batch of examples.

        Parameters
        ----------
        source_data : :class:`numpy.ndarray`
            Data from a source.
        source_name : str
            The name of the source being operated upon.

        """


class Flatten(SourcewiseTransformer):
    """Flattens selected sources.

    If the wrapped data stream produces batches, they will be flattened
    along all but the first axis.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    """
    def __init__(self, **kwargs):
        # Modify the axis_labels dict to reflect the fact that all non-batch
        # axes will be grouped together under the same 'feature' axis.
        kwargs.setdefault('name', 'Flatten')
        super(Flatten, self).__init__(**kwargs)

    @property
    def data_stream(self):
        return super(Flatten, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        if self.axis_labels is None:
            if self._data_stream.axis_labels:
                self.axis_labels = self._infer_axis_labels(self._data_stream, self.which_sources)

    def _infer_axis_labels(self, data_stream, which_sources):
        axis_labels = {}
        for source, labels in iteritems(data_stream.axis_labels):
            if source in which_sources:
                if not labels:
                    axis_labels[source] = None
                elif data_stream.produces_examples:
                    axis_labels[source] = ('feature',)
                else:
                    axis_labels[source] = (labels[0], 'feature')
            else:
                axis_labels[source] = labels
        return axis_labels

    def transform_source_example(self, source_example, _):
        return numpy.asarray(source_example).flatten()

    def transform_source_batch(self, source_batch, _):
        return numpy.asarray(source_batch).reshape((len(source_batch), -1))


class ScaleAndShift(AgnosticSourcewiseTransformer):
    """Scales and shifts selected sources by scalar quantities.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    Parameters
    ----------
    scale : float
        Scaling factor.
    shift : float
        Shifting factor.

    """
    def __init__(self, scale, shift, **kwargs):
        kwargs.setdefault('name', 'ScaleAndShift')
        super(ScaleAndShift, self).__init__(**kwargs)
        self.scale = scale
        self.shift = shift

    def transform_any_source(self, source_data, _):
        return numpy.asarray(source_data) * self.scale + self.shift


class Cast(AgnosticSourcewiseTransformer):
    """Casts selected sources as some dtype.

    Incoming sources will be treated as numpy arrays (i.e. using
    `numpy.asarray`).

    Parameters
    ----------
    dtype : str
        Data type to cast to. Can be any valid numpy dtype, or 'floatX',
        in which case ``fuel.config.floatX`` is used.

    """
    def __init__(self, dtype, **kwargs):
        kwargs.setdefault('name', 'Cast')
        if dtype == 'floatX':
            dtype = config.floatX
        self.dtype = dtype
        super(Cast, self).__init__(**kwargs)

    def transform_any_source(self, source_data, _):
        return numpy.asarray(source_data, dtype=self.dtype)


class ForceFloatX(AgnosticSourcewiseTransformer):
    """Force all floating point numpy arrays to be floatX."""
    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'ForceFloatX')
        super(ForceFloatX, self).__init__(**kwargs)

    def transform_any_source(self, source_data, _):
        source_needs_casting = (isinstance(source_data, numpy.ndarray) and
                                source_data.dtype.kind == "f" and
                                source_data.dtype != config.floatX)
        if source_needs_casting:
            source_data = source_data.astype(config.floatX)
        return source_data


class Filter(Transformer):
    """Filters samples that meet a predicate.

    Parameters
    ----------
    data_stream : instance of :class:`DataStream`
        The filtered data stream.
    predicate : callable
        Should return ``True`` for the samples to be kept.

    """
    def __init__(self, predicate, **kwargs):
        kwargs.setdefault('name', 'Filter')
        super(Filter, self).__init__(**kwargs)
        self.predicate = predicate

    def get_epoch_iterator(self, **kwargs):
        super(Filter, self).get_epoch_iterator(**kwargs)
        return ifilter(self.predicate, self.child_epoch_iterator)


class Cache(Transformer):
    """Cache examples when sequentially reading a dataset.

    Given a data stream which reads large chunks of data, this data
    stream caches these chunks and returns smaller batches from it until
    exhausted.

    Parameters
    ----------
    iteration_scheme : :class:`.IterationScheme`
        Note that this iteration scheme must return batch sizes (integers),
        which must necessarily be smaller than the child data stream i.e.
        the batches returned must be smaller than the cache size.

    Attributes
    ----------
    cache : list of lists of objects
        This attribute holds the cache at any given point. It is a list of
        the same size as the :attr:`sources` attribute. Each element in
        this list in its turn a list of examples that are currently in the
        cache. The cache gets emptied at the start of each epoch, and gets
        refilled when needed through the :meth:`get_data` method.

    """
    def __init__(self, iteration_scheme, **kwargs):
        # Note: produces_examples will always be False because of this
        # restriction: the only iteration schemes allowed are BatchSizeScheme,
        # which produce batches.
        kwargs.setdefault('name', 'Cache')
        if not isinstance(iteration_scheme, BatchSizeScheme):
            raise ValueError('iteration scheme must be an instance of '
                             'BatchSizeScheme')
        super(Cache, self).__init__(iteration_scheme=iteration_scheme, **kwargs)
        self.cache = [[] for _ in self.sources]

    def get_data(self, request=None):
        if request is None:
            raise ValueError
        if request > len(self.cache[0]):
            self._cache()
        data = []
        for i, cache in enumerate(self.cache):
            data.append(numpy.asarray(cache[:request]))
            self.cache[i] = cache[request:]
        return tuple(data)

    def get_epoch_iterator(self, **kwargs):
        self.cache = [[] for _ in self.sources]
        return super(Cache, self).get_epoch_iterator(**kwargs)

    def _cache(self):
        try:
            for cache, data in zip(self.cache,
                                   next(self.child_epoch_iterator)):
                cache.extend(data)
        except StopIteration:
            if not self.cache[0]:
                raise


class SortMapping(object):
    """Callable class for creating sorting mappings.

    This class can be used to create a callable that can be used by the
    :class:`Mapping` constructor.

    Parameters
    ----------
    key : callable
        The mapping that returns the value to sort on. Its input will be
        a tuple that contains a single data point for each source.
    reverse : boolean value that indicates whether the sort order should
        be reversed.

    """
    def __init__(self, key, reverse=False):
        self.key = key
        self.reverse = reverse

    def __call__(self, batch):
        output = sorted(zip(*batch), key=self.key, reverse=self.reverse)
        output = tuple(numpy.asarray(i) if isinstance(j, numpy.ndarray)
                       else list(i)
                       for i, j in zip(zip(*output), batch))
        return output


class Batch(Transformer):
    """Creates minibatches from data streams providing single examples.

    Some datasets only return one example at at time e.g. when reading text
    files a line at a time. This wrapper reads several examples
    sequentially to turn those into minibatches.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to wrap.
    iteration_scheme : :class:`.BatchSizeScheme` instance
        The iteration scheme to use; should return integers representing
        the size of the batch to return.
    strictness : int, optional
        How strictly the iterator should adhere to the batch size. By
        default, the value 0 means that the last batch is returned
        regardless of its size, so it can be smaller than what is actually
        requested. At level 1, the last batch is discarded if it is not of
        the correct size. At the highest strictness level, 2, an error is
        raised if a batch of the requested size cannot be provided.

    """
    def __init__(self, iteration_scheme, strictness=0, **kwargs):
        # The value for `produces_examples` is inferred from the iteration
        # scheme's `requests_examples` attribute. We expect the scheme to
        # request batches.
        kwargs.setdefault('name', 'Batch')
        if iteration_scheme.requests_examples:
            raise ValueError('the iteration scheme must request batches, '
                             'not individual examples.')
        super(Batch, self).__init__(iteration_scheme=iteration_scheme,
                                    produces_examples=False,
                                    **kwargs)
        self.strictness = strictness

    @property
    def data_stream(self):
        return super(Batch, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        if not self._data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce examples, '
                             'not batches of examples.')
        if self.axis_labels is None:
            if self._data_stream.axis_labels:
                self.axis_labels = dict((source, ('batch',) + labels if labels else None) for
                                        source, labels in iteritems(self._data_stream.axis_labels))

    def get_data(self, request=None):
        """Get data from the dataset."""
        if request is None:
            raise ValueError
        data = [[] for _ in self.sources]
        for i in range(request):
            try:
                for source_data, example in zip(
                        data, next(self.child_epoch_iterator)):
                    source_data.append(example)
            except StopIteration:
                # If some data has been extracted and `strict` is not set,
                # we should spit out this data before stopping iteration.
                if not self.strictness and data[0]:
                    break
                elif self.strictness > 1 and data[0]:
                    raise ValueError
                raise
        return tuple(numpy.asarray(source_data) for source_data in data)


class Unpack(Transformer):
    """Unpacks batches to compose a stream of examples.

    This class is the inverse of the Batch class: it turns a minibatch into
    a stream of examples.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` instance
        The data stream to unpack

    """
    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Unpack')
        super(Unpack, self).__init__(produces_examples=True, **kwargs)
        self.data = None

    @property
    def data_stream(self):
        return super(Unpack, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        if self._data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')
        if self.axis_labels is None:
            if self._data_stream.axis_labels:
                self.axis_labels = dict((source, labels[1:] if labels else None) for
                                        source, labels in iteritems(self._data_stream.axis_labels))
        
    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        if not self.data:
            data = next(self.child_epoch_iterator)
            self.data = izip(*data)
        try:
            return next(self.data)
        except StopIteration:
            self.data = None
            return self.get_data()


class Padding(Transformer):
    """Adds padding to variable-length sequences.

    When your batches consist of variable-length sequences, use this class
    to equalize lengths by adding zero-padding. To distinguish between
    data and padding masks can be produced. For each data source that is
    masked, a new source will be added. This source will have the name of
    the original source with the suffix ``_mask`` (e.g. ``features_mask``).

    Elements of incoming batches will be treated as numpy arrays (i.e.
    using `numpy.asarray`). If they have more than one dimension,
    all dimensions except length, that is the first one, must be equal.

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
    def __init__(self, mask_sources=None, mask_dtype=None,
                 **kwargs):
        kwargs.setdefault('name', 'Padding')
        super(Padding, self).__init__(produces_examples=False, **kwargs)
        if mask_sources is None:
            mask_sources = self.data_stream.sources
        self.mask_sources = mask_sources
        if mask_dtype is None:
            self.mask_dtype = config.floatX
        else:
            self.mask_dtype = mask_dtype

    @property
    def data_stream(self):
        return super(Padding, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        if self._data_stream.produces_examples:
            raise ValueError('the wrapped data stream must produce batches of '
                             'examples, not examples')

    @property
    def sources(self):
        sources = []
        for source in self.data_stream.sources:
            sources.append(source)
            if source in self.mask_sources:
                sources.append(source + '_mask')
        return tuple(sources)

    def transform_batch(self, batch):
        batch_with_masks = []
        for i, (source, source_batch) in enumerate(
                zip(self.data_stream.sources, batch)):
            if source not in self.mask_sources:
                batch_with_masks.append(source_batch)
                continue

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
            batch_with_masks.append(padded_batch)

            mask = numpy.zeros((len(source_batch), max_sequence_length),
                               self.mask_dtype)
            for i, sequence_length in enumerate(lengths):
                mask[i, :sequence_length] = 1
            batch_with_masks.append(mask)
        return tuple(batch_with_masks)


class Merge(AbstractDataStream):
    """Merges several datastreams into a single one.

    Parameters
    ----------
    data_streams : iterable
        The data streams to merge.
    sources : iterable
        A collection of strings, determining what sources should be called.

    Examples
    --------
    >>> from fuel.datasets import IterableDataset
    >>> english = IterableDataset(['Hello world!'])
    >>> french = IterableDataset(['Bonjour le monde!'])
    >>> from fuel.streams import DataStream
    >>> streams = (DataStream(english),
    ...            DataStream(french))
    >>> merged_stream = Merge(streams, ('english', 'french'))
    >>> merged_stream.sources
    ('english', 'french')
    >>> next(merged_stream.get_epoch_iterator())
    ('Hello world!', 'Bonjour le monde!')

    """
    def __init__(self, data_streams, sources, axis_labels=None):
        super(Merge, self).__init__(
            iteration_scheme=None, axis_labels=axis_labels)
        if not all(data_stream.produces_examples ==
                   data_streams[0].produces_examples
                   for data_stream in data_streams):
            raise ValueError('all data streams must produce the same type of '
                             'output (batches or examples)')
        self.data_streams = data_streams
        self.produces_examples = self.data_streams[0].produces_examples

        if len(list(chain(*[data_stream.sources for data_stream
                            in data_streams]))) != len(sources):
            raise ValueError("wrong number of sources given")
        self.sources = sources

    def close(self):
        for data_stream in self.data_streams:
            data_stream.close()

    def reset(self):
        for data_stream in self.data_streams:
            data_stream.reset()

    def next_epoch(self):
        for data_stream in self.data_streams:
            data_stream.next_epoch()

    def get_epoch_iterator(self, **kwargs):
        self.child_epoch_iterators = [data_stream.get_epoch_iterator()
                                      for data_stream in self.data_streams]
        return super(Merge, self).get_epoch_iterator(**kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        result = []
        for child_epoch_iterator in self.child_epoch_iterators:
            result.extend(next(child_epoch_iterator))
        return tuple(result)


class Rename(AgnosticTransformer):
    """Renames the sources of the stream.

    Parameters
    ----------
    data_stream : :class:`DataStream` or :class:`Transformer`.
        The data stream.
    names : dict
        A dictionary mapping the old and new names of the sources
        to rename.
    on_non_existent : str, optional
        Desired behaviour when a source specified as a key in `names`
        is not provided by the streams: see `on_overwrite` above for
        description of possible values. Default is 'raise'.

    """
    def __init__(self, names, on_non_existent='raise', **kwargs):
        kwargs.setdefault('name', 'Rename')
        if on_non_existent not in ('raise', 'ignore', 'warn'):
            raise ValueError("on_non_existent must be one of 'raise', "
                             "'ignore', 'warn'")
        # We allow duplicate values in the full dictionary, but those
        # that correspond to keys that are real sources in the data stream
        # must be unique. This lets you use one piece of code including
        # a Rename transformer to map disparately named sources in
        # different datasets to a common name.
        self.names = names
        self.on_non_existent = on_non_existent
        super(Rename, self).__init__(**kwargs)

    @property
    def data_stream(self):
        return super(Rename, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        usable_names = {k: v for k, v in iteritems(self.names)
                        if k in self._data_stream.sources}

        if len(set(usable_names.values())) != len(usable_names):
            raise KeyError("multiple old source names cannot map to "
                           "the same new source name")
        sources = list(self._data_stream.sources)
        sources_lookup = {n: i for i, n in enumerate(sources)}
        for old, new in iteritems(self.names):
            if new in sources_lookup and new not in self.names:
                if old in usable_names:
                    message = ("Renaming source '{}' to '{}' "
                               "would create two sources named '{}'"
                               .format(old, new, new))
                    raise KeyError(message)
            if old not in sources_lookup:
                message = ("Renaming source '{}' to '{}': "
                           "stream does not provide a source '{}'"
                           .format(old, new, old))
                if self.on_non_existent == 'raise':
                    raise KeyError(message)
                else:
                    log_level = {'warn': logging.WARNING,
                                 'ignore': logging.DEBUG}
                    log.log(log_level[self.on_non_existent], message)
            else:
                sources[sources_lookup[old]] = new
        self.sources = tuple(sources)
        if self.axis_labels is None:
            if self._data_stream.axis_labels:
                self.axis_labels = dict((source, ('batch',) + labels if labels else None) for
                                        source, labels in iteritems(self._data_stream.axis_labels))

    def transform_any(self, data):
        return data


class FilterSources(AgnosticTransformer):
    """Selects a subset of the stream sources.

    Order of data stream's sources is maintained. The order of sources
    given as parameter to FilterSources does not matter.

    Parameters
    ----------
    data_stream : :class:`AbstractDataStream` or :class:`Transformer`.
        The data stream.
    sources : tuple of strings
        The names of the data sources returned by this transformer.
        Must be a subset of the sources given by the stream.

    """
    def __init__(self, sources, **kwargs):
        kwargs.setdefault('name', 'FilterSources')
        super(FilterSources, self).__init__(**kwargs)
        # keep order of data_stream.sources
        self.reserved_sources = sources

    @property
    def data_stream(self):
        return super(FilterSources, self).data_stream

    @data_stream.setter
    def data_stream(self, value):
        self._data_stream = value
        if any(source not in self._data_stream.sources for source in self.reserved_sources):
            raise ValueError("sources must all be contained in "
                             "data_stream.sources")
        self.sources = tuple(s for s in self._data_stream.sources if s in self.reserved_sources)
        if self.axis_labels is None:
            if self._data_stream.axis_labels:
                self.axis_labels = dict((source, ('batch',) + labels if labels else None) for
                                        source, labels in iteritems(self._data_stream.axis_labels))

    def transform_any(self, data):
        return [d for d, s in izip(data, self.data_stream.sources)
                if s in self.sources]


# TODO: Test these transformers and add doc string

class OutputNoise(Transformer):
    '''Add noise to output'''
    def __init__(self, output_source, max_noise_prob, label2freq, decay_rate=1., **kwargs):
        '''
        For a given example, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(OutputNoise, self).__init__(**kwargs)
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


class FeatureSample(Transformer):
    def __init__(self, sample_source, sample_prob, sample_exp=1., *args, **kwargs):
        '''
        For a given example, sample its feature with probability given by sample_prob
        :param data_stream:
        :param sample_sources: Features on which sampling is applied. Commonly, if mask is applied, sample_sources should be mask names
        :param sample_prob:
        :param kwargs:
        :return:
        '''
        super(FeatureSample, self).__init__(**kwargs)
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
            source_batch[rvs > self.sample_prob] *= 0.
            batch_with_samplings.append(source_batch)
        return tuple(batch_with_samplings)


class NegativeSample(Transformer):

    def __init__(self, dist_tables, sample_sources, sample_sizes, **kwargs):
        # produces_examples = False: invoke transform_batch() otherwise transform_example()
        super(NegativeSample, self).__init__(produces_examples=False, **kwargs)
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
            i += 1
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
        import theano
        new_batch = []
        for source, source_batch in zip(self.data_stream.sources, batch):
            if source in self.sparse_sources:
                # turn list of ndarray to one ndarray
                tmp = numpy.concatenate(source_batch, axis=0)
                if len(tmp) > 0:
                    mask = numpy.ones(len(tmp), dtype=theano.config.floatX)
                else:
                    tmp = numpy.array([0], dtype='int32')
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
