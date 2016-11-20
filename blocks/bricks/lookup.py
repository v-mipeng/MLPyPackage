"""Introduces Lookup brick."""
from blocks.bricks import Initializable, Feedforward
from blocks.bricks.base import application, lazy
from blocks.roles import WEIGHT, add_role
from blocks.utils import check_theano_variable, shared_floatx_nans


class LookupTable(Initializable, Feedforward):
    """Encapsulates representations of a range of integers.

    This brick can be used to embed integers, e.g. word indices,
    into a vector space.

    Parameters
    ----------
    length : int
        The size of the lookup table, or in other words, one plus the
        maximum index for which a representation is contained.
    dim : int
        The dimensionality of representations.

    Notes
    -----
    See :class:`.Initializable` for initialization parameters.

    """
    has_bias = False

    @lazy(allocation=['length', 'dim'])
    def __init__(self, length, dim, **kwargs):
        super(LookupTable, self).__init__(**kwargs)
        self.length = length
        self.dim = dim

    @property
    def W(self):
        return self.parameters[0]

    def get_subsets(self):
        if hasattr(self.W.tag, 'subsets'):
            return self.W.tag.subsets
        else:
            return None

    def get_subset_idxes(self):
        if hasattr(self.W.tag, 'idxes'):
            return self.W.tag.idxes
        else:
            return None

    def remove_subsets(self):
        if hasattr(self.W.tag, 'subsets'):
            delattr(self.W.tag, 'subsets')
            delattr(self.W.tag, 'idxes')

    def _allocate(self):
        self.parameters.append(shared_floatx_nans((self.length, self.dim),
                               name='W'))
        add_role(self.parameters[-1], WEIGHT)

    def _initialize(self):
        self.weights_init.initialize(self.W, self.rng)

    @application(inputs=['indices'], outputs=['output'])
    def apply(self, indices):
        """Perform lookup.

        Parameters
        ----------
        indices : :class:`~tensor.TensorVariable`
            The indices of interest. The dtype must be integer.

        Returns
        -------
        output : :class:`~tensor.TensorVariable`
            Representations for the indices of the query. Has :math:`k+1`
            dimensions, where :math:`k` is the number of dimensions of the
            `indices` parameter. The last dimension stands for the
            representation element.

        """
        check_theano_variable(indices, None, ("int", "uint"))
        output_shape = [indices.shape[i]
                        for i in range(indices.ndim)] + [self.dim]
        idx = indices.flatten()
        subset = self.W[idx]
        if not hasattr(self.W.tag, 'subsets'):
            subset.name = '{0}_subset_0'.format(self.name)
            self.W.tag.idxes = [idx]
            self.W.tag.subsets = [subset]
        else:
            subset.name = '{0}_subset_{1}'.format(self.name, len(self.W.tag.subsets))
            self.W.tag.idxes.append(idx)
            self.W.tag.subsets.append(subset)
        return subset.reshape(output_shape)

    def get_dim(self, name):
        if name == 'output':
            return self.dim
        if name == 'indices':
            return 0
        return super(LookupTable, self).get_dim(name)

    @property
    def input_dim(self):
        return 0

    @input_dim.setter
    def input_dim(self, dim):
        if dim != 0:
            raise ValueError("LookupTable input must be integer")

    @property
    def output_dim(self):
        return self.dim

    @output_dim.setter
    def output_dim(self, dim):
        self.dim = dim
