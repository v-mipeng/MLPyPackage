"""Model - heavily annotated computation graph.

A model in Blocks is simply an annotated computation graph.  The class
:class:`Model` extends :class:`blocks.graph.ComputationGraph` :class:,
which is able to handle annotations and roles in general, but is
deliberately made unaware of specific annotations that a Theano graph
created by Blocks typically has, such as bricks and application calls.  The
:class:`Model` adds this functionality. Using :class:`Model` you can do
things like query all the bricks used to build the computation graph,
request "hierarchical names" of the parameters (a hierarchical name is a
path-like string which in addition to the parameter's name contains names
of the bricks on the path from a root brick to the brick that owns the
parameters, e.g. ``/mlp/linear/W``).

For more information, see :class:`Model` docstring.

"""
import logging

import blocks.model as model

import pml.blocks.graph as graph

logger = logging.getLogger(__name__)


class Model(model.Model, graph.ComputationGraph):
    """Handles annotations in Blocks-built computation graphs.

    Use this class to handle your Blocks-created computation graph.

    Examples
    --------
    >>> from theano import tensor
    >>> from blocks.bricks import MLP, Tanh
    >>> x = tensor.matrix('x')
    >>> mlp = MLP([Tanh(), Tanh()], [10, 10, 10])
    >>> y = mlp.apply(x)
    >>> model = Model(y)

    With :class:`Model` you can get access to the brick hierarchy. The
    brick hierarchy is defined by ``children`` attributes that every brick
    has.  The bricks that are not children of other bricks are called top
    bricks.  It is often useful to have access to top bricks of a brick
    hierarchy used to build a computation graph, and here is how you can do
    it:

    >>> model.get_top_bricks() #doctest: +ELLIPSIS
    [<blocks.bricks.sequences.MLP object at ...]

    You can also get "hierarchical" names for the parameters,
    which encode the position of the owning brick in the
    brick hierarchy.

    >>> model.get_parameter_dict() #doctest: +NORMALIZE_WHITESPACE
    OrderedDict([('/mlp/linear_1.b', b), ('/mlp/linear_0.b', b),
    ('/mlp/linear_0.W', W), ('/mlp/linear_1.W', W)])

    """

    def replace(self, replacements):
        return graph.ComputationGraph.replace(self, replacements=replacements)