"""Annotated computation graph management."""
import logging
from collections import OrderedDict
import warnings

import numpy
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.compile.pfunc import rebuild_collect_shared
from blocks.config import config
from blocks.roles import (add_role, has_roles, AUXILIARY, PARAMETER, DROPOUT,
                     COLLECTED, COLLECTOR)
from blocks.utils import (shared_floatx_zeros)
import blocks.graph as graph

logger = logging.getLogger(__name__)


class ComputationGraph(graph.ComputationGraph):
    r"""Encapsulates a managed Theano computation graph.

    This implies that it not only contains the variables required to
    compute the given outputs, but also all the auxiliary variables and
    updates that were attached to these variables through the annotation
    system.

    All variables are presented in topologically sorted order according to
    the apply nodes that they are an input to.

    Parameters
    ----------
    outputs : (list of) :class:`~tensor.TensorVariable`
        The output(s) of the computation graph.

    Attributes
    ----------
    inputs : list of :class:`~tensor.TensorVariable`
        The inputs of the computation graph. This does not include shared
        variables and constants.
    shared_variables : list of :class:`~tensor.TensorSharedVariable`
        All the shared variables in the graph.
    parameters : list of :class:`~tensor.TensorSharedVariable`
        All the shared variables which have the :const:`.PARAMETER` role.
    outputs : list of :class:`~tensor.TensorVariable`
        The outputs of the computations graph (as passed to the
        constructor).
    auxiliary_variables : list of :class:`~tensor.TensorVariable`
        All variables which have the :const:`.AUXILIARY` role.
    intermediary_variables : list of :class:`~tensor.TensorVariable`
        Any variable that is not part of :attr:`inputs` or :attr:`outputs`.
    variables : list of :class:`~tensor.TensorVariable`
        All variables (including auxiliary) in the managed graph.
    scans : list of :class:`~theano.scan_module.scan_op.Scan`
        All Scan ops used in this computation graph.
    scan_variables : list of :class:`~tensor.TensorVariable`
        All variables of the inner graphs of Scan ops.
    updates : :class:`~tensor.TensorSharedVariable` updates
        All the updates found attached to the annotations.

    """

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        Parameters
        ----------
        replacements : dict
            The mapping from variables to be replaced to the corresponding
            substitutes.

        Examples
        --------

        Let's suppose we have dependent replacements like

        '(((x + TensorConstant{2}) + TensorConstant{3}) +
        TensorConstant{5})'
        '(((x * TensorConstant{2}) * TensorConstant{3}) +
        TensorConstant{5})'

        First two sums turned into multiplications

        23.0

        """
        # Due to theano specifics we have to make one replacement in time
        replacements = OrderedDict(replacements)

        outputs_cur = self.outputs

        # `replacements` with previous replacements applied. We have to track
        # variables in the new graph corresponding to original replacements.
        replacement_keys_cur = []
        replacement_vals_cur = []
        # Sort `replacements` in topological order
        # variables in self.variables are in topological order
        remaining_replacements = replacements.copy()
        for variable in self.variables:
            if variable in replacements:
                if has_roles(variable, [AUXILIARY]):
                    warnings.warn(
                        "replace method was asked to replace a variable ({}) "
                        "that is an auxiliary variable.".format(variable))
                replacement_keys_cur.append(variable)
                # self.variables should not contain duplicates,
                # otherwise pop() may fail.
                replacement_vals_cur.append(
                    remaining_replacements.pop(variable))

        # if remaining_replacements is not empty
        if remaining_replacements:
            warnings.warn(
                "replace method was asked to replace a variable(s) ({}) "
                "that is not a part of the computational "
                "graph.".format(str(remaining_replacements.keys())))

        # Replace step-by-step in topological order
        while replacement_keys_cur:
            replace_what = replacement_keys_cur[0]
            replace_by = replacement_vals_cur[0]
            # We also want to make changes in future replacements
            # Replace with clone
            outputs_new = clone(
                outputs_cur + replacement_keys_cur[1:] +
                replacement_vals_cur[1:],
                replace={replace_what: replace_by})
            # Reconstruct outputs, keys, and values
            outputs_cur = outputs_new[:len(outputs_cur)]
            replacement_keys_cur = outputs_new[len(outputs_cur):
                                               len(outputs_cur) +
                                               len(replacement_keys_cur) - 1]
            replacement_vals_cur = outputs_new[len(outputs_cur) +
                                               len(replacement_keys_cur):]

        return ComputationGraph(outputs_cur)


def apply_noise(computation_graph, variables, level, seed=None):
    """Add Gaussian noise to certain variable of a computation graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph.
    variables : :class:`~tensor.TensorVariable`
        Variables to add noise to.
    level : float
        Noise level.
    seed : int, optional
        The seed with which
        :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` is initialized,
        is set to 1 by default.

    """
    if not seed:
        seed = config.default_seed
    rng = MRG_RandomStreams(seed)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return computation_graph.replace(replace)


def collect_parameters(computation_graph, parameters):
    """Replace parameters with a single shared variable.

    This can be useful if you need to calculate the full Hessian of a
    computational graph. It replaces parameters with slices of a single
    large vectors like


    Parameters
    ----------
    computation_graph : :class:`ComputationGraph` instance
        The managed Theano graph in which to collect parameters.
    parameters : list of Theano shared variables
        The parameters whose values should be collected.

    Returns
    -------
    ComputationGraph instance
        A new Theano graph which has all the given parameters collected
        into a single large shared variable.

    Notes
    -----
    Note that this replacement makes the training of the model
    significantly slower because of the large amount of Theano's
    ``set_subtensor`` calls needed to train the model.

    Examples
    --------

    The new graph only has a single shared variable. This variable receives
    the :const:`COLLECTOR` role.

    [collected_parameters]

    The bricks' variables have been replaced with reshaped segments of this
    single shared variable. These replacements are given the
    :const:`.COLLECTED` role.

    [Reshape{1}.0, Reshape{1}.0, Reshape{2}.0, Reshape{2}.0]

    """
    parameter_values, parameter_sizes, parameter_shapes = [], [], []
    for parameter in parameters:
        parameter_values.append(parameter.get_value(borrow=True))
        parameter_sizes.append(parameter_values[-1].size)
        parameter_shapes.append(parameter_values[-1].shape)

    new_parameters = shared_floatx_zeros(sum(parameter_sizes))
    new_parameters.set_value(numpy.concatenate([value.flatten()
                             for value in parameter_values]))
    new_parameters.name = 'collected_parameters'
    add_role(new_parameters, COLLECTOR)

    replacements = {}
    for parameter, shape, i, j in zip(parameters, parameter_shapes,
                                      numpy.cumsum([0] + parameter_sizes[:-1]),
                                      numpy.cumsum(parameter_sizes)):
        new_parameter = new_parameters[i:j].reshape(shape)
        new_parameter.replacement_of = parameter
        add_role(new_parameter, COLLECTED)
        replacements[parameter] = new_parameter
    return computation_graph.replace(replacements)


def apply_dropout(computation_graph, variables, drop_prob, rng=None,
                  seed=None, custom_divisor=None):
    """Apply dropout to specified variables in a graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph.
    variables : list of :class:`~tensor.TensorVariable`
        Variables to be dropped out.
    drop_prob : float
        Probability of dropping out. If you want to apply the dropout
        with different probabilities for different layers, call it
        several times.
    rng : :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams`
        Random number generator.
    seed : int
        Random seed to be used if `rng` was not specified.
    custom_divisor : float or None, optional
        Divide dropped variables by a given scalar value. If `None`,
        (default) dropped variables will be divided by `(1 - drop_prob)`
        which is equivalent to scaling by `(1 - drop_prob)` at test
        time as recommended in [DROPOUT]_.

    Returns
    -------
    dropped_computation_graph : instance of :class:`ComputationGraph`
        A new computation graph with dropout applied to the specified
        variables. In order to train with, or monitor, the outputs
        of the original computation graph with dropout applies, use
        the variables contained in `dropped_computation_graph.outputs`.

    Notes
    -----
    For more information, see [DROPOUT]_.

    .. [DROPOUT] Hinton et al. *Improving neural networks by preventing
       co-adaptation of feature detectors*, arXiv:1207.0580.

    Examples
    --------

    We are going to drop out all the input variables


    Here we apply dropout with default setting to our computation graph


    Dropped out variables have role `DROPOUT` and are tagged with
    `replacement_of` tag. Let's filter these variables and check if they
    have the links to original ones.
    True

    Compiling theano functions to forward propagate in original and dropped
    out graphs

    Initialize an MLP and apply these functions

    array([[ 42.,  42.],
           [ 42.,  42.],
           [ 42.,  42.]]...
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]...

    And after the second run answer is different

    array([[   0.,   52.],
           [ 100.,    0.],
           [   0.,    0.]]...

    """
    if not rng and not seed:
        seed = config.default_seed
    if not rng:
        rng = MRG_RandomStreams(seed)
    if custom_divisor is None:
        divisor = (1 - drop_prob)
    else:
        divisor = custom_divisor
    replacements = [(var, var *
                     rng.binomial(var.shape, p=1 - drop_prob,
                                  dtype=theano.config.floatX) /
                     divisor)
                    for var in variables]
    for variable, replacement in replacements:
        add_role(replacement, DROPOUT)
        replacement.tag.replacement_of = variable

    return computation_graph.replace(replacements)


DEPRECATED_ARG = object()


def clone(output,
          replace=None,
          strict=True,
          share_inputs=True,
          copy_inputs=DEPRECATED_ARG):
    """
    Function that allows replacing subgraphs of a computational graph.

    It returns a copy of the initial subgraph with the corresponding
    substitutions.

    Parameters
    ----------
    output : Theano Variables (or Theano expressions)
        Theano expression that represents the computational graph.
    replace : dict
        Dictionary describing which subgraphs should be replaced by what.
    share_inputs : bool
        If True, use the same inputs (and shared variables) as the original
        graph. If False, clone them. Note that cloned shared variables still
        use the same underlying storage, so they will always have the same
        value.
    copy_inputs
        Deprecated, use share_inputs.

    """
    if copy_inputs is not DEPRECATED_ARG:
        warnings.warn('In `clone()` function, the argument `copy_inputs` has been deprecated and renamed into `share_inputs`')
        assert share_inputs  # since we used `copy_inputs` we should have default value for `share_inputs`
        share_inputs = copy_inputs

    if isinstance(replace, dict):
        items = list(replace.items())
    elif isinstance(replace, (list, tuple)):
        items = replace
    elif replace is None:
        items = []
    else:
        raise ValueError(("replace is neither a dictionary, list, "
                          "tuple or None ! The value provided is %s,"
                          "of type %s")%(str(replace), str(type(replace))))
    tmp_replace = [(x, x.type()) for x, y in items]
    new_replace = [(x, y) for ((_, x), (_, y)) in zip(tmp_replace,
                                                           items)]
    _, _outs, _ = rebuild_collect_shared(output,
                                         [],
                                         tmp_replace,
                                         [],
                                         strict,
                                         share_inputs)

    # TODO Explain why we call it twice ?!
    _, outs, clones = rebuild_collect_shared(_outs,
                                        [],
                                        new_replace,
                                        [],
                                        strict,
                                        share_inputs)
    for clone in clones[0].values():
        if hasattr(clone, 'tag'):
            if hasattr(clone.tag, 'subsets'):
                subsets = clone.tag.subsets
                clone.tag.subsets = []
                for subset in subsets:
                    if not hasattr(subset, 'name'):
                        raise ValueError('Name and unique name should be assigned to subset!')
                    match_subset = [i for i in clones[0].values() if hasattr(i, 'name') and i.name == subset.name]
                    if len(match_subset) > 1:
                        raise ValueError('Subset name conflict! Multiple subsets of {0} have the same name {1}'.
                                         format(clone, subset.name))
                    if len(match_subset) == 1:
                        clone.tag.subsets.append(match_subset[0])
    return outs


