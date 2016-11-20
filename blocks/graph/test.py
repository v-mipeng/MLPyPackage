import theano

from pml.blocks.graph import clone

if __name__ == '__main__':
    import theano.tensor as tensor
    import numpy as np
    W = theano.shared(np.ones((3, 3)).astype(theano.config.floatX), name='W')
    x = tensor.vector('x', dtype='int32')
    y = W[x]
    y.name = 'y'
    W.tag.subsets = [y]

    z = y.sum()+W.sum()
    z_clone = clone(z)
    # grad_z = theano.grad(z, y)	# This is ok
    inputs = theano.gof.graph.ancestors([z_clone])
    grad_z_clone = theano.grad(z_clone, inputs[3])  # Error raised!
    f_grad_z_clone = theano.function([x], [grad_z_clone])
    nx = np.array([1,0]).astype('int32')
    print(f_grad_z_clone(nx))


