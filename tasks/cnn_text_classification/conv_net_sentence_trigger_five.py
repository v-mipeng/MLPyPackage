"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import os
os.environ['THEANO_FLAGS'] = 'device=gpu0'
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
warnings.filterwarnings("ignore")

from ..cnn_text_classification.conv_net_classes_trigger_five import *

#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

key_num = 1       
def train_conv_net(datasets,
                   datasets_trigger,
                   U,
                   img_w=100, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,36061], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Tanh],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (100 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1 
    img_h_trigger = len(datasets_trigger[0][0])-1

    print "img:\t", img_h
    print "img_h_triger:\t", img_h_trigger
 
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    filter_trigger_shapes = []
    pool_trigger_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))     # 100(kernel num) * 1 * 3 * 100(embed dim)
        pool_sizes.append(((img_h-filter_h+1)/key_num, img_w-filter_w+1))
    filter_trigger_shapes.append((1, 1, 5, img_w))
    pool_trigger_sizes.append((1, img_w))
        
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    x_trigger = T.matrix('x_trigger')
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    layer_trigger_input = Words[T.cast(x_trigger.flatten(),dtype="int32")].reshape((x_trigger.shape[0],1,x_trigger.shape[1],Words.shape[1]))                                  
    
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        
        #conv_trigger_layer = TriggerWord(rng, input=layer_trigger_input,image_shape=(batch_size, 1, img_h, img_w),
        #                        filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)

        
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    #out_layer = layer1_inputs
    conv_trigger_layer = TriggerWord(rng, input=layer0_input, input_three=layer_trigger_input, image_shape=(batch_size, 1, img_h, img_w), image_shape_three=(batch_size, 1, img_h_trigger, img_w),
                                filter_shape=filter_trigger_shapes[0], poolsize=pool_trigger_sizes[0], non_linear=conv_non_linear)
    layer_hidden_input = conv_trigger_layer.output.flatten(2)
#    conv_layers.append(conv_trigger_layer)
    
    #hidden_layer = HiddenTriggerLayer(rng, input=layer_hidden_input,n_in=feature_maps,n_out=feature_maps, activation=Tanh)
    #layer1_input = hidden_layer.output.flatten(2)
    #layer1_input_test = conv_trigger_layer.output.flatten(2)
    #conv_layers.append(conv_trigger_layer)
    #conv_layers.append(hidden_layer)
#    layer1_inputs.append(layer_hidden_input)


    #trigger_filter = conv_trigger_layer.filter
    trigger_filter = x_trigger
    #tmp1_out = conv_trigger_layer.tmp1
    #print tmp1_out

    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = key_num*feature_maps*(len(filter_hs))  
    #classifier = MLP(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    classifier = MLP(rng, input=layer1_input, n_in=hidden_units[0], n_hidden=500, n_out=hidden_units[1], activations=activations)
    
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) 
    #dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, cost, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.8))
    #divide train set into train/val sets 
#    test_set_x = datasets[1][:,:img_h] 
#    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    #test_set_x, test_set_y = (datasets[1][:,:img_h],datasets[1][:,-1])
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    #val_set_x, val_set_y = (val_set[:,:img_h],val_set[:,-1])

    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num_test = batch_size - datasets[1].shape[0] % batch_size
        test_set = np.random.permutation(datasets[1])   
        extra_data_test = test_set[:extra_data_num_test]
        new_data_test = np.append(datasets[1],extra_data_test,axis=0)
    else:
        new_data_test = datasets[1]
    new_data_test = np.random.permutation(new_data_test)
    n_batches_test = new_data_test.shape[0]/batch_size
    test_set = new_data_test
    test_set_x, test_set_y = shared_dataset((test_set[:,:img_h], test_set[:,-1]))


    print "trigger data process"    
    np.random.seed(3435)
    if datasets_trigger[0].shape[0] % batch_size > 0:
        extra_data_num_trigger = batch_size - datasets_trigger[0].shape[0] % batch_size
        train_set_trigger = np.random.permutation(datasets_trigger[0])   
        extra_data_trigger = train_set_trigger[:extra_data_num_trigger]
        new_data_trigger=np.append(datasets_trigger[0],extra_data_trigger,axis=0)
    else:
        new_data_trigger = datasets_trigger[0]
    new_data_trigger = np.random.permutation(new_data_trigger)
    n_batches = new_data_trigger.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.99))
    #divide train set into train/val sets 
#    test_set_x = datasets[1][:,:img_h] 
#    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    #test_set_x, test_set_y = (datasets[1][:,:img_h],datasets[1][:,-1])
    train_set_trigger = new_data_trigger[:n_train_batches*batch_size,:]
    val_set_trigger = new_data_trigger[n_train_batches*batch_size:,:]     
    train_set_x_trigger, train_set_y_trigger = shared_dataset((train_set_trigger[:,:img_h_trigger],train_set_trigger[:,-1]))
    val_set_x_trigger, val_set_y_trigger = shared_dataset((val_set_trigger[:,:img_h_trigger],val_set_trigger[:,-1]))
    #val_set_x, val_set_y = (val_set[:,:img_h],val_set[:,-1])

    if datasets_trigger[1].shape[0] % batch_size > 0:
        extra_data_num_test_trigger = batch_size - datasets_trigger[1].shape[0] % batch_size
        test_set_trigger = np.random.permutation(datasets_trigger[1])   
        extra_data_test_trigger = test_set_trigger[:extra_data_num_test_trigger]
        new_data_test_trigger = np.append(datasets_trigger[1],extra_data_test_trigger,axis=0)
    else:
        new_data_test_trigger = datasets_trigger[1]
    new_data_test_trigger = np.random.permutation(new_data_test_trigger)
    n_batches_test = new_data_test_trigger.shape[0]/batch_size
    test_set_trigger = new_data_test_trigger
    test_set_x_trigger, test_set_y_trigger = shared_dataset((test_set_trigger[:,:img_h_trigger], test_set_trigger[:,-1]))



    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
            x_trigger: val_set_x_trigger[index * batch_size: (index + 1) * batch_size],
            y: val_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True, on_unused_input='ignore')
            
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                x_trigger: train_set_x_trigger[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True, on_unused_input='ignore')               
    train_model = theano.function([index], [cost, layer1_input, trigger_filter], updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
            x_trigger: train_set_x_trigger[index*batch_size:(index+1)*batch_size],
            y: train_set_y[index*batch_size:(index+1)*batch_size]}, allow_input_downcast=True, on_unused_input='ignore')     
    #test_model_all = theano.function([x,y], classifier.errors(y), allow_input_downcast=True)   
    test_model_all = theano.function([index], classifier.errors_raw(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                x_trigger: test_set_x_trigger[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True, on_unused_input='ignore')               
    
    test_model_all_tops = theano.function([index], [classifier.errors_raw(y), classifier.p_y_given_x, classifier.y_pred, y],
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                x_trigger: test_set_x_trigger[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]}, allow_input_downcast=True, on_unused_input='ignore')               
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    
    while (epoch < n_epochs):        
        epoch = epoch + 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch,layer1,trigger = train_model(minibatch_index)
               # print ".....................................", layer1.shape
               # print "......................", trigger.shape
               # print "......................", tmp1_out.shape
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch,layer1 = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch %i, train perf %f %%, val perf %f %%, test perf %f %%' % (epoch, train_perf * 100., val_perf*100., test_perf*100.))
        if val_perf + train_perf >= best_val_perf:
            best_val_perf = val_perf + train_perf
            test_losses = [numpy.mean(test_model_all(i)) for i in xrange(n_batches_test-1)]        
            test_last_loss = numpy.mean(test_model_all(n_batches_test - 1)[:datasets_trigger[1].shape[0]%batch_size])
            test_losses.append(test_last_loss)
            test_perf = 1- np.mean(test_losses)    
        if epoch % 50 == 0: 
            f_write_result = open('./result','w')
            for i in xrange(n_batches_test - 1):
                presult,pred,pred_y,y = test_model_all_tops(i)
               
                for j in xrange(len(y)):
                   f_write_result.write(str(y[j]) + "\t")
                   for r in pred[j]:
                       f_write_result.write(str(r) + " ")
                   f_write_result.write('\n')
            
            presult,pred,pred_y,y = test_model_all_tops(n_batches_test - 1)
            #print presult
            #print np.argmax(pred, axis=1)
            #print pred_y
            #print y
            
            for j in xrange(datasets_trigger[1].shape[0]%batch_size):
                f_write_result.write(str(y[j]) + "\t")
                for r in pred[j]:
                    f_write_result.write(str(r) + " ")
                f_write_result.write('\n')
            f_write_result.close()
    
    return test_perf * 100.

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=100, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    #print "................", sent
    for word in words:
        
        if word in word_idx_map:
            x.append(word_idx_map[word])
            #print word_idx_map[word]
        else:
           print word
    while len(x) < max_l+2*pad:
        x.append(0)
    #print "....................", x
    return x

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=100, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)   
        #print sent
        sent.append(rev["y"])
        if rev["split"]==0:            
            test.append(sent)        
        else:  
            train.append(sent)  
    #print train 
    train = np.array(train,dtype="int")
    #print test
    test = np.array(test,dtype="int")
    return [train, test]     
  
   
if __name__=="__main__":
    print "loading data...",
    # You should modify here in case you do not have such pickled file.
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    
    #print revs
    #print "revs",revs[1]
    #print "revs",revs[2]
    W = np.array(W, dtype="float32") * 10
   
    W2 = np.array(W2, dtype="float32")
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes_trigger_five.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,1)    
    for i in r:
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=68, k=100, filter_h=3)
        datasets_trigger = make_idx_data_cv(revs, word_idx_map, i, max_l=68, k=100, filter_h=5)
        #print "datasets:\t",datasets
        perf = train_conv_net(datasets,
                              datasets_trigger,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="tanh",
                              hidden_units=[100,36061], 
                              shuffle_batch=True, 
                              n_epochs=500, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=200,
                              dropout_rate=[0.95])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)  
    print str(np.mean(results))
