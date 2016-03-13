import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
  """
  """
  
  def __init__(self, input_dim=(3, 32, 32), reg=0.0, num_classes=10, weight_scale=1e-3,
               filter_sizes=[3, 5], num_filters=[32, 64], hidden_dims=[50, 50],
               dtype=np.float32, cv_dropout=0.0, fc_dropout=0.0, use_batchnorm=False):
    """
    
    Network architecture:
    [Conv -> ReLU (dropout) -> Pool]xN -> [Affine -> ReLU (dropout)]xM -> Affine -> Softmax loss
    
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - filter_sizes: List of filter sizes at each convolution layer
    - num_filters: List of number of filters at each convolution layer
    - hidden_dims: List of number of units in each affine layer
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      network should not use dropout at all
    - use_batchnorm: Whether or not network should batch normalization. Convolution
      layers use spatial batch normalization
       
    """
    
    assert len(num_filters) == len(filter_sizes), 'Number of filters and filter sizes must be same!'
    
    self.reg = reg
    self.use_batchnorm = use_batchnorm
    self.num_cv_layers = len(filter_sizes)
    self.num_fc_layers = len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.cn_params = []
    self.cv_bn_params = []
    self.fc_bn_params = []
    self.cv_dropout_param = {}
    self.fc_dropout_param = {}
    self.pool_param = {}
    
    C, H, W = input_dim
    
    assert H % (2 ** len(num_filters)) == 0, 'Input image size is not compatible with number of pooling layers'
    assert W % (2 ** len(num_filters)) == 0, 'Input image size is not compatible with number of pooling layers'

    num_filters[:0] = [C]
    hidden_dims[:0] = [(num_filters[-1] * H * W / (4 ** (self.num_cv_layers)))]
    
    # Define conv params for each conv layer
    self.cn_params = [{'stride': 1, 'pad': (filter_sizes[i] - 1) / 2, } for i in xrange(self.num_cv_layers)]
    
    # Define batchnorm params for all conv and affine layers
    if self.use_batchnorm:
      self.cv_bn_params = [{'mode': 'train', 'running_mean': 0.0, 'running_var': 0.0} for i in xrange(self.num_cv_layers)]
      self.fc_bn_params = [{'mode': 'train', 'running_mean': 0.0, 'running_var': 0.0} for i in xrange(self.num_fc_layers)]
    
    # Initialize weights and biases for all conv layers
    for idx in xrange(self.num_cv_layers):
      p = 'Wcv' + np.str(idx+1)
      self.params[p] = np.random.normal(0, weight_scale * (2.0 / ((filter_sizes[idx] ** 2) * num_filters[idx])), [num_filters[idx+1], num_filters[idx], filter_sizes[idx], filter_sizes[idx]])
      p = 'bcv' + np.str(idx+1)
      self.params[p] = np.zeros(num_filters[idx+1])
      if self.use_batchnorm:
    	p = 'gammacv' + np.str(idx+1)
        self.params[p] = np.float32(1.0)
        p = 'betacv' + np.str(idx+1)
        self.params[p] = np.float32(0.0)
    
    # Initialize weights and biases for all affine layers (except last layer)  
    for idx in xrange(self.num_fc_layers):
      p = 'Wfc' + np.str(idx+1)
      self.params[p] = np.random.normal(0, weight_scale * (2.0 / hidden_dims[idx]), [hidden_dims[idx], hidden_dims[idx+1]])
      p = 'bfc' + np.str(idx+1)
      self.params[p] = np.zeros(hidden_dims[idx+1])
      if self.use_batchnorm:
    	p = 'gammafc' + np.str(idx+1)
        self.params[p] = np.float32(1.0)
        p = 'betafc' + np.str(idx+1)
        self.params[p] = np.float32(0.0)       
    
    # Initialize weights and biases for lasst affine layer
    self.params['Wlast'] = np.random.normal(0, weight_scale * (2.0 / hidden_dims[-1]), [hidden_dims[-1], num_classes])
    self.params['blast'] = np.zeros(num_classes)
    
    self.cv_dropout_param = {'mode': 'train', 'p': cv_dropout}
    self.fc_dropout_param = {'mode': 'train', 'p': fc_dropout}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
    
    
  def loss(self, X, y=None):
    """
    Evaluate loss and gradients for the convolutional network.
    
    """

    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    self.cv_dropout_param['mode'] = mode
    self.fc_dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.cv_bn_params:
        bn_param['mode'] = mode
      for bn_param in self.fc_bn_params:
        bn_param['mode'] = mode  
    
    scores = None
    outs = []
    caches = []
    
    outs.append(X)
    # Forward pass of all convolution layers
    for idx in xrange(self.num_cv_layers):
      #print 'Forward pass of conv layer ', idx
      W = self.params['Wcv' + np.str(idx+1)]
      b = self.params['bcv' + np.str(idx+1)]
      out, cache = conv_forward_fast(outs.pop(), W, b, self.cn_params[idx])
      caches.append(cache)
      if self.use_batchnorm:
        gamma = self.params['gammacv' + np.str(idx+1)]
        beta = self.params['betacv' + np.str(idx+1)]
        out, cache = spatial_batchnorm_forward(out, gamma, beta, self.cv_bn_params[idx])
        caches.append(cache)
      out, cache = relu_forward(out)
      caches.append(cache)
      out, cache = max_pool_forward_fast(out, self.pool_param)
      caches.append(cache)
      out, cache = dropout_forward(out, self.cv_dropout_param)
      caches.append(cache)
      outs.append(out)
        
    # Forward pass of all affine (except last) layers
    for idx in xrange(self.num_fc_layers):
      #print 'Forward pass of affine layer ', idx
      W = self.params['Wfc' + np.str(idx+1)]
      b = self.params['bfc' + np.str(idx+1)]
      if self.use_batchnorm:
        gamma = self.params['gammafc' + np.str(idx+1)]
        beta = self.params['betafc' + np.str(idx+1)]
        bn_param = self.fc_bn_params[idx]
        out, cache = affine_batchnorm_relu_forward(outs.pop(), W, b, gamma, beta, bn_param)
        outs.append(out)
        caches.append(cache)
      else:
        out, cache = affine_relu_forward(outs.pop(), W, b) 
        outs.append(out)
        caches.append(cache)  
      out, cache = dropout_forward(outs.pop(), self.fc_dropout_param)
      outs.append(out)
      caches.append(cache)
        
    # Forward pass of lass affine layer
    W = self.params['Wlast']
    b = self.params['blast']
    scores, cache = affine_forward(outs.pop(), W, b)
    caches.append(cache)

    # If test mode return early
    if mode == 'test':
      return scores
    
    # Compute loss and gradients using backward pass      
    loss, grads = 0.0, {}
    data_loss = 0.0
    reg_loss = 0.0
    
    # Data loss
    data_loss, dscores = softmax_loss(scores, y)
    #print 'Data loss: ', data_loss
    
    # Regularization loss
    reg_loss += 0.5 * self.reg * np.sum(self.params['Wlast'] ** 2)
    for idx in xrange(self.num_fc_layers):
      W = self.params['Wfc' + np.str(idx+1)]
      reg_loss += 0.5 * self.reg * np.sum(W ** 2)
      
    for idx in xrange(self.num_cv_layers):
      W = self.params['Wcv' + np.str(idx+1)]
      reg_loss += 0.5 * self.reg * np.sum(W ** 2)
    
    #print 'Regularization loss: ', reg_loss
    loss = data_loss + reg_loss
    
    # Compute gradients
    
    # Gradient of lasst affine layer
    dout, dW, db = affine_backward(dscores, caches.pop())  
    grads['Wlast'] = dW
    grads['blast'] = db
    
    douts = []
    douts.append(dout)
    # Gradients through fully-connected layers
    for idx in reversed(xrange(self.num_fc_layers)):
      dout = dropout_backward(douts.pop(), caches.pop())
      douts.append(dout)
      if self.use_batchnorm:
        dout, dW, db, dgamma, dbeta = affine_batchnorm_relu_backward(douts.pop(), caches.pop())
        grads['gammafc' + np.str(idx+1)] = dgamma
        grads['betafc' + np.str(idx+1)] = dbeta
      else:
        dout, dW, db = affine_relu_backward(douts.pop(), caches.pop())
      douts.append(dout)
      grads['Wfc' + np.str(idx+1)] = dW
      grads['bfc' + np.str(idx+1)] = db
      
    # Gradients through convolution layers    
    for idx in reversed(xrange(self.num_cv_layers)):
      dout = dropout_backward(douts.pop(), caches.pop())
      douts.append(dout)
      dout = max_pool_backward_fast(douts.pop(), caches.pop())
      douts.append(dout)
      dout = relu_backward(douts.pop(), caches.pop())
      douts.append(dout)
      if self.use_batchnorm:
        dout, dgamma, dbeta = spatial_batchnorm_backward(douts.pop(), caches.pop())
        douts.append(dout)
        grads['gammacv' + np.str(idx+1)] = dgamma
        grads['betacv' + np.str(idx+1)] = dbeta
      dout, dW, db = conv_backward_fast(douts.pop(), caches.pop())
      douts.append(dout)
      grads['Wcv' + np.str(idx+1)] = dW
      grads['bcv' + np.str(idx+1)] = db
    
    # Gradient on X
    dX = douts.pop()
    
    # Regularization loss gradients
    for idx in xrange(self.num_cv_layers):
      grads['Wcv' + np.str(idx+1)] += self.reg * self.params['Wcv' + np.str(idx+1)]
    
    for idx in xrange(self.num_fc_layers):
      grads['Wfc' + np.str(idx+1)] += self.reg * self.params['Wfc' + np.str(idx+1)]
        
    # End of backward pass
    
    return loss, grads
    


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    
    C, H, W = input_dim
    
    self.params['W1'] = np.random.normal(0, weight_scale, [num_filters, C, filter_size, filter_size])
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = np.random.normal(0, weight_scale, [num_filters * H * W / 4, hidden_dim])
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
    self.params['b3'] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out2, cache2 = affine_relu_forward(out1, W2, b2)
    scores, cache3 = affine_forward(out2, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) + np.sum(W3 ** 2))
    
    loss = data_loss + reg_loss
    dout2, dW3, db3 = affine_backward(dscores, cache3)
    dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
    dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)
    
    dW3 += self.reg * W3
    dW2 += self.reg * W2
    dW1 += self.reg * W1
    
    grads['W3'] = dW3
    grads['b3'] = db3
    grads['W2'] = dW2
    grads['b2'] = db2
    grads['W1'] = dW1
    grads['b1'] = db1
    
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
