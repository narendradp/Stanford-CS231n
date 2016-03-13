import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *

def four_layer_convnet(X, model, y=None, reg=0.0, dropout=1.0, leak=0.0):
  """
  Compute the loss and gradient for a simple 4 layer ConvNet.
  
  The network architecture is:
  Input->(Conv->ReLU->pool)->(Conv->ReLU->pool)->(FC->ReLU)->(FC)->scores/softmax
  
  - All pool layers use max pooling of size 2*2 and stride of 2
  - All conv layers use stride of 1
  - ReLU layers can be configured as leaky
  - Dropout layers can be configured after any ReLU layer in the network
  - L2 regularization loss is used for all conv layers and affine layers
  
  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. The 4-layer Convnet
    expects the model to have the following parameters:
      - W1, b1: Weights and biases for conv layer 1
      - W2, b2: Weights and biases for conv layer 2
      - W3, b3: Weights and biases for fully connected layer 1
      - W4, b4: Weights and biases for fully connected layer 2
  - y: vector of ground truth labels of shape (N,). y[i] gives the label 
    for the point X[i]
  - reg: regularization strength. Default is 0
  - dropout: Dropout probability for dropout layers
  - leak: Leak strength for leaky ReLU activation layer
  
  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c
  
  If y not None, then returns:
  - loss: scalar value giving the softmax loss
  - grads: dictionary with the same keys as model, mapping parameter names to their
    gradients    
   
  """

  # Unpack ConvNet parameters
  W1, b1 = model['W1'], model['b1']
  W2, b2 = model['W2'], model['b2']
  W3, b3 = model['W3'], model['b3']
  W4, b4 = model['W4'], model['b4']
  N, C, H, W = X.shape
  
  # Check conv filter sizes
  cv_layer1_filter_height, cv_layer1_filter_width = W1.shape[2:]
  cv_layer2_filter_height, cv_layer2_filter_width = W2.shape[2:]
  assert cv_layer1_filter_height == cv_layer1_filter_width, 'Conv filter-1 must be square'
  assert cv_layer2_filter_height == cv_layer2_filter_width, 'Conv filter-2 must be square'
  assert cv_layer1_filter_height % 2 == 1, 'Conv-1 filter size must be odd'
  assert cv_layer2_filter_height % 2 == 1, 'Conv-2 filter size must be odd'
  
  # Define conv layer parameters and pool layer parameters
  conv1_param = {'stride':1, 'pad': (cv_layer1_filter_height - 1) / 2}
  conv2_param = {'stride':1, 'pad': (cv_layer2_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
  
  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv1_param, pool_param, leak=leak)
  a2, cache2 = conv_relu_pool_forward(a1, W2, b2, conv2_param, pool_param, leak=leak)
  a3, cache3 = affine_relu_forward(a2, W3, b3, dropout=dropout, leak=leak)
  scores, cache4 = affine_forward(a3, W4, b4)
  
  if y is None:
    return scores
    
  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)
  
  # Compute regularization loss
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2, W3, W4])
  
  # Compute the gradients
  da3, dW4, db4 = affine_backward(dscores, cache4)
  da2, dW3, db3 = affine_relu_backward(da3, cache3)
  da1, dW2, db2 = conv_relu_pool_backward(da2, cache2)
  dX, dW1, db1 = conv_relu_pool_backward(da1, cache1)
  
  # Add regularization to gradients
  dW1 += reg * W1
  dW2 += reg * W2
  dW3 += reg * W3
  dW4 += reg * W4
  
  loss = data_loss + reg_loss
  
  grads = {}
  grads['W1'] = dW1
  grads['W2'] = dW2
  grads['W3'] = dW3
  grads['W4'] = dW4
  grads['b1'] = db1
  grads['b2'] = db2
  grads['b3'] = db3
  grads['b4'] = db4
  
  #print dscores
  #print data_loss
  #print reg_loss
  #print W4
  #print dW4
    
  return loss, grads
  

def two_layer_convnet(X, model, y=None, reg=0.0, dropout=1.0, leak=0.0):
  """
  Compute the loss and gradient for a simple two-layer ConvNet. The architecture
  is conv-relu-pool-affine-softmax, where the conv layer uses stride-1 "same"
  convolutions to preserve the input size; the pool layer uses non-overlapping
  2x2 pooling regions. We use L2 regularization on both the convolutional layer
  weights and the affine layer weights.

  Inputs:
  - X: Input data, of shape (N, C, H, W)
  - model: Dictionary mapping parameter names to parameters. A two-layer Convnet
    expects the model to have the following parameters:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the affine layer
  - y: Vector of labels of shape (N,). y[i] gives the label for the point X[i].
  - reg: Regularization strength.

  Returns:
  If y is None, then returns:
  - scores: Matrix of scores, where scores[i, c] is the classification score for
    the ith input and class c.

  If y is not None, then returns a tuple of:
  - loss: Scalar value giving the loss.
  - grads: Dictionary with the same keys as model, mapping parameter names to
    their gradients.
  """
  
  # Unpack weights
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  N, C, H, W = X.shape

  # We assume that the convolution is "same", so that the data has the same
  # height and width after performing the convolution. We can then use the
  # size of the filter to figure out the padding.
  conv_filter_height, conv_filter_width = W1.shape[2:]
  assert conv_filter_height == conv_filter_width, 'Conv filter must be square'
  assert conv_filter_height % 2 == 1, 'Conv filter height must be odd'
  assert conv_filter_width % 2 == 1, 'Conv filter width must be odd'
  conv_param = {'stride': 1, 'pad': (conv_filter_height - 1) / 2}
  pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

  # Compute the forward pass
  a1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param, dropout, leak)
  scores, cache2 = affine_forward(a1, W2, b2)

  if y is None:
    return scores

  # Compute the backward pass
  data_loss, dscores = softmax_loss(scores, y)

  # Compute the gradients using a backward pass
  da1, dW2, db2 = affine_backward(dscores, cache2)
  dX,  dW1, db1 = conv_relu_pool_backward(da1, cache1)

  # Add regularization
  dW1 += reg * W1
  dW2 += reg * W2
  reg_loss = 0.5 * reg * sum(np.sum(W * W) for W in [W1, W2])

  loss = data_loss + reg_loss
  grads = {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}
  
  return loss, grads


def init_two_layer_convnet(weight_scale=1e-3, bias_scale=0, input_shape=(3, 32, 32),
                           num_classes=10, num_filters=32, filter_size=5):
  """
  Initialize the weights for a two-layer ConvNet.

  Inputs:
  - weight_scale: Scale at which weights are initialized. Default 1e-3.
  - bias_scale: Scale at which biases are initialized. Default is 0.
  - input_shape: Tuple giving the input shape to the network; default is
    (3, 32, 32) for CIFAR-10.
  - num_classes: The number of classes for this network. Default is 10
    (for CIFAR-10)
  - num_filters: The number of filters to use in the convolutional layer.
  - filter_size: The width and height for convolutional filters. We assume that
    all convolutions are "same", so we pick padding to ensure that data has the
    same height and width after convolution. This means that the filter size
    must be odd.

  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
    - W1, b1: Weights and biases for the convolutional layer
    - W2, b2: Weights and biases for the fully-connected layer.
  """
  C, H, W = input_shape
  assert filter_size % 2 == 1, 'Filter size must be odd; got %d' % filter_size

  model = {}
  model['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
  model['b1'] = bias_scale * np.random.randn(num_filters)
  model['W2'] = weight_scale * np.random.randn(num_filters * H * W / 4, num_classes)
  model['b2'] = bias_scale * np.random.randn(num_classes)
  return model


def init_four_layer_convnet(weight_scale=8e-3, bias_scale=8e-3, input_shape=(3,32,32),
                            num_classes=10, num_cv_layer1_filters=32, 
                            cv_layer1_filter_size=5, num_cv_layer2_filters=32, 
                            cv_layer2_filter_size=5, num_fc_layer1=24):
  """
  Initialize the weights for a 4-layer ConvNet.
  
  Network architecture:
  Input -> (Conv->ReLU->pool) -> (Conv->ReLU->pool)->(FC->ReLU)->(FC)->scores
  
  Inputs:
  - weigth_scale: Scale at which the weights are initialized. Default 1e-3
  - bias_scale: Scale at which the biases are initialized. Default is 0
  - input_shape: Tuple giving the input shape to the network; default is 
    (3,32,32) for CIFAR-10
  - num_classes: The number of output classes for this network. Default is 10
    (for CIFAR-10)
  - num_cv_layer1_filters: Number of filters for conv layer 1. Default is 32
  - cv_layer1_filter_size: The width and height for conv filters for conv layer 1. Default
    is 5
  - num_cv_layer2_filters: Number of filters for conv layer 2. Default is 24
  - cv_layer2_filter_size: The width and height for conv filters for conv layer 2. Default
    is 3
  - num_fc_layer1: Number of units in fully connected layer 1. Default is 64
  
  Returns:
  A dictionary mapping parameter names to numpy arrays containing:
  - W1, b1: Weights and biases for conv layer 1
  - W2, b2: Weights and biases for conv layer 2
  - W3, b3: Weights and biases for fully connected layer 1
  - W4, b4: Weights and biases for fully connected layer 2
  """
  
  C, H, W = input_shape
  assert cv_layer1_filter_size % 2 == 1, 'Filter size must be odd; got %d' % cv_layer1_filter_size
  assert cv_layer2_filter_size % 2 == 1, 'Filter size must be odd; got %d' % cv_layer2_filter_size
  
  model = {}
  model['W1'] = weight_scale * np.random.randn(num_cv_layer1_filters, C, cv_layer1_filter_size, cv_layer1_filter_size)
  model['b1'] = bias_scale * np.random.randn(num_cv_layer1_filters)
  model['W2'] = weight_scale * np.random.randn(num_cv_layer2_filters, num_cv_layer1_filters, cv_layer2_filter_size, cv_layer2_filter_size)
  model['b2'] = bias_scale * np.random.randn(num_cv_layer2_filters)
  model['W3'] = weight_scale * np.random.randn(num_cv_layer2_filters * H * W / 16, num_fc_layer1)
  model['b3'] = bias_scale * np.random.randn(num_fc_layer1)
  model['W4'] = weight_scale * np.random.randn(num_fc_layer1, num_classes)
  model['b4'] = bias_scale * np.random.randn(num_classes)
  return model      


