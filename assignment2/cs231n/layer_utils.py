from cs231n.layers import *
from cs231n.fast_layers import *

def conv_relu_forward(x, w, b, conv_param, dropout=1.0):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  dropout_mask = (np.random.rand(*out.shape) < dropout) / dropout
  out *= dropout_mask
  cache = (conv_cache, relu_cache, dropout_mask)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache, dropout_mask = cache
  dout *= dropout_mask
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param, dropout=1.0, leak=0.0):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = leaky_relu_forward(a, leak)
  dropout_mask = (np.random.rand(*s.shape) < dropout) / dropout
  s *= dropout_mask
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache, dropout_mask)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache, dropout_mask = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  ds *= dropout_mask
  da = leaky_relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def affine_relu_forward(x, w, b, dropout=1.0, leak=0.0):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = leaky_relu_forward(a, leak)
  dropout_mask = (np.random.rand(*out.shape) < dropout) / dropout
  out *= dropout_mask
  cache = (fc_cache, relu_cache, dropout_mask)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache, dropout_mask = cache
  dout *= dropout_mask
  da = leaky_relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db
