import numpy as np

def affine_forward(x, w, b, dropout=1.0):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  x_reshape = np.reshape(x, (x.shape[0], -1))
  out = np.dot(x_reshape, w) + b
  dropout_mask = (np.random.rand(*out.shape) < dropout) / dropout
  out *= dropout_mask
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, dropout_mask)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b, dropout_mask = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dout *= dropout_mask
  x_reshape = np.reshape(x, (x.shape[0], -1))
  dx_reshape = np.dot(dout, w.T)
  dx = np.reshape(dx_reshape, x.shape)
  dw = np.dot(x_reshape.T, dout)
  db = np.sum(dout, axis=0)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout * (x > 0)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def leaky_relu_forward (x, leak=0.0):
  """
  Computes the forward pass for a layer of Leaky ReLUs
  
  Input:
  - x: Inputs, of any shape
  - leak: leak ratio for inputs of less than 0
  
  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x, leak
  """
  out = None
  out = np.maximum(0, x) + x * leak * (x < 0)
  cache = (x, leak)
  return out, cache
  
  
def leaky_relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of leaky ReLUs
  
  Input:
  - dout: Upstream derivates, of any shape
  - cache: Input x, of same shape as dout and leak ratio
  
  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, leak = cache
  dx = dout * (x > 0) + dout * leak * (x < 0)
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  pad = conv_param['pad']
  stride = conv_param['stride']
  (N, C, H, W) = x.shape
  (F, _, HH, WW) = w.shape
  
  x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  
  out_H = 1 + (H + 2*pad - HH) / stride
  out_W = 1 + (W + 2*pad - WW) / stride
  out = np.zeros([N,F,out_H,out_W])
  
  for f in range(F):
      W_f = w[f,:,:,:]
      for y_ind in range(out_H):
          for x_ind in range(out_W):
              x_depth_column = x_padded[:,:,stride*y_ind:stride*y_ind+HH,stride*x_ind:stride*x_ind+WW]
              out[:,f,y_ind,x_ind] = np.sum(W_f * x_depth_column, axis=(1,2,3)) + b[f]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  pad = conv_param['pad']
  stride = conv_param['stride']
  (N, C, H, W) = x.shape
  (F, _, HH, WW) = w.shape
  
  out_H = 1 + (H + 2*pad - HH) / stride
  out_W = 1 + (W + 2*pad - WW) / stride
  
  x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant')
  
  dx_padded = np.zeros_like(x_padded)
  dx = np.zeros_like(x)
  dw = np.zeros_like(w)
  db = np.zeros_like(b)
  
  for f in range(F):
      W_f = w[f,:,:,:]
      for y_ind in range(out_H):
          for x_ind in range(out_W):
              x_depth_column = x_padded[:,:,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW]
              dx_padded[:,:,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW] += np.reshape(dout[:,f,y_ind,x_ind], (N,1,1,1)) * np.tile(W_f, (N,1,1,1))
              dw[f,:,:,:] += np.sum(np.reshape(dout[:,f,y_ind,x_ind],(N,1,1,1)) * x_depth_column, axis=0)
              db[f] += np.sum(dout[:,f,y_ind,x_ind])  
  
  dx = dx_padded[:,:,pad:-pad,pad:-pad]
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  stride = pool_param['stride']
  (N, C, H, W) = x.shape
  
  out_H = 1 + (H - HH) / stride
  out_W = 1 + (W - WW) / stride
  
  out = np.zeros([N, C, out_H, out_W])
  for y_ind in range(out_H):
      for x_ind in range(out_W):
          for depth_ind in range(C):
              for data_ind in range(N):
                  out[data_ind,depth_ind,y_ind,x_ind] = np.amax(x[data_ind,depth_ind,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW])
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  (x, pool_param) = cache
  WW = pool_param['pool_width']
  HH = pool_param['pool_height']
  stride = pool_param['stride']
  N, C, H, W = x.shape
  
  dx = np.zeros_like(x)
  _, _, out_H, out_W = dout.shape
  for y_ind in range(out_H):
      for x_ind in range(out_W):
          for depth_ind in range(C):
              for data_ind in range(N):
                  max_val = np.amax(x[data_ind,depth_ind,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW])
                  dx[data_ind,depth_ind,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW] += \
                      dout[data_ind,depth_ind,y_ind,x_ind] * (x[data_ind,depth_ind,y_ind*stride:y_ind*stride+HH,x_ind*stride:x_ind*stride+WW] >= max_val)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

