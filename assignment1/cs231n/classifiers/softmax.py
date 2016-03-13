import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[0]
  num_train = X.shape[1]
  for i in xrange(num_train):
      correct_class = y[i]
      loss_i = 0.0
      for j in xrange(num_classes):
          loss_i += np.exp(W[j,:].dot(X[:,i]))
      for j in xrange(num_classes):
          dW[j,:] += (np.exp(W[j,:].dot(X[:,i])) / loss_i) * X[:,i].T
      loss_i = np.log(loss_i) - W[correct_class,:].dot(X[:,i])
      dW[correct_class,:] = dW[correct_class,:] - X[:,i].T
      loss += loss_i

  ## Data loss
  loss /= num_train
  dW /= num_train
  
  ## regularization loss
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = W.dot(X)
  scores_exp = np.exp(scores)
  
  loss = np.sum(np.log(np.sum(scores_exp, axis=0)))
  loss -= np.sum(scores[y,np.arange(num_train)])
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW = (scores_exp /  np.sum(scores_exp, axis=0)).dot(X.T)
  tmp = np.zeros_like(scores)
  tmp[y,np.arange(num_train)] = 1
  dW = dW - tmp.dot(X.T)
  dW /= num_train
  dW += reg * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
