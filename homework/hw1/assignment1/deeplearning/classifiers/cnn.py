import numpy as np

from deeplearning.layers import *
from deeplearning.fast_layers import *
from deeplearning.layer_utils import *


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
    F = num_filters
    HH = WW = filter_size
    
    conv_stride = 1
    pad = (filter_size - 1) / 2
    H_conv = 1 + (H + 2 * pad - HH) / conv_stride
    W_conv = 1 + (W + 2 * pad - WW) / conv_stride 

    pool_height = pool_width = 2
    pool_stride = 2
    H_pool = 1 + (H_conv - pool_height) / pool_stride
    W_pool = 1 + (W_conv - pool_width) / pool_stride

    self.params["W1"] = weight_scale * np.random.randn(num_filters, C, HH, WW)
    self.params["b1"] = np.zeros(num_filters)

    self.params["W2"] = weight_scale * np.random.randn(F * H_pool * W_pool, hidden_dim)
    self.params["b2"] = np.zeros(hidden_dim)

    self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b3"] = np.zeros(num_classes)
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
    out, cache_conv_relu_pool = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out, cache_affine_relu = affine_relu_forward(out, W2, b2)
    out, cache_affine = affine_forward(out, W3, b3)
    scores = out
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
    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2) + np.sum(self.params["W3"] ** 2))
    dx3, dw3, db3 = affine_backward(dx, cache_affine)
    dw3 += self.reg * self.params["W3"]
    grads["W3"] = dw3
    grads["b3"] = db3
    dx2, dw2, db2 = affine_relu_backward(dx3, cache_affine_relu)
    dw2 += self.reg * self.params["W2"]
    grads["W2"] = dw2
    grads["b2"] = db2
    _, dw1, db1 = conv_relu_pool_backward(dx2, cache_conv_relu_pool)
    dw1 += self.reg * self.params["W1"]
    grads["W1"] = dw1
    grads["b1"] = db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


def conv_relu_pool_batchnorm_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param, use_leakyReLU=False):
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
  s, relu_cache = leaky_relu_forward(a) if use_leakyReLU else relu_forward(a)
  p, pool_cache = max_pool_forward_fast(s, pool_param)
  out, batchnorm_cache = spatial_batchnorm_forward(p, gamma, beta, bn_param)
  cache = (conv_cache, relu_cache, pool_cache, batchnorm_cache)
  return out, cache


def conv_relu_pool_batchnorm_backward(dout, cache, use_leakyReLU=False):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache, batchnorm_cache = cache
  dp, dgamma, dbeta = spatial_batchnorm_backward(dout, batchnorm_cache)
  ds = max_pool_backward_fast(dp, pool_cache)
  da = leaky_relu_backward(ds, relu_cache) if use_leakyReLU else relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db, dgamma, dbeta

  
class MyConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  [conv - relu - 2x2 max pool - batchnorm] x 2 - [affine - relu] x 2 - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=64, filter_size=5,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32, use_leakyReLU=False):
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
    self.use_leakyReLU = use_leakyReLU
    
    C, H, W = input_dim
    F = num_filters
    HH = WW = filter_size
    
    conv1_stride = 1
    pad1 = (filter_size - 1) / 2
    H_conv1 = 1 + (H + 2 * pad1 - HH) / conv1_stride
    W_conv1 = 1 + (W + 2 * pad1 - WW) / conv1_stride 

    pool_height = pool_width = 2
    pool1_stride = 2
    H_pool1 = 1 + (H_conv1 - pool_height) / pool1_stride
    W_pool1 = 1 + (W_conv1 - pool_width) / pool1_stride

    self.params["W1"] = weight_scale * np.random.randn(num_filters, C, HH, WW)
    self.params["b1"] = np.zeros(num_filters)

    self.params["gamma1"] = np.ones(F)
    self.params["beta1"] = np.zeros(F)

    conv2_stride = 1
    pad2 = (filter_size - 1) / 2
    H_conv2 = 1 + (H_pool1 + 2 * pad2 - HH) / conv2_stride
    W_conv2 = 1 + (W_pool1 + 2 * pad2 - WW) / conv2_stride 

    pool_height = pool_width = 2
    pool2_stride = 2
    H_pool2 = 1 + (H_conv2 - pool_height) / pool2_stride
    W_pool2 = 1 + (W_conv2 - pool_width) / pool2_stride

    self.params["W2"] = weight_scale * np.random.randn(num_filters, F, HH, WW)
    self.params["b2"] = np.zeros(num_filters)

    self.params["gamma2"] = np.ones(F)
    self.params["beta2"] = np.zeros(F)

    self.params["W3"] = weight_scale * np.random.randn(F * H_pool2 * W_pool2, hidden_dim)
    self.params["b3"] = np.zeros(hidden_dim)

    self.params["W4"] = weight_scale * np.random.randn(hidden_dim, hidden_dim)
    self.params["b4"] = np.zeros(hidden_dim)

    self.params["W5"] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params["b5"] = np.zeros(num_classes)

    self.bn_params = [{'mode': 'train'} for i in xrange(4)]
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
    W4, b4 = self.params['W4'], self.params['b4']
    W5, b5 = self.params['W5'], self.params['b5']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    mode = 'test' if y is None else 'train'
    for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out, cache_conv_relu_pool_batchnorm1 = conv_relu_pool_batchnorm_forward(X, W1, b1, conv_param, pool_param, self.params["gamma1"], self.params["beta1"], bn_param, self.use_leakyReLU)
    out, cache_conv_relu_pool_batchnorm2 = conv_relu_pool_batchnorm_forward(out, W2, b2, conv_param, pool_param, self.params["gamma2"], self.params["beta2"], bn_param, self.use_leakyReLU)
    out, cache_affine_relu3 = affine_relu_forward(out, W3, b3)
    out, cache_affine_relu4 = affine_relu_forward(out, W4, b4)
    out, cache_affine = affine_forward(out, W5, b5)
    scores = out
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
    loss, dx = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2) + np.sum(self.params["W3"] ** 2) + np.sum(self.params["W4"] ** 2) + np.sum(self.params["W5"] ** 2))
    dx5, dw5, db5 = affine_backward(dx, cache_affine)
    dw5 += self.reg * self.params["W5"]
    grads["W5"] = dw5
    grads["b5"] = db5
    dx4, dw4, db4 = affine_relu_backward(dx5, cache_affine_relu4)
    dw4 += self.reg * self.params["W4"]
    grads["W4"] = dw4
    grads["b4"] = db4
    dx3, dw3, db3 = affine_relu_backward(dx4, cache_affine_relu3)
    dw3 += self.reg * self.params["W3"]
    grads["W3"] = dw3
    grads["b3"] = db3
    dx2, dw2, db2, dgamma2, dbeta2 = conv_relu_pool_batchnorm_backward(dx3, cache_conv_relu_pool_batchnorm2, self.use_leakyReLU)
    dw2 += self.reg * self.params["W2"]
    grads["W2"] = dw2
    grads["b2"] = db2
    grads["gamma2"] = dgamma2
    grads["beta2"] = dbeta2
    _, dw1, db1, dgamma1, dbeta1 = conv_relu_pool_batchnorm_backward(dx2, cache_conv_relu_pool_batchnorm1, self.use_leakyReLU)
    dw1 += self.reg * self.params["W1"]
    grads["W1"] = dw1
    grads["b1"] = db1
    grads["gamma1"] = dgamma1
    grads["beta1"] = dbeta1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads


def data_augmentation(data):

  data_augmented = {}

  for k, v in data.iteritems():
    if k.startswith('X'):
      dataset = v.copy()
      num = v.shape[0]

      for n in range(num):
        flip_rotate = np.random.randint(3)

        if flip_rotate == 1:
          left_right = np.random.randint(2) + 1
          dataset[n, :, :, :] = np.flip(v[n, :, :, :].copy(), axis=left_right)

        elif flip_rotate == 2:
          rotate_num = np.random.randint(3) + 1
          dataset[n, :, :, :] = np.rot90(v[n, :, :, :].copy(), rotate_num, (1, 2))

      data_augmented[k] = dataset

    else:
      data_augmented[k] = v

  return data_augmented
