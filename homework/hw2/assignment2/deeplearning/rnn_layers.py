from __future__ import print_function, division
from builtins import range
import numpy as np


"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """
    next_h, cache = None, None
    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    pre_next_h = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
    next_h = np.tanh(pre_next_h)
    cache = (x, prev_h, Wx, Wh, b, next_h)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.

    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass

    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    x, prev_h, Wx, Wh, b, next_h = cache

    dnext_mul_dtanh = dnext_h * (1 - next_h ** 2)
    dx = np.matmul(dnext_mul_dtanh, Wx.T)
    dprev_h = np.matmul(dnext_mul_dtanh, Wh.T)
    dWx = np.matmul(x.T, dnext_mul_dtanh)
    dWh = np.matmul(prev_h.T, dnext_mul_dtanh)
    db = np.sum(dnext_mul_dtanh, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    N, T, _ = x.shape
    _, H = h0.shape

    h = np.zeros( (N, T, H), dtype="float")
    h_i = h0
    cache = []
    for i in range(T):
        x_i = x[:, i, :]
        h_i, cache_i = rnn_step_forward(x_i, h_i, Wx, Wh, b)
        h[:, i, :] = h_i
        cache.append(cache_i)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    N, T, _ = dh.shape
    _, D = cache[0][0].shape

    dx = np.zeros( (N, T, D), dtype="float" )
    dh0 = dWx = dWh = db = 0
    for i in range(T - 1, -1, -1):
        dx_i, dh_i, dWx_i, dWh_i, db_i = rnn_step_backward(dh[:, i, :], cache[i])
        dx[:, i, :] += dx_i
        dh0 += dh_i if i == 0 else 0
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        
        dh_j = dh_i
        for j in range(i - 1, -1, -1):
            dx_j, dh_j, dWx_j, dWh_j, db_j = rnn_step_backward(dh_j, cache[j])
            dx[:, j, :] += dx_j
            dh0 += dh_j if j == 0 else 0
            dWx += dWx_j
            dWh += dWh_j
            db += db_j
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    ##############################################################################
    # TODO: Implement the forward pass for word embeddings.                      #
    #                                                                            #
    # HINT: This can be done in one line using NumPy's array indexing.           #
    ##############################################################################
    out, cache = W[x], (x, W)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return out, cache


def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.

    HINT: Look up the function np.add.at

    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass

    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    ##############################################################################
    # TODO: Implement the backward pass for word embeddings.                     #
    #                                                                            #
    # Note that Words can appear more than once in a sequence.                   #
    # HINT: Look up the function np.add.at                                       #
    ##############################################################################
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dW


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.

    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.

    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)

    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    next_h, next_c, cache = None, None, None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    #############################################################################
    _, H = prev_h.shape

    activation = np.matmul(x, Wx) + np.matmul(prev_h, Wh) + b
    a_i = activation[:, : H]
    a_f = activation[:, H : 2 * H]
    a_o = activation[:, 2 * H : 3 * H]
    a_g = activation[:, 3 * H :]

    i = sigmoid(a_i)
    f = sigmoid(a_f)
    o = sigmoid(a_o)
    g = np.tanh(a_g)

    next_c = f * prev_c + i * g
    next_h = o * np.tanh(next_c)

    cache = x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    _, H = dnext_h.shape

    x, prev_h, prev_c, Wx, Wh, b, i, f, o, g, next_h, next_c = cache
    tanh_next_c = np.tanh(next_c)
    dtanh_next_c_dnext_c = 1 - tanh_next_c ** 2
    di_da_i = i * (1 - i)
    df_da_f = f * (1 - f)
    do_da_o = o * (1 - o)
    dg_da_g = 1 - g ** 2

    # Calculate dx
    dnext_h_mul_tanh_next_c_mul_do_da_o = dnext_h * tanh_next_c * do_da_o
    dx = np.matmul(dnext_h_mul_tanh_next_c_mul_do_da_o, Wx[:, 2 * H : 3 * H].T)
    pre_dc = dnext_h * o * dtanh_next_c_dnext_c + dnext_c
    pre_df = pre_dc * prev_c * df_da_f
    dx += np.matmul(pre_df, Wx[:, H : 2 * H].T)
    pre_di = pre_dc * g * di_da_i
    dx += np.matmul(pre_di, Wx[:, : H].T)
    pre_dg = pre_dc * i * dg_da_g
    dx += np.matmul(pre_dg, Wx[:, 3 * H :].T)

    # Calculate dprev_h
    dprev_h = np.matmul(dnext_h_mul_tanh_next_c_mul_do_da_o, Wh[:, 2 * H : 3 * H].T)
    dprev_h += np.matmul(pre_df, Wh[:, H : 2 * H].T)
    dprev_h += np.matmul(pre_di, Wh[:, : H].T)
    dprev_h += np.matmul(pre_dg, Wh[:, 3 * H :].T)

    # Calculate dprev_c
    dprev_c = pre_dc * f

    # Calculate dWx
    dWx = np.zeros_like(Wx)
    dWx[:, 2 * H : 3 * H] = np.matmul(x.T, dnext_h_mul_tanh_next_c_mul_do_da_o)
    dWx[:, H : 2 * H] = np.matmul(x.T, pre_df)
    dWx[:, : H] = np.matmul(x.T, pre_di)
    dWx[:, 3 * H :] = np.matmul(x.T, pre_dg)

    # Calculate dWh
    dWh = np.zeros_like(Wh)
    dWh[:, 2 * H : 3 * H] = np.matmul(prev_h.T, dnext_h_mul_tanh_next_c_mul_do_da_o)
    dWh[:, H : 2 * H] = np.matmul(prev_h.T, pre_df)
    dWh[:, : H] = np.matmul(prev_h.T, pre_di)
    dWh[:, 3 * H :] = np.matmul(prev_h.T, pre_dg)

    # Calculate db
    db = np.zeros_like(b)
    db[2 * H : 3 * H] = np.sum(dnext_h_mul_tanh_next_c_mul_do_da_o, axis=0)
    db[H : 2 * H] = np.sum(pre_df, axis=0)
    db[: H] = np.sum(pre_di, axis=0)
    db[3 * H :] = np.sum(pre_dg, axis=0)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.

    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.

    Inputs:
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    N, T, _ = x.shape
    _, H = h0.shape

    h = np.zeros( (N, T, H), dtype="float")
    h_i = h0
    c_i = np.zeros_like(h_i)
    cache = []
    for i in range(T):
        x_i = x[:, i, :]
        h_i, c_i, cache_i = lstm_step_forward(x_i, h_i, c_i, Wx, Wh, b)
        h[:, i, :] = h_i
        cache.append(cache_i)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    N, T, H = dh.shape
    _, D = cache[0][0].shape

    dx = np.zeros( (N, T, D), dtype="float" )
    dc_i = np.zeros( (N, H), dtype="float" )
    dh0 = dWx = dWh = db = 0
    for i in range(T - 1, -1, -1):
        dx_i, dh_i, dc_i, dWx_i, dWh_i, db_i = lstm_step_backward(dh[:, i, :], dc_i, cache[i])
        dx[:, i, :] += dx_i
        dh0 += dh_i if i == 0 else 0
        dWx += dWx_i
        dWh += dWh_i
        db += db_i
        
        dh_j = dh_i
        dc_j = np.zeros( (N, H), dtype="float" )
        for j in range(i - 1, -1, -1):
            dx_j, dh_j, dc_j, dWx_j, dWh_j, db_j = lstm_step_backward(dh_j, dc_j, cache[j])
            dx[:, j, :] += dx_j
            dh0 += dh_j if j == 0 else 0
            dWx += dWx_j
            dWh += dWh_j
            db += db_j
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.

    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)

    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.

    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.

    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.

    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
