import numpy as np
import functools
from scipy import signal

def _conv2d(x, kernel):
    """
    Computes two dimensional correlation between x and kernel.
    The output is the same shape as x.
    Input:
    - x: Input data of shape (h, w)
    - w: Filter weights of shape (k_h, k_w)
    - b: Biases, of shape (F,)

    Returns a tuple of:
    - out: Output data.
    """
    assert x.ndim == kernel.ndim == 2
    h, w = x.shape
    k_h, k_w = kernel.shape
    p = int(np.floor(np.divide(k_h, 2)))
    q = int(np.floor(np.divide(k_w, 2)))
    paded_x = np.pad(x, ((p, p), (q, q)), mode='constant')
    flat_kernel = kernel.reshape(-1, )
    out = np.zeros(x.shape)
    for i in range(p, h+p):
        for j in range(p, w+p):
            window = paded_x[i-p:i+p+1, j-q:j+q+1]
            flat_window = window.reshape(-1, )
            out[i-p, j-q] = flat_window @ flat_kernel
    return out

conv2d = functools.partial(signal.correlate2d, mode='same')

def conv3d(x, kernel, bias):
    assert x.ndim == kernel.ndim == 3
    assert x.shape[0] == kernel.shape[0]
    accum = list()
    for dim, (c_x, c_k) in enumerate(zip(x, kernel)):
        accum.append(conv2d(c_x, c_k))
    res = np.sum(accum, axis=0) + bias
    res3d = np.expand_dims(res, axis=0)
    return res3d



def conv4d(x, seq_kernels, seq_bias):
    # TODO: Add bias.
    assert x.ndim == seq_kernels.ndim
    assert seq_kernels.shape[0] == seq_bias.shape[0]
    accum = list()
    for example in x:
        feature_maps = list()
        for kernel, bias in zip(seq_kernels, seq_bias):
            feature_maps.append(conv3d(example, kernel, bias))
        accum.append(np.concatenate(feature_maps, axis=0))
    out = np.array(accum)
    assert out.shape[0] == x.shape[0]
    return out
