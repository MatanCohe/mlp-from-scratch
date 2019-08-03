import numpy as np
from scipy import signal

def backward_conv2d(x, dout, kernel):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - x: Layer input
    - dout: Upstream derivatives.
    - kernel: Weight kernel.   

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
     """
    assert x.shape == dout.shape
    assert x.ndim == kernel.ndim
    dx = signal.convolve2d(dout, kernel, mode='same')
    h, w = dout.shape
    k_h, k_w = kernel.shape
    dw = np.zeros_like(kernel)
    p = int(np.floor(np.divide(k_h, 2)))
    q = int(np.floor(np.divide(k_w, 2)))
    padded_x = np.pad(x, ((p, p), (q, q)), mode='constant')
    dw = signal.correlate2d(padded_x, dout, mode='valid')
    return dx, dw

def backward_conv3d(x, dout, kernel):
    assert x.ndim == kernel.ndim
    dx_accum, dw_accum, db_accum = [], [], []
    for ch, k_c in zip(x, kernel):
        dx, dw = backward_conv2d(ch, dout, k_c)
        dx_accum.append(dx)
        dw_accum.append(dw)
    dx = np.stack(dx_accum, axis=0)
    dw = np.stack(dw_accum, axis=0)
    db = dout.sum()
    return dx, dw, db

def backward_conv4d(x, dout, seq_kernels):
    dx_accum, dw_accum, db_accum = [], [], []
    for kernel, kernel_dout in zip(seq_kernels, dout):
        dx, dw, db = backward_conv3d(x, kernel_dout, kernel)
        dx_accum.append(dx)
        dw_accum.append(dw)
        db_accum.append(db)
    dx = np.stack(dx_accum, axis=0).sum(axis=0)
    dw = np.stack(dw_accum, axis=0)
    db = np.stack(db_accum, axis=0)
    return dx, dw, db

def backward_conv5d(x, dout, seq_kernels):
    assert x.ndim == dout.ndim
    assert x.shape[0] == dout.shape[0]
    dx_accum, dw_accum, db_accum = [], [], []
    for example, error in zip(x, dout):
        dx, dw, db = backward_conv4d(example, error, seq_kernels)
        dx_accum.append(dx)
        dw_accum.append(dw)
        db_accum.append(db)
    dx = np.stack(dx_accum, axis=0)
    dw = np.stack(dw_accum, axis=0).sum(axis=0)
    db = np.stack(db_accum, axis=0).sum(axis=0)
    return dx, dw, db
    