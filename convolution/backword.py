import numpy as np


from convolution import convolution

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
    flipped_kernel = np.flipud(np.fliplr(kernel))
    dx = convolution.conv2d(dout, flipped_kernel)
    h, w = dout.shape
    k_h, k_w = kernel.shape
    dw = np.zeros_like(kernel)
    p = int(np.floor(np.divide(k_h, 2)))
    q = int(np.floor(np.divide(k_w, 2)))
    padded_x = np.pad(x, ((p, p), (q, q)), mode='constant')
    padded_dout = np.pad(dout, ((p, p), (q, q)), mode='constant')
    for i in range(k_h):
        for j in range(k_w):
            dw[i, j] = padded_x[i:h+i, j:w+j].reshape(-1, ).dot(padded_dout[i:h+i, j:w+j].reshape(-1, ))
    return dx, dw
