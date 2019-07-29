import numpy as np


from convolution import convolution

def backward_conv2d(x, dout, kernel):
    """
    Calculate the gradient dx, dw, db.
    
    x - pervious layer input.
    dout - error.
    kernel...
    returns:
        dx - current x error.
        dw - kernel error.
        db - bias error.
    """
    assert x.shape == dout.shape
    assert x.ndim == kernel.ndim
    flipped_kernel = np.flipud(np.fliplr(kernel))
    dx = convolution.conv2d(dout, flipped_kernel)
    db = dout.sum()
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
    return dx, dw, db