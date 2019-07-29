import numpy as np


def backward_conv(x, dout, kernal):
    """
    Calculate the gradient dx, dw, db.
    
    x - pervious layer input.
    dout - error.
    kernal...
    returns:
        dx - current x error.
        dw - kernal error.
        db - bias error.
    """
    assert x.shape == dout.shape
    assert x.ndim == kernal.ndim
    flipped_kernal = np.flipud(np.fliplr(kernal))
    dx = conv2d(dout, flipped_kernal)
    db = dout.sum()
    h, w = dout.shape
    k_h, k_w = kernal.shape
    dw = np.zeros_like(kernal)
    p = int(np.floor(np.divide(k_h, 2)))
    q = int(np.floor(np.divide(k_w, 2)))
    padded_x = np.pad(x, ((p, p), (q, q)), mode='constant')
    padded_dout = np.pad(dout, ((p, p), (q, q)), mode='constant')
    for i in range(k_h):
        for j in range(k_w):
            dw[i, j] = padded_x[i:h+i, j:w+j].reshape(-1, ).dot(padded_dout[i:h+i, j:w+j].reshape(-1, ))
    return dx, dw, db