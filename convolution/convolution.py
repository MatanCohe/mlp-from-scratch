import numpy as np

def conv2d(x, kernal):
    assert x.ndim == kernal.ndim == 2
    h, w = x.shape
    k_h, k_w = kernal.shape
    p = int(np.floor(np.divide(k_h, 2)))
    q = int(np.floor(np.divide(k_w, 2)))
    paded_x = np.pad(x, ((p, p), (q, q)), mode='constant')
    flat_kernal = kernal.reshape(-1, )
    out = np.zeros(x.shape)
    for i in range(p, h+p):
        for j in range(p, w+p):
            window = paded_x[i-p:i+p+1, j-q:j+q+1]
            flat_window = window.reshape(-1, )
            out[i-p, j-q] = flat_window @ flat_kernal
    return out