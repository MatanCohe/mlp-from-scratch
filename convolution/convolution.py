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


def conv3d(x, kernal):
    assert x.ndim == kernal.ndim == 3
    assert x.shape[0] == kernal.shape[0]
    accum = list()
    for dim, (c_x, c_k) in enumerate(zip(x, kernal)):
        accum.append(conv2d(c_x, c_k))
    res = np.stack(accum, axis=0).sum(axis=0)
    return res



def conv4d(x, seq_kernals):
    assert x.ndim == seq_kernals.ndim
    accum = list()
    for example in x:
        feature_maps = list()
        for kernal in seq_kernals:
            feature_maps.append(conv3d(example, kernal))
        if len(feature_maps) < 2:
            accum.append(np.array(feature_maps))
        else:
            accum.append(np.concatenate(feature_maps, axis=0))
    if len(accum) < 2:
        out = np.array(accum)
    else:
        out = np.concatenate(accum, axis=0)
    assert out.shape[0] == x.shape[0]
    return out
        