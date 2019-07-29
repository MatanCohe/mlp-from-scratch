import os
import pytest

from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis import given, example
from hypothesis import strategies as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import correlate2d, correlate


from cs231n import fast_layers
from cs231n import layers

from convolution import convolution, backword

    

@given(arrays(np.float,
              array_shapes(min_dims=2, max_dims=2, 
                           min_side=1, max_side=32),
              elements=st.floats(-1e6, 1e6)),
       arrays(np.float, (3, 3), elements=st.floats(0, 1)))
@example(np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]]),
         np.ones((3, 3)))
def test_conv(x, kernel):
    assert np.allclose(convolution.conv2d(x, kernel), correlate2d(x, kernel, mode='same'))
    
def test_nconv(sample_data):
    new_x = np.expand_dims(sample_data, axis=0)
    kernel = np.ones((1, 3, 3, 3))
    expected, mem = fast_layers.conv_forward_im2col(new_x, kernel, np.array([0]), {'pad': 1, 'stride': 1})
    res = convolution.conv4d(new_x, kernel)
    assert np.allclose(res, expected)

def test_back_conv(sample_data):
    new_x = np.expand_dims(sample_data, axis=0)
    kernel = np.ones((1, 1, 3, 3))
    out, cache = layers.conv_forward_naive(new_x[:, 0:1, :, :], 
                                                kernel, np.array([0]), 
                                                {'pad': 1, 'stride': 1})
     
    expected_dx, expected_dw, expected_db = layers.conv_backward_naive(np.ones((1, 1, 32, 32)), cache)
    dx, dw, db = backword.backward_conv2d(new_x[0, 0, :, :], np.ones((32, 32)), kernel[0, 0, :, :])
    assert np.allclose(db, expected_db)
    assert np.allclose(dw, expected_dw)
    assert np.allclose(dx, expected_dx)

@pytest.fixture
def sample_data():
    data_point_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data_point.csv')
    x = pd.read_csv(data_point_path, header=None).values
    y, x = x[:, :1], x[:, 1:]
    x = x.reshape(3, 32, 32)    
    return x