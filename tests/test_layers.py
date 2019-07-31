import pytest

import numpy as np

import hypothesis
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays, array_shapes

import layers


@hypothesis.given(arrays(np.float, 
                        array_shapes(min_dims=4, max_dims=4, 
                                     min_side=1, max_side=32),
                        elements=st.floats(0, 1)))
def test_conv2linear_layer(x):
    layer = layers.Conv2Linear()
    assert np.array_equal(x, layer.backward(layer.forward(x)))