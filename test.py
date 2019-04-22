import pytest
import numpy as np

from NNClassifier import Layer
from functions import sigmoid_activation


def test_layer_forward():
    x = np.array([[1, 4, 5]]).transpose()
    t = np.array([[.1, .05]]).transpose()

    theta2 = np.array([[.1, .3, .5],
                       [.2, .4, .6]])
    b2 = np.array([[.5, .5]]).transpose()

    layer = Layer(weights_matrix=theta2, bias=b2, activation_function=sigmoid_activation.f, activation_function_derivative=sigmoid_activation.derivative)

    layer_output = layer.forward(previous_layer_output=x)

    expected_output = np.array([[.9866, .9950]]).transpose()

    assert np.allclose(layer_output, expected_output, rtol=0.0001)

def test_layer_backward():
    pass