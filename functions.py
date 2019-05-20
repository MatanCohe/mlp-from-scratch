import numpy as np
from scipy.special import softmax
from collections import namedtuple

column_wise_softmax = lambda x: softmax(x, axis=0)

activation_function = namedtuple('activation_function', ['f', 'derivative'])

loss_function = namedtuple('loss_function', ['f', 'delta'])

def relu(x):
    """Return relu(x)"""
    return np.maximum(x, 0)

def relu_derivative(x):
    """Return relue'(x)"""
    d = np.copy(x)
    d[x<=0] = 0
    d[x>0] = 1
    return d

def sigmoid(x):
    """Return sigmoid(x)"""
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def sigmoid_derivative(x):
    """Returns sigmoid'(x)"""
    sigm = 1. / (1. + np.exp(-x))
    return sigm * (1. - sigm)


def tanh(x):
    tanh = np.divide(np.exp(x) - np.exp(-x), (np.exp(x) + np.exp(-x)))
    return tanh

def tanh_derivative(x):
    deriv = 1 - np.power(tanh(x), 2)
    return deriv



sigmoid_activation = activation_function(f=sigmoid, derivative=sigmoid_derivative)

relu_activation = activation_function(f=relu, derivative=relu_derivative)

mse_loss = loss_function(f=mse, delta=delta_mse)

cross_entropy_loss = loss_function(f=cross_entropy, delta=delta_cross_entropy)

tanh_activation = activation_function(f=tanh, derivative=tanh_derivative)