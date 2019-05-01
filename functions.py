import numpy as np
from scipy.special import softmax
from collections import namedtuple

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

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    p = softmax(X)
    # We use multidimensional array indexing to extract
    # softmax probability of the correct label for each sample.
    # Refer to https://docs.scipy.org/doc/numpy/user/basics.indexing.html#indexing-multi-dimensional-arrays for understanding multidimensional array indexing.
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector.
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

def mse(x, y):
    return 0.5 * np.square(x - y).mean()

def delta_mse(x, y):
    return x - y



sigmoid_activation = activation_function(f=sigmoid, derivative=sigmoid_derivative)

relu_activation = activation_function(f=relu, derivative=relu_derivative)

mse_loss = loss_function(f=mse, delta=delta_mse)

cross_entropy_loss = loss_function(f=cross_entropy, delta=delta_cross_entropy)

tanh_activation = activation_function(f=tanh, derivative=tanh_derivative)