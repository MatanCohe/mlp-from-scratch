import numpy as np

class NeuralNetworkClassifier:
    """This class represents a Neural Network classifier"""

    def __init__(self, layers, learning_rate, loss_function):
        """Create a neural network classifier.

        :param layers: List or tuple of layers.
        :param learning_rate: scalar.
        :param loss_function: String.

        :return: A neural network with the specified requirements.
        """
        self.layers, self.alpha = layers, learning_rate
        self.loss = loss_function

    def train(self, x, y):
        """Train the classifier.

        Train the classifier over the examples from x and labels for y.
        self.loss_function = loss_function
        The ith example in x should match the ith label in y when iterating over them.

        :param x: Training examples.
        :param y: Labels.
        :return: Trained classifier over the given data.
        """
        for row, label in zip(x, y):
            label = label.reshape((len(label), 1))
            a = row.reshape(len(row), 1)
            for layer in self.layers:
                a = layer.forward(a)
            if self.loss == 'mse':
                diff = (a - label)
                delta =  diff * layer.activation_derivative(layer.z)
                layer.delta = delta
            else:
                raise ValueError('loss function not implemented')
            theta = layer.theta
            for layer in reversed(self.layers[:-1]):
                delta, theta = layer.backward(theta, delta), layer.theta
            a = x.transpose()
            for layer in self.layers:
                layer.weights_update(a)
                a = layer.a
        return self
    def predict(self, x):
        """Make prediction for x.

        :param x:
        :return:
        """
        pass

class Layer:
    """This class implements a single hidden layer of a Neural Network.

    With the forward method for the forward propagation step,
    backward for the back propagation step and weight update.

    """
    def __init__(self, weights_matrix, bias, activation_function, activation_function_derivative, learning_rate):
        """Create a hidden layer.

        :param weights_matrix: A weight matrix where the ith jth entry is the weight from the ith neuron
        in the previous layer to the jth neuron in the current layer.
        :param bias: Vector for the layer bias
        :param activation_function: Pointer to activation function.
        :param activation_function_derivative: Pointer to the activation function derivative.
        """
        self.theta = weights_matrix
        self.b = bias
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.alpha = learning_rate

    def forward(self, previous_layer_output):
        """Calculate the a value of the layer.

        :param previous_layer_output:
        :return: The a value.
        """
        a_prev = previous_layer_output
        z = np.dot(self.theta, a_prev) + self.b
        self.z, self.a = z, self.activation(z)

        return self.a

    def backward(self, next_layer_weights, next_layer_delta):
        """Calculate the delta value of the layer.

        :param next_layer_weights:
        :param next_layer_delta:
        :return: current layer delta.
        """
        delta = np.dot(next_layer_weights.T, next_layer_delta) * self.activation_derivative(self.z)
        self.delta = delta
        return delta

    def weights_update(self, previous_layer_output):
        """Update the layer weights and bias"""
        theta, b, delta, alpha = self.theta, self.b, self.delta, self.alpha
        dc_dtheta = np.outer(previous_layer_output, delta).transpose()
        new_theta = theta - alpha * dc_dtheta
        new_b = b - alpha * delta
        self.theta, self.b = new_theta, new_b