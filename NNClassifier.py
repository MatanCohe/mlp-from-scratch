import numpy as np

class NeuralNetworkClassifier:
    """This class represents a Neural Network classifier"""

    def __init__(self, layers_sizes, learning_rate, loss_function):
        """Create a neural network classifier.

        :param layers_sizes: An array of integers.
        :param learning_rate: scalar.
        :param loss_function: String.

        :return: A neural network with the specified requirements.
        """
        pass

    def train(self, x, y):
        """Train the classifier.

        Train the classifier over the examples from x and labels for y.
        The ith example in x should match the ith label in y when iterating over them.

        :param x: Training examples.
        :param y: Labels.
        :return: Trained classifier over the given data.
        """
        pass

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
    def __init__(self, weights_matrix, bias, activation_function, activation_function_derivative):
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
        pass

    def weights_update(self):
        """Update the layer weights and bias"""
        pass