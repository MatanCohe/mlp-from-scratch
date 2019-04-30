from copy import deepcopy
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

    def train(self, x, y, number_of_epochs, validation_x=None, validation_y=None):
        """Train the classifier.

        Train the classifier over the examples from x and labels for y.
        self.loss_function = loss_function
        The ith example in x should match the ith label in y when iterating over them.

        :param x: Training examples.
        :param y: Labels.
        :return: Trained classifier over the given data.
        """
        train_epochs_errors = []
        validation_epochs_errors = []
        number_of_train_examples = x.shape[0]
        for epoch in range(number_of_epochs):
            curr_epoch_err = 0
            for row, label in zip(x, y):
                label = label.reshape((len(label), 1))
                a = row.reshape(len(row), 1)
                a, layer = self.forward_propagation(a)
                delta, err = self.calculate_loss(a, label, layer)
                curr_epoch_err += err
                theta = layer.theta
                self.backpropagation(delta, theta)
                a = x.transpose()
                self.update_network(a)
            train_epochs_errors.append(curr_epoch_err/number_of_train_examples)

            # test the network on the validation set
            validation_epochs_errors = []
            if not validation_x is None:
                validation_error = self.validate(validation_x, validation_y)
                # validation_y_hat, layer_val = self.forward_propagation(validation_x)
                #_, err_val = self.calculate_loss(validation_y_hat, validation_y, layer_val)
                validation_epochs_errors.append(validation_error)

        return train_epochs_errors, validation_epochs_errors

    def validate(self, x, y):
        y_hat, layer = self.forward_propagation(x)
        _, err = self.calculate_loss(y_hat, y, layer)
        return err.mean()

    def update_network(self, a):
        for layer in self.layers:
            layer.weights_update(a, self.alpha)
            a = layer.a

    def backpropagation(self, delta, theta):
        for layer in reversed(self.layers[:-1]):
            delta, theta = layer.backward(theta, delta), layer.theta

    def calculate_loss(self, a, label, layer):
        if self.loss == 'mse':
            diff = a - label
            delta = diff * layer.activation_derivative(layer.z)   # TODO delta might be a matrix in case of batch
            layer.delta = delta
            err = np.square(diff).mean(axis=0)  # axis 0 means the average of every col
        else:
            raise ValueError('loss function not implemented')
        return delta, err

    def forward_propagation(self, a):
        y_hat = a
        for layer in self.layers:
            y_hat = layer.forward(y_hat)
        return y_hat, layer

    def predict(self, x):
        """Make prediction for x.

        :param x: each row in x represents one training example
        :return:
        """
        a, _ = self.forward_propagation(x.transpose())
        a = a.transpose()
        return np.argmax(a, axis=1)


    def check_gradient(self, x, y):
        x = x.transpose()
        y = y.transpose()
        layers_copy = deepcopy(self.layers)
        epsilon = 10 ** -4
        a, layer = self.forward_propagation(x)
        delta, _ = self.calculate_loss(a, y, layer)
        self.backpropagation(delta=delta, theta=layer.theta)
        previous_layer_output = x
        for layer in self.layers:
            theta_copy = deepcopy(layer.theta)
            real_theta_size = theta_copy.shape
            delta = layer.delta
            dc_dtheta = np.outer(previous_layer_output, delta).transpose()
            previous_layer_output = layer.a
            R, C = theta_copy.shape
            for i in range(R):
                for j in range(C):
                    theta_plus = deepcopy(theta_copy)
                    theta_plus[i, j] += epsilon
                    layer.theta = theta_plus
                    a_plus, l_plus = self.forward_propagation(x)
                    _, err_plus = self.calculate_loss(a_plus, y, l_plus)
                    theta_minus = deepcopy(theta_copy)
                    theta_minus[i, j] -= epsilon
                    layer.theta = theta_minus
                    a_minus, l_minus = self.forward_propagation(x)
                    _, err_minus = self.calculate_loss(a_minus, y, l_minus)
                    limit = (err_plus - err_minus)/(2*epsilon)
                    # print(f'limit = {abs(limit)}')
                    # print(f'diff = {abs(dc_dtheta[i,j] - limit)}')
                    grad_diff = abs(dc_dtheta[i,j] - limit)
                    assert grad_diff < 10 ** -6, f"Diff {grad_diff} is too big."
                    layer.theta = theta_copy




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

        :param previous_layer_output: col vector or matrix where number of rows is number of features,
                number of cols is number f training examples
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

    def weights_update(self, previous_layer_output, learning_rate):
        """Update the layer weights and bias"""
        theta, b, delta, alpha = self.theta, self.b, self.delta, learning_rate
        dc_dtheta = np.outer(previous_layer_output, delta).transpose()
        new_theta = theta - alpha * dc_dtheta
        new_b = b - alpha * delta
        self.theta, self.b = new_theta, new_b