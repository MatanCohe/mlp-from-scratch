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

    def train(self, x, y, number_of_epochs, validation_x=None, validation_y=None, batch_size=1):
        """Train the classifier.

        Train the classifier over the examples from x and labels for y.
        self.loss_function = loss_function
        The ith example in x should match the ith label in y when iterating over them.

        :param x: Training examples.
        :param y: Labels.
        :return: Trained classifier over the given data.
        """
        train_epochs_errors = []
        train_epochs_acc = []
        validation_errors = []
        validation_acc = []
        batch_correct_predictions = 0
        number_of_train_examples = x.shape[0]
        for epoch in range(number_of_epochs):
            curr_epoch_err = 0
            epoch_correct_predictions = 0
            number_of_batches = number_of_train_examples / batch_size
            x_batches = np.array_split(x, number_of_batches)
            y_batches = np.array_split(y, number_of_batches)
            for x_batch, y_batch in zip(x_batches, y_batches):
                a = x_batch.transpose()
                label = y_batch.transpose()
                a, layer = self.forward_propagation(a, True)
                pred = a.argmax(axis=0)
                err = self.calculate_loss(a, label)
                delta = self.calculate_delta(a, label, layer)
                layer.delta = delta
                curr_epoch_err += err
                theta = layer.theta
                self.backpropagation(delta, theta)
                a = x_batch.transpose()
                self.update_network(a)

                batch_correct_predictions = (y_batch.argmax(axis=1) == pred).sum()
                # batch_accuracy = np.divide(batch_correct_predictions, float(y.shape[0])) * 100
                epoch_correct_predictions += batch_correct_predictions

            train_epochs_errors.append(curr_epoch_err/number_of_train_examples)
            train_epochs_acc.append(epoch_correct_predictions/number_of_train_examples)

            # test the network on the validation set
            if not validation_x is None:
                validation_error, validation_accuracy = self.validate(validation_x.T, validation_y.T)
                validation_errors.append(np.divide(validation_error, validation_x.shape[0]))
                validation_acc.append(validation_accuracy)

            # debug printing
            print('epoch', epoch, ': train error:', train_epochs_errors[epoch], ', \ttrain accuracy: ', train_epochs_acc[epoch]*100, '%')
            print('\t\t: validation error:', validation_errors[epoch], ', validation accuracy: ',  validation_acc[epoch]*100, '%')

        return train_epochs_errors, validation_errors, train_epochs_acc, validation_acc

    def validate(self, x, y):
        y_hat, layer = self.forward_propagation(x)
        err = self.calculate_loss(y_hat, y)
        acc = (y.argmax(axis=0) == y_hat.argmax(axis=0)).sum()
        acc = np.divide(acc, x.shape[1])
        return err, acc #TODO calculate_loss returns a scalar because of the np.sum, maybe we need to change it.

    def update_network(self, a):
        for layer in self.layers:
            layer.weights_update(a, self.alpha)
            a = layer.a

    def backpropagation(self, delta, theta):
        for layer in reversed(self.layers[:-1]):
            delta, theta = layer.backward(theta, delta), layer.theta

    def calculate_loss(self, a, label):
        """Calculate the loss value of a given prediction.
            a and label is k x n matrices where n is the number of data points and k is the number of output neurons.
        :param a: network output.
        :param label: one hot encoded.
        :return:
        """
        if self.loss == 'mse':
            diff = a - label
            err = np.square(diff).mean(axis=0).mean()  # axis 0 means the mean of every col
        elif self.loss == 'ce':
            return sum(-np.log2(a[label > 0]))
        #return -np.log2(a[label > 0])
        else:
            raise ValueError('loss function not implemented')
        return err

    def calculate_delta(self, a, label, layer):
        """Calculate delta values of a given prediction.

            layer activation function must be softmax or sigmoid.
        :param a:
        :param label:
        :param layer:
        :return:
        """
        diff = a - label
        if self.loss == 'mse':
            delta = diff * layer.activation_derivative(layer.z)
            #delta = np.sum(diff * layer.activation_derivative(layer.z), axis=1)
            #delta = delta.reshape(len(delta), 1)
            layer.delta = delta
        elif self.loss == 'ce':
            #delta = np.sum(diff, axis=1)
            # TODO TODO TODO
            delta = diff
        else:
            raise ValueError('delta for this loss function is not implemented')
        return delta

    def forward_propagation(self, a, is_training=False):
        y_hat = a
        for layer in self.layers:
            y_hat = layer.forward(y_hat, is_training)
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
        delta = self.calculate_delta(a, y, layer)
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
                    err_plus = self.calculate_loss(a_plus, y)
                    theta_minus = deepcopy(theta_copy)
                    theta_minus[i, j] -= epsilon
                    layer.theta = theta_minus
                    a_minus, l_minus = self.forward_propagation(x)
                    err_minus = self.calculate_loss(a_minus, y)
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
    def __init__(self, weights_matrix, bias, activation_function, activation_function_derivative, dropout_rate=0):
        """Create a hidden layer.

        :param dropout_rate:
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
        self.dropout_rate = dropout_rate

    def forward(self, previous_layer_output, is_training=False):
        """Calculate the a value of the layer.

        :param is_training:
        :param previous_layer_output: col vector or matrix where number of rows is number of features,
                number of cols is number f training examples
        :return: The a value.
        """
        a_prev = previous_layer_output
        z = np.dot(self.theta, a_prev)
        z = z + self.b
        a = self.activation(z)
        if is_training:
            mask = np.random.rand(a.shape[0], 1) < (1 - self.dropout_rate)
        else:
            mask = (1 - self.dropout_rate)
        a = a * mask
        self.z, self.a, self.mask = z, a, mask
        return self.a

    def backward(self, next_layer_weights, next_layer_delta):
        """Calculate the delta value of the layer.

        :param next_layer_weights:
        :param next_layer_delta:
        :return: current layer delta.
        """
        delta = np.dot(next_layer_weights.T, next_layer_delta)
        delta = delta * self.mask * self.activation_derivative(self.z)
        #delta = np.dot(delta, self.activation_derivative(self.z))
        self.delta = delta
        return delta

    def weights_update(self, previous_layer_output, learning_rate):
        """Update the layer weights and bias"""
        theta, b, delta, alpha = self.theta, self.b, self.delta, learning_rate
        dc_dtheta = np.dot(previous_layer_output, delta.T).transpose()
        new_theta = theta - alpha * dc_dtheta
        b_prime = np.sum(alpha * delta, axis=1).reshape(b.shape)
        new_b = b - b_prime
        self.theta, self.b = new_theta, new_b

