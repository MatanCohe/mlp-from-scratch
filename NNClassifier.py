from copy import deepcopy
import numpy as np


class NeuralNetworkClassifier:
    """This class represents a Neural Network classifier"""

    def __init__(self, layers, learning_rate, loss_function, l2_lambda=0, noise_type=None):
        """Create a neural network classifier.

        :param l2_lambda:
        :param layers: List or tuple of layers.
        :param learning_rate: scalar.
        :param loss_function: String.

        :return: A neural network with the specified requirements.
        """
        self.layers, self.alpha = layers, learning_rate
        self.loss = loss_function
        self.l2_lambda = l2_lambda
        self.noise_type = noise_type

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
        train_data = list(zip(x, y))
        for epoch in range(number_of_epochs):
            np.random.shuffle(train_data)
            curr_epoch_err = 0
            epoch_correct_predictions = 0
            for x_batch, y_batch in self.split_to_batches(train_data, batch_size):
                if self.noise_type:
                    self.noise_data(x_batch)
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
                self.update_network(a, x_batch.shape[0])

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
            print(f'epoch: {epoch} train_error: {train_epochs_errors[epoch]:.3f} train_acc: {train_epochs_acc[epoch]*100:.3f}% val_error:, {validation_errors[epoch]:.3f} ,val_acc: {validation_acc[epoch]*100:.3f}%')

        return train_epochs_errors, validation_errors, train_epochs_acc, validation_acc

    def noise_data(self, x):
        """Add gaussian noise to x"""
        return x + np.random.normal(size=x.shape)


    def split_to_batches(self, train_data, batch_size):
        """Split the train data into batches of batch_size

        :param train_data:
        :param batch_size:
        :return:
        """
        num_of_training_examples = len(train_data)
        for i in range(0, num_of_training_examples, batch_size):
            x, y = zip(*train_data[i: i+batch_size])
            yield np.vstack(x), np.vstack(y)

    def validate(self, x, y):
        """Calculate the network error and accuracy over x"""
        y_hat, layer = self.forward_propagation(x)
        err = self.calculate_loss(y_hat, y)
        acc = (y.argmax(axis=0) == y_hat.argmax(axis=0)).sum()
        acc = np.divide(acc, x.shape[1])
        return err, acc

    def update_network(self, a, batch_size):
        """Updates the network weights.

        :param a: network output.
        :param batch_size:
        :return:
        """
        for layer in self.layers:
            layer.weights_update(a, self.alpha, self.l2_lambda, batch_size)
            a = layer.a

    def backpropagation(self, delta, theta):
        """propagate the error through the network.

        :param delta: output delta.
        :param theta: last layer weights.
        :return: None
        """
        for layer in reversed(self.layers[:-1]):
            delta, theta = layer.backward(theta, delta), layer.theta

    def calculate_loss(self, a, label):
        """Calculate the loss value of a given prediction.
            a and label is k x n matrices where n is the number of data points and k is the number of output neurons.
        :param a: network output.
        :param label: one hot encoded.
        :return: real number.
        """
        if self.loss == 'mse':
            diff = a - label
            err = np.square(diff).mean(axis=0).mean()
        elif self.loss == 'ce':
            return sum(-np.log2(a[label > 0]))
        else:
            raise ValueError('loss function not implemented')
        return err

    def calculate_delta(self, a, label, layer):
        """Calculate delta values of a given prediction.

            layer activation function must be softmax or sigmoid.
        :param a: model prediction.
        :param label: true label.
        :param layer: the last layer of the model.
        :return: numpy array the same shape as a
        """
        diff = a - label
        if self.loss == 'mse':
            delta = diff * layer.activation_derivative(layer.z)
            layer.delta = delta
        elif self.loss == 'ce':
            delta = diff
        else:
            raise ValueError('delta for this loss function is not implemented')
        return delta

    def forward_propagation(self, a, is_training=False):
        """Forward the examples through the network.

        :param a: Batch of examples shape=(num_of_features, num_of_examples)
        :param is_training: A flag indicates in what mode the function has been called.
        :return: y_hat - the prediction of the model.
        :return: layer - the last layer of the network.
        """
        y_hat = a
        for layer in self.layers:
            y_hat = layer.forward(y_hat, is_training)
        return y_hat, layer

    def predict(self, x):
        """Make prediction for x.

        :param x: each row in x represents one training example
        :return: vector of labels with a row for each example.
        """
        a, _ = self.forward_propagation(x.transpose())
        a = a.transpose()
        return np.argmax(a, axis=1)


    def check_gradient(self, x, y):
        """Check whether the gradient calculate through the network is valid"""
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
        self.delta = delta
        return delta

    def weights_update(self, previous_layer_output, learning_rate, l2_lambda, batch_size):
        """Update the layer weights and bias"""
        theta, b, delta, alpha = self.theta, self.b, self.delta, learning_rate
        dc_dtheta = np.dot(previous_layer_output, delta.T).transpose()
        dc_dtheta = np.divide(1, batch_size) * dc_dtheta
        new_theta = theta*(1 - l2_lambda * alpha) - alpha * dc_dtheta
        b_prime = np.sum(delta, axis=1).reshape(b.shape)
        b_prime = b_prime * np.divide(1, batch_size)
        new_b = b - alpha * b_prime
        self.theta, self.b = new_theta, new_b


class CnnLayer:

    def __init__(self, in_channels, out_channels, kernal_size, pad, stride, activation_function, activation_function_derivative):
        r, s = kernal_size
        self.kernals = np.random.uniform(-0.5, 0.5, size=(out_channels, in_channels, r, s))
        # TODO: confirm the right size of the bias.
        self.bias = np.random.uniform(-0.5, 0.5, size=(in_channels*r*s, 1))
        self.pad = pad
        self.stride = stride
        self.activation = activation_function
        self.activation_derivative = activation_function_derivative
        self.cache = None


    def forward(self, previous_layer_output):
        n, c, h, w = previous_layer_output.shape
        k, _, r, s = self.kernals.shape
        assert (h-r + 2*self.pad) % self.stride == 0
        assert (w-s + 2*self.pad) % self.stride == 0

        new_h = (h-r + 2*self.pad) / self.stride + 1
        new_w = (w-s + 2*self.pad) / self.stride + 1
        from cs231n.im2col import im2col_indices, col2im_indices
        input_as_cols = im2col_indices(previous_layer_output, r, s, self.pad, self.stride)
        flat_kernals = self.kernals.reshape((k, -1))

        res = np.matmul(flat_kernals, input_as_cols) + self.bias

        res = res.reshape(k, new_h, new_w, n)

        res = res.transpose(3, 0, 1, 2)

        self.cache = (previous_layer_output, input_as_cols)

        return res

    def backward(self, delta):
        x, x_as_cols = self.cache

        db = np.sum(delta, axis=(0, 2, 3))
        k, _, r, s = self.kernals.shape
        matrix_delta = delta.transpose(1, 2, 3, 0).reshape(k, -1)

        dw = np.matmul(matrix_delta, x_as_cols.T).reshape(self.kernals.shape)

        dx_cols = np.matmul(self.kernals.reshape(k, -1).T, matrix_delta)
        dx = col2im_indices(dx_cols, x.shape, r, s, self.pad, self.stride)

        self.db, self.dw, self.dx = db, dw, dx

        return dx

