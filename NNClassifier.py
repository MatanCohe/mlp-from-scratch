from copy import deepcopy
import numpy as np

# from tests import profile
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
    # @profile.do_cprofile
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
                # a = x_batch.transpose()
                a = x_batch
                label = y_batch.transpose()
                a, layer = self.forward_propagation(a, True)
                pred = a.argmax(axis=0)
                err = self.calculate_loss(a, label)
                delta = self.calculate_delta(a, label, layer)
                curr_epoch_err += err
                self.backpropagation(delta)
                self.update_network(x_batch.shape[0])

                batch_correct_predictions = (y_batch.argmax(axis=1) == pred).sum()
                # batch_accuracy = np.divide(batch_correct_predictions, float(y.shape[0])) * 100
                epoch_correct_predictions += batch_correct_predictions

            train_epochs_errors.append(curr_epoch_err/number_of_train_examples)
            train_epochs_acc.append(epoch_correct_predictions/number_of_train_examples)

            # test the network on the validation set
            if not validation_x is None:
                validation_error, validation_accuracy = self.validate(validation_x, validation_y.T)
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
        acc = np.divide(acc, y_hat.shape[1])
        return err, acc

    def update_network(self, batch_size):
        """Updates the network weights.

        :param a: network output.
        :param batch_size:
        :return:
        """
        for layer in self.layers:
            layer.weights_update(self.alpha, self.l2_lambda, batch_size)

    def backpropagation(self, delta):
        """propagate the error through the network.

        :param delta: output delta.
        :param theta: last layer weights.
        :return: None
        """
        output_layer = self.layers[-1]
        output_layer.delta = delta
        da = output_layer.compute_previous_da()
        for layer in reversed(self.layers[:-1]):
            da = layer.backward(da)

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
        a, _ = self.forward_propagation(x)
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
        if is_training:
            self.prev_a ,self.z, self.a, self.mask = a_prev, z, a, mask
        return a

    def backward(self, da):
        """Calculate the delta value of the layer.

        :param delta:
        :return: current layer delta.
        """
        self.delta = da * self.mask * self.activation_derivative(self.z)
        
        prev_da = self.compute_previous_da()
        
        return prev_da

    def compute_previous_da(self):
        """
        Compute the dE/da gradient.
        
        return:
            dE/da gradient.
        """
        return np.dot(self.theta.T, self.delta)
    
    def weights_update(self, learning_rate, l2_lambda, batch_size):
        """Update the layer weights and bias"""
        prev_a, theta, b, delta, alpha = self.prev_a, self.theta, self.b, self.delta, learning_rate
        dc_dtheta = np.dot(prev_a, delta.T).transpose()
        dc_dtheta = np.divide(1, batch_size) * dc_dtheta
        new_theta = theta*(1 - l2_lambda * alpha) - alpha * dc_dtheta
        b_prime = np.sum(delta, axis=1).reshape(b.shape)
        b_prime = b_prime * np.divide(1, batch_size)
        new_b = b - alpha * b_prime
        self.theta, self.b = new_theta, new_b

