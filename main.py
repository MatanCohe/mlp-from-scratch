from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from NNClassifier import NeuralNetworkClassifier, Layer
from functions import relu_activation
from functions import my_softmax
from utils import read_labeled_data

import pandas as pd

learning_rate = 0.01
loss_func = 'ce'
train_file = './data/train.csv'
dev_file = './data/validate.csv'
figures_folder = './figures'
number_of_epochs = 150
batch_size = 20
NUMBER_OF_LABELS = 10
dropout_rate = 0.50
input_vector_dim = 3072
regularization_lambda = 0.0001 * 10 *5
#train_file = dev_file


def generate_weights(rows, cols):
    return np.random.uniform(-0.5, 0.5, size=(rows, cols))

if __name__ == '__main__':

    np.random.seed(1234)

    # read train data
    TRAIN = pd.read_csv(train_file, header=None)
    # normalize train data
    mean = TRAIN.values[:, 1:].mean()
    std = TRAIN.values[:, 1:].std()
    standardize_data = lambda x: np.divide(x - mean, std)

    x = standardize_data(TRAIN.values[:, 1:])
    y = TRAIN.values[:, 0] - 1
    y = pd.get_dummies(y).values
    print('train data was read!')

    # read validation data
    DEV = pd.read_csv(dev_file, header=None)

    dev_x = standardize_data(DEV.values[:, 1:])
    dev_y = DEV.values[:, 0]
    dev_y = pd.get_dummies(dev_y).values
    print('dev data was read!')

    # create the model
    l1 = Layer(weights_matrix=generate_weights(256, input_vector_dim), bias=generate_weights(256, 1),
               activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative,dropout_rate=dropout_rate)
    l2 = Layer(weights_matrix=generate_weights(NUMBER_OF_LABELS, 256), bias=generate_weights(NUMBER_OF_LABELS, 1),
               activation_function=my_softmax, activation_function_derivative=None)
    network = NeuralNetworkClassifier(layers=[l1, l2], learning_rate=learning_rate, loss_function=loss_func, l2_lambda=regularization_lambda, noise_type='gauss')
    # train
    print('about to train now...')
    train_errors, validation_errors, train_epochs_acc, validation_acc = network.train(x, y, number_of_epochs, dev_x, dev_y, batch_size)

    # draw loss plot
    plot_file_prefix = datetime.now().strftime('%Y_%m_%d_%H_%M')
    plt.figure(1)
    plt.plot(range(len(train_errors)), train_errors, 'r')
    plt.plot(range(len(validation_errors)), validation_errors, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    loss_figure_path = os.path.join(figures_folder, f'{plot_file_prefix}_loss.png')
    plt.savefig(loss_figure_path, bbox_inches='tight')

    # draw accuracy plot
    plt.figure(2)
    plt.plot(range(len(train_epochs_acc)), train_epochs_acc, 'r')
    plt.plot(range(len(train_epochs_acc)), validation_acc, 'b')
    plt.xlabel('Epoch')
    plt.ylabel('accuracy in %')
    plt.legend(['Train', 'Validation'])
    acc_figure_path = os.path.join(figures_folder, f'{plot_file_prefix}_acc.png')
    plt.savefig(acc_figure_path, bbox_inches='tight')
    plt.show()
