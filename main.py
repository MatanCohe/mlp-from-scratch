from datetime import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from NNClassifier import NeuralNetworkClassifier, Layer
from functions import relu_activation
from functions import column_wise_softmax


learning_rate = 0.01
loss_func = 'ce'
number_of_epochs = 150
batch_size = 20
NUMBER_OF_LABELS = 10
dropout_rate = 0.50
input_vector_dim = 3072
regularization_lambda = 0.0001 * 10 *5


def generate_weights(rows, cols):
    return np.random.uniform(-0.5, 0.5, size=(rows, cols))

if __name__ == '__main__':
    main_folder = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(main_folder, 'data', 'train.csv')
    dev_file = os.path.join(main_folder, 'data', 'validate.csv')
    test_file = os.path.join(main_folder, 'data', 'test.csv')
    figures_folder = os.path.join(main_folder, 'figures')

    np.random.seed(1234)

    # read train data
    # TRAIN = pd.read_csv(train_file, header=None)
    TRAIN = pd.DataFrame(np.load('./bin_data/train.npy'))
    # normalize train data
    mean = TRAIN.values[:, 1:].mean()
    std = TRAIN.values[:, 1:].std()
    standardize_data = lambda x: np.divide(x - mean, std)

    x = standardize_data(TRAIN.values[:, 1:]).reshape(-1, 1, 3, 32, 32).astype(np.float64)
    y = TRAIN.values[:, 0] - 1
    y = pd.get_dummies(y).values
    print('train data was read!')

    # read validation data
    # DEV = pd.read_csv(dev_file, header=None)
    DEV = pd.DataFrame(np.load('./bin_data/validate.npy'))

    dev_x = standardize_data(DEV.values[:400, 1:]).reshape(-1, 3, 32, 32).astype(np.float64)
    dev_y = DEV.values[:400, 0]
    dev_y = pd.get_dummies(dev_y).values
    print('dev data was read!')

    # create the model
    import layers
    conv1 = layers.FastConv(kernels=np.random.uniform(-0.5, 0.5, size=(2, 3, 3, 3)), bias=np.zeros((2, )),
                          activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative)
    batch_norm1 = layers.BatchNorm2d(np.random.uniform(-0.5, 0.5, size=(1)),
                                    np.random.uniform(-0.5, 0.5, size=(1)),
                                    activation_function=relu_activation.f, 
                                    activation_function_derivative=relu_activation.derivative)
    pool1 = layers.MaxPool2d(2)
    conv2 = layers.FastConv(kernels=np.random.uniform(-0.5, 0.5, size=(4, 2, 3, 3)), bias=np.zeros((4, )),
                          activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative)
    batch_norm2 = layers.BatchNorm2d(np.random.uniform(-0.5, 0.5, size=(1)),
                                    np.random.uniform(-0.5, 0.5, size=(1)),
                                    activation_function=relu_activation.f, 
                                    activation_function_derivative=relu_activation.derivative)
    pool2 = layers.MaxPool2d(2)
    conv3 = layers.FastConv(kernels=np.random.uniform(-0.5, 0.5, size=(8, 4, 3, 3)), bias=np.zeros((8, )),
                          activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative)
    batch_norm3 = layers.BatchNorm2d(np.random.uniform(-0.5, 0.5, size=(1)),
                                    np.random.uniform(-0.5, 0.5, size=(1)),
                                    activation_function=relu_activation.f, 
                                    activation_function_derivative=relu_activation.derivative)
    pool3 = layers.MaxPool2d(2)
    flatten = layers.Conv2Linear()
    l1 = Layer(weights_matrix=generate_weights(256, 128), bias=generate_weights(256, 1),
               activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative,dropout_rate=dropout_rate)
    l2 = Layer(weights_matrix=generate_weights(NUMBER_OF_LABELS, 256), bias=generate_weights(NUMBER_OF_LABELS, 1),
               activation_function=column_wise_softmax, activation_function_derivative=None)
    network = NeuralNetworkClassifier(layers=[
        conv1, 
        batch_norm1,  
        pool1,
        conv2, 
        batch_norm2,  
        pool2,
        conv3, 
        batch_norm3,  
        pool3, 
        flatten, 
        l1, 
        l2], 
                                      learning_rate=learning_rate, loss_function=loss_func, 
                                      l2_lambda=regularization_lambda, noise_type='gauss')
    # train
    print('about to train now...')
    train_errors, validation_errors, train_epochs_acc, validation_acc = network.train(x, y, number_of_epochs, dev_x, dev_y, batch_size)

    TEST = pd.read_csv(test_file, header=None)
    # TEST = pd.DataFrame(np.load('./bin_data/test.npy'))
    test_x = standardize_data(TEST.values[:, 1:].astype(np.float64)).reshape(-1, 3, 32, 32)
    test_predict = network.predict(test_x) + 1
    np.savetxt(os.path.join(main_folder, 'output.txt'), np.array(test_predict, dtype=int), fmt='%d')

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
