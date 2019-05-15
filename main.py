import numpy as np
import matplotlib.pyplot as plt
from NNClassifier import NeuralNetworkClassifier, Layer
from functions import relu_activation
from functions import my_softmax
from utils import read_labeled_data
import pandas as pd

learning_rate = 0.005
loss_func = 'ce'
train_file = './data/train.csv'
dev_file = './data/validate.csv'
number_of_epochs = 150
batch_size = 20
NUMBER_OF_LABELS = 10
dropout_rate = 0
input_vector_dim = 3072
regularization_lambda = 0.0001

#train_file = dev_file


def generate_weights(rows, cols):
    return np.random.uniform(-0.5, 0.5, size=(rows, cols))

if __name__ == '__main__':

    # read train data
    TRAIN = pd.read_csv(train_file, header=None)
    # normalize train data
    mean = TRAIN.values[:, 1:].mean()
    std = TRAIN.values[:, 1:].std()
    # choose the columns indexes we wish to change
    idx = np.arange(1, TRAIN.shape[1])
    TRAIN[idx] = np.divide(TRAIN[idx] - mean, std)
    x = TRAIN.values[:, 1:]
    y = TRAIN.values[:, 0] - 1
    y = pd.get_dummies(y).values
    print('train data was read!')

    # read validation data
    DEV = pd.read_csv(dev_file, header=None)
    # choose the columns indexes we wish to change for normailzation
    idx = np.arange(1, DEV.shape[1])
    DEV[idx] = np.divide(DEV[idx] - mean, std)
    dev_x = DEV.values[:, 1:]
    dev_y = DEV.values[:, 0]
    dev_y = pd.get_dummies(dev_y).values
    print('dev data was read!')

    # create the model
    l1 = Layer(weights_matrix=generate_weights(128, input_vector_dim), bias=generate_weights(128, 1),
               activation_function=relu_activation.f, activation_function_derivative=relu_activation.derivative,dropout_rate=dropout_rate)
    l2 = Layer(weights_matrix=generate_weights(NUMBER_OF_LABELS, 128), bias=generate_weights(NUMBER_OF_LABELS, 1),
               activation_function=my_softmax, activation_function_derivative=None)
    network = NeuralNetworkClassifier(layers=[l1, l2], learning_rate=learning_rate, loss_function=loss_func, l2_lambda=regularization_lambda)
    # train
    print('about to train now...')
    train_errors, validation_errors, train_epochs_acc, validation_acc = network.train(x, y, number_of_epochs, dev_x, dev_y, batch_size)

    # draw loss plot
    plt.figure(1)
    plt.plot(np.arange(0, len(train_errors), 1), train_errors, 'r')
    plt.plot(np.arange(0, len(train_errors), 1), validation_errors, 'b')
    #plt.savefig(folder_name + "/train_loss.png")
    #plt.clf()
    #plt.show()

    # draw accuracy plot
    plt.figure(2)
    plt.plot(np.arange(0, len(train_epochs_acc), 1), train_epochs_acc, 'r')
    plt.plot(np.arange(0, len(validation_acc), 1), validation_acc, 'b')
    plt.show()
