import numpy as np
import matplotlib.pyplot as plt
from NNClassifier import NeuralNetworkClassifier, Layer, DropoutLayer
from functions import relu_activation
from functions import softmax
from utils import read_labeled_data
import pandas as pd

learning_rate = 0.001
loss_func = 'ce'
train_file = './data/train.csv'
dev_file = './data/validate.csv'
number_of_epochs = 10
batch_size = 10
NUMBER_OF_LABELS = 10

train_file = dev_file


if __name__ == '__main__':

    # read validation data
    DEV = pd.read_csv(dev_file, header=None)
    dev_x, dev_y = DEV.values[:, 1:], DEV.values[:,0]
    #dev_y_one_hot = np.zeros((dev_y.shape[0], NUMBER_OF_LABELS))
    #dev_y_one_hot[np.arange(dev_y.shape[0]), dev_y_one_hot.astype(np.int64)] = 1
    #dev_y = dev_y_one_hot
    dev_y = pd.get_dummies(dev_y).values
    print('dev data was read!')

    # read train data
    # x, y = read_labeled_data(train_file)
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


    # create the model
    l1 = Layer(weights_matrix=np.random.rand(256, 3072), bias=np.random.rand(256, 1),
               activation_function=relu_activation.f,
               activation_function_derivative=relu_activation.derivative)
    l2 = Layer(weights_matrix=np.random.rand(NUMBER_OF_LABELS, 256), bias=np.random.rand(NUMBER_OF_LABELS, 1),
               activation_function=softmax,
               activation_function_derivative=None)
    network = NeuralNetworkClassifier(layers=[l1, l2],
                                      learning_rate=learning_rate,
                                      loss_function=loss_func)
    # train
    print('about to train now...')
    train_errors, validation_errors = network.train(x, y, number_of_epochs, dev_x, dev_y, batch_size)

    # draw errors plot
    plt.plot(np.arange(0, len(train_errors), 1), train_errors, 'r')
    plt.plot(np.arange(0, len(train_errors), 1), validation_errors, 'r')
    #plt.savefig(folder_name + "/train_loss.png")
    #plt.clf()
    plt.show()
