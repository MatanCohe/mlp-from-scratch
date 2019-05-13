import csv
import numpy as np
number_of_classes = 10

def read_labeled_data(file_name):
    """" reads labeld data from a csv file
    :param file_name - the name of a csv file containing the data
    :returns np.array of the data set - each line represents an example
             & np.array of the labels - each line is a one hot vector of the labels
             shape would be num_of_exmaples*num_of_features
    """
    data_set = []
    labels = []
    #f = open(file_name)
    #data = csv.reader(f, delimiter=",")
    #for line in data:
    with open(file_name) as f:
        for line in f:
            line_array = [x for x in line.strip().split(',')]
            label = int(line_array[0])
            one_hot_label = np.zeros(number_of_classes)
            one_hot_label[label-1] = 1
            example = line_array[1:]

            data_set.append(example)
            labels.append(one_hot_label)
    return np.array(data_set), np.array(labels)

def read_unlabeled_data(file_name):
    data_set = []
    data = csv.reader(file_name, delimiter=",")
    for line in data:
        data_set.append(line)
    return np.array(data_set)

# DEV_SET = read_labeled_data('data\\validate.csv')
# TRAIN_SET = read_labeled_data('data\\train.csv')
# TEST_SET = read_unlabeled_data('data\\test.csv')
