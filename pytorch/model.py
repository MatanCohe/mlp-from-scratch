import torch
import torch.nn as nn
from utils import TRAIN, DEV
import numpy as np
import matplotlib.pyplot as plt

NUMBER_OF_LABELS = 10
folder_name = '../figures'
# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         self.layer1 = nn.Linear(3072, 1000)
#         self.layer2 = nn.Linear(1000, NUMBER_OF_LABELS)
#     def forward(self, x):
#         x = self.layer1(x)
#         x = nn.ReLU(x)
#         x = self.layer2(x)
#         x = nn.ReLU(x)
#         return x
#
# def train_model(train_loader, criterion, opimizer, epochs, dev_data_loader)
#     epoch_accuracy_list = []
#     for k in epochs:
#         Classifier.train()
#         epoch_loss = 0
#         correct_per_epoch = 0
#         num_of_training_exm = train_loader.dataset.__len__()


def test(model):
    #torch.no_grad()
    model.eval()
    x = DEV
    # forward
    y_hat = model(x[:, 1:])
    y = (x[:, 0] - 1).to(torch.long)  # fix the classes since torch expects classes 0-9

    loss = loss_function(y_hat, y)
    average_loss = np.divide(loss.item(), DEV.shape[0])
    pred = y_hat.argmax(dim=1)
    accuracy = np.divide((y == pred).sum().data, DEV.shape[0])
    return average_loss, accuracy

def train(model):
    batch_size = 32
    epochs_accuracy = []
    epochs_loss = []
    devs_loss = []
    devs_accuracy = []
    num_of_training_examples = TRAIN.shape[0]
    lr_decay = 0.99
    learning_rate = LR
    # test on validation set
    validation_loss, validation_accuracy = test(model)
    devs_accuracy.append(validation_accuracy)
    devs_loss.append(validation_loss)
    print("validation loss is:", validation_loss.item(), "\tvalidation accuracy is:", validation_accuracy.item() * 100,
          "%")
    for k in range(num_of_epochs):
        model.train()
        #model.zero_grad()
        epoch_correct_predictions = 0
        epoch_loss = 0
        for i in range(0, num_of_training_examples, batch_size):
            #optimizer.zero_grad()
            x = TRAIN[i:i + batch_size, :]  # x contains a batch
            # forward
            y_hat = model(x[:, 1:])
            y = (x[:, 0] - 1).to(torch.long) # fix the classes since toech expects classes 0-9
            # y_one_hot = np.zeros((y.shape[0], NUMBER_OF_LABELS))
            # y_one_hot[np.arange(y.shape[0]), y.to(torch.int64)] = 1

            batch_loss = loss_function(y_hat, y)
            epoch_loss += batch_loss.item()
            pred = y_hat.argmax(dim=1)
            batch_correct_predictions = (y == pred).sum().data
            batch_accuracy = np.divide(batch_correct_predictions, float(y.shape[0])) * 100
            epoch_correct_predictions += batch_correct_predictions
            #print("epoch", k, ",batch", np.divide(i, batch_size), ",loss is:", batch_loss.item(), ",batch accuracy is:", batch_accuracy.item(), "%")

            model.zero_grad()
           # optimizer.zero_grad()
            batch_loss.backward()

           # weight update (using gradient descent)
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad
            #optimizer.step()
        epochs_accuracy.append(np.divide(epoch_correct_predictions, num_of_training_examples))
        epochs_loss.append(np.divide(epoch_loss, num_of_training_examples))

        # test on validation set
        validation_loss, validation_accuracy = test(model)
        devs_accuracy.append(validation_accuracy)
        devs_loss.append(validation_loss)

        # print
        print("epoch", k, "\n", "train loss is:", epochs_loss[k], "\ttrain epoch accuracy is:",
              epochs_accuracy[k].item()*100, "%")
        print("validation loss is:", validation_loss.item(), "\tvalidation accuracy is:", validation_accuracy.item()*100, "%")

        #optimizer.param_group['lr'] = optimizer.param_group['lr']* lr_decay
        learning_rate = learning_rate * lr_decay

    # draw plots for train
    plt.plot(np.arange(0, len(epochs_loss), 1), epochs_loss, 'r')
    plt.savefig(folder_name + "/train_loss.png")
    plt.clf()
    plt.plot(np.arange(0, len(epochs_accuracy), 1), epochs_accuracy, 'r')
    plt.savefig(folder_name + "/train_accuracy.png")
    plt.clf()
    # draw plots for validation
    plt.plot(np.arange(0, len(devs_loss), 1), devs_loss, 'r')
    plt.savefig(folder_name + "/validation_loss.png")
    plt.clf()
    plt.plot(np.arange(0, len(devs_accuracy), 1), devs_accuracy, 'r')
    plt.savefig(folder_name + "/validation_accuracy.png")
    plt.clf()

my_model = nn.Sequential(
    torch.nn.Linear(3072, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 10)
)

LR = 0.001#0.0000001
#optimizer = torch.optim.SGD(my_model.parameters(), lr=LR, weight_decay=1)
loss_function = torch.nn.CrossEntropyLoss()
num_of_epochs = 150

train(my_model)
torch.save(my_model.state_dict(), 'model_file')