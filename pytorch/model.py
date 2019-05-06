import torch
import torch.nn as nn
from utils import TRAIN
import numpy as np

NUMBER_OF_LABELS = 10

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

def train():
    batch_size = 32
    for k in range(number_of_epochs):
        for i in range(0, TRAIN.shape[0], batch_size):
            x = TRAIN[i:i + batch_size, :]  # x contains a batch
            # forward
            y_hat = model(x[:, 1:])
            y = (x[:, 0] - 1).to(torch.long) # fix the classes since toech expects classes 0-9
            # y_one_hot = np.zeros((y.shape[0], NUMBER_OF_LABELS))
            # y_one_hot[np.arange(y.shape[0]), y.to(torch.int64)] = 1

            loss = loss_function(y_hat, y)
            pred = y_hat.argmax(dim=1)
            batch_accuracy = np.divide((y == pred).sum().data, float(y.shape[0])) * 100
            print("epoch", k, ",batch", np.divide(i, batch_size), ",loss is:", loss.item(), ",batch accuracy is:", batch_accuracy.item(), "%")
            model.zero_grad()
            loss.backward()

            # weight update (using gradient descent)
            with torch.no_grad():
                for param in model.parameters():
                    param -= learning_rate * param.grad


model = nn.Sequential(
    torch.nn.Linear(3072, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 10),

)

loss_function = torch.nn.CrossEntropyLoss(reduction='sum')
learning_rate = 0.0001
number_of_epochs = 50

train()

