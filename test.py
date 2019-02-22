# test.py
# RVL Lab Meeting
# Author: David Niblick
# Date: 14DEC18


import csv
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as fn
import matplotlib.pyplot as plt


# Parameters
batch_sz = 10
init_learn_rt_adam = 0.0001
learn_rt_decay_epoch = 10
wt_decay = 0.001355
momentum = 0.9
num_epochs = 150


class NibsNetRVLPyTorch(nn.Module):
    def __init__(self):
        super(NibsNetRVLPyTorch, self).__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 3)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x


class CustomDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, data_directory):
        # Read CSV file
        with open(data_directory, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        # Parse Data to np array
        myFile = np.genfromtxt(data_directory, delimiter=',')
        self.data_example = np.array(myFile[1:, 2:], dtype=float)

        # One-Hot encoding for targets (of three classes)
        self.target_example = np.zeros([len(data)-1, 3], dtype=float)
        for i in range(1, len(data)):
            if data[i][1] == 'c1':
                self.target_example[i-1, 0] = 1
            elif data[i][1] == 'c2':
                self.target_example[i-1, 1] = 1
            elif data[i][1] == 'c3':
                self.target_example[i-1, 2] = 1

    def __getitem__(self, index):

        return self.data_example[index], self.target_example[index]

    def __len__(self):
        return len(self.data_example)


def main():
    print('Welcome to RVL Neural-A-Thon!')

    dataset_train = CustomDataset('training_9000_strongly_overlapping.csv')
    dataset_test = CustomDataset('test_30_strongly_overlapping.csv')

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_sz,
                                               shuffle=True,
                                               drop_last=True)

    loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                               batch_size=1)

    net = NibsNetRVLPyTorch()
    net = net.float()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=init_learn_rt_adam, weight_decay=wt_decay)

    epoch = 0
    train_loss = []
    val_loss = []
    epoch_time = []
    learn_rt = []

    for i in range(0, num_epochs):
        epoch += 1
        net.train()
        loss_epoch = 0
        print('')

        for batch_idx, (data, target) in enumerate(loader_train):
            optimizer.zero_grad()
            out = net(data.float())
            loss = loss_fn(out, target.float())
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('\rTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                    i, batch_idx * len(data), len(loader_train.dataset),
                    100.0 * batch_idx / len(loader_train), loss.item()
                ), end='')
        train_loss.append(loss_epoch)

        net.eval()
        loss_epoch = 0
        print('')
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader_train):
                out = net(data.float())
                loss = loss_fn(out, target.float())
                loss_epoch += loss.item()

                if batch_idx % 10 == 0:
                    print('\rValidate Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                        i, batch_idx * len(data), len(loader_train.dataset),
                               100.0 * batch_idx / len(loader_train), loss.item()
                    ), end='')
            val_loss.append(loss_epoch)

    # Build Confidence Matrix
    confmat = np.zeros([3, 3])
    pred_correct = 0
    total = 0
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(loader_test):
            out = net(data.float())
            actual = torch.argmax(target)
            predict = torch.argmax(out)
            total += 1
            if actual == predict:
                pred_correct += 1
            confmat[actual][predict] += 1

    accuracy = pred_correct / total * 100
    print('\n\nAccuracy: {}'.format(pred_correct / total * 100))
    print('\n\nConfidence Matrix [actual x predicted]')
    print(confmat)

    # plot results, save to file
    plt.figure(0)
    plt.plot(train_loss)

    plt.plot(val_loss)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Validation'])
    plt.title('Batch = {}, eta = {}, Test Acc = {}'.format(batch_sz, init_learn_rt_adam, accuracy))
    plt.suptitle('Validate and Training Error')
    plt.savefig('my_err_b25_eta25')
    plt.show()

if __name__ == '__main__':
    main()
