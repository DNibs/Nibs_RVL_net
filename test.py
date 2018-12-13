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


# Parameters
batch_sz = 10
init_learn_rt_adam = 0.002
learn_rt_decay_epoch = 10
wt_decay = 0.001355
momentum = 0.9
num_epochs = 20


class NibsNetRVLPyTorch(nn.Module):
    def __init__(self):
        super(NibsNetRVLPyTorch, self).__init__()
        self.fc1 = nn.Linear(3, 27)
        self.fc2 = nn.Linear(27, 3)

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

    dataset_train = CustomDataset('test_30.csv')

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=batch_sz,
                                               shuffle=True,
                                               drop_last=True)

    loader_test = torch.utils.data.DataLoader(dataset=dataset_train,
                                               batch_size=1,
                                               shuffle=True,
                                               drop_last=True)

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

        for batch_idx, (data, target) in enumerate(loader_train):
            optimizer.zero_grad()
            out = net(data.float())
            loss = loss_fn(out, target.float())
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                i, batch_idx * len(data), len(loader_train.dataset),
                100.0 * batch_idx / len(loader_train), loss.item()
            ))

        net.eval()
        loss_epoch = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader_train):
                out = net(data.float())
                loss = loss_fn(out, target.float())
                loss_epoch += loss.item()
                print('Validate Epoch: {} [{}/{} ({:.0f}%)]\tLoss for batch: {:.6f}'.format(
                    i, batch_idx * len(data), len(loader_train.dataset),
                           100.0 * batch_idx / len(loader_train), loss.item()
                ))

    # Build Confidence Matrix
    confmat = np.zeros([3, 3])
    with torch.no_grad():

        for batch_idx, (data, target) in enumerate(loader_test):
            out = net(data.float())
            actual = torch.argmax(target)
            predict = torch.argmax(out)

            confmat[actual][predict] += 1

    print('\n\nConfidence Matrix [actual x predicted]')
    print(confmat)


if __name__ == '__main__':
    main()
