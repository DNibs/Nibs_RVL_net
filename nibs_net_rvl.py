# nibs_net_rvl.py
# RVL Lab Meeting
# Author: David Niblick
# Date: 14DEC18


"""
Implements my custom neural network API (nibTorch) for RVL Meeting
Dataset is classification of randomly distributed values to three classes
Each item contains three real values
data is in a .csv file
"""


import csv
import numpy as np
import torch
import nibTorch as nn
import time
import matplotlib.pyplot as plt


class NibsNetRVL:

    def __init__(self):
        #   local variables and hyperparameters
        self.error_arr = np.zeros(1)
        self.eta = 0.001
        self.batch_size = 25
        self.epochsN = 150

        # Select as False if you want to use gpu
        self.no_cuda = True

        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # architecture *need 3 or 4 hidden layers, but don't do more*
        self.nibs_net_rvl = nn.NeuralNetwork([3, 5, 3], no_cuda=self.no_cuda)

    # forward takes in one image (as a 28*28 tensor) and predicts class
    def forward(self, in_img):

        # reshape image as a 1*784
        in_img_1d = in_img.view(1, 3).to(self.device)

        out = self.nibs_net_rvl.forward(in_img_1d)
        res_class = torch.argmax(out)
        return res_class

    def train(self):

        # download data
        # Read CSV file
        with open('training_9000_strongly_overlapping.csv', newline='') as csvfile:
            file_data = list(csv.reader(csvfile))

        # Parse Data to np array
        myFile = np.genfromtxt('training_9000_strongly_overlapping.csv', delimiter=',')
        data_train = myFile[1:, 2:]

        # One-Hot encoding for targets (of three classes)
        target_train = np.zeros([len(file_data) - 1, 3])
        for i in range(1, len(file_data)):
            if file_data[i][1] == 'c1':
                target_train[i - 1, 0] = 1
            elif file_data[i][1] == 'c2':
                target_train[i - 1, 1] = 1
            elif file_data[i][1] == 'c3':
                target_train[i - 1, 2] = 1

        data_train = torch.from_numpy(data_train)
        target_train = torch.from_numpy(target_train)
        # print(np.shape(data_example))
        # print(np.shape(target_example))

        with open('test_300_strongly_overlapping.csv', newline='') as csvfile:
            file_data = list(csv.reader(csvfile))

        # Parse Data to np array
        myFile = np.genfromtxt('test_300_strongly_overlapping.csv', delimiter=',')
        data_test = myFile[1:, 2:]

        # One-Hot encoding for targets (of three classes)
        target_test = np.zeros([len(file_data) - 1, 3])
        for i in range(1, len(file_data)):
            if file_data[i][1] == 'c1':
                target_test[i - 1, 0] = 1
            elif file_data[i][1] == 'c2':
                target_test[i - 1, 1] = 1
            elif file_data[i][1] == 'c3':
                target_test[i - 1, 2] = 1

        data_test = torch.from_numpy(data_test)
        target_test = torch.from_numpy(target_test)

        # Find number of training batches
        batchN = int(len(data_train) / self.batch_size)

        # Create arrays for holding error, accuracy
        train_error = np.zeros(self.epochsN)
        validate_error = np.zeros(self.epochsN)
        validate_accuracy = np.zeros(self.epochsN)
        epoch_time = np.zeros(self.epochsN)
        len_train = len(data_train)
        len_test = len(data_test)

        for i in range(0, self.epochsN):
            print('epoch {}'.format(i))
            start_time = time.time()

            # Train loop
            for j in range(0, len_train):
                # flatten each image, then forward pass
                data = data_train[j].to(self.device).unsqueeze(0)
                res = self.nibs_net_rvl.forward(data)

                # onehot encode of target, then forward pass
                target_onehot = target_train[j].unsqueeze(1)
                # target_onehot = target_onehot.t()
                target_onehot = target_onehot.to(self.device)

                self.nibs_net_rvl.backward(target_onehot)

                # update params
                self.nibs_net_rvl.updateParams(self.eta)

            # Save error from training loop
            train_error[i] = self.nibs_net_rvl.getError()
            print('train_err {}'.format(train_error[i]))

        # Test Loop
        test_iter = 0
        pass_iter = 0
        actual = np.array([0, 0, 0])
        pred = np.array([0, 0, 0])
        confmat = np.zeros([3, 3])

        for i in range(0, len_test):
            test_iter += 1
            data = data_test[i].unsqueeze(0).to(self.device)
            res = self.nibs_net_rvl.forward(data)
            target = target_test[i].to(self.device)
            # print('Target: {}'.format(target))
            # print('Guess: {}'.format(res))

            target = torch.argmax(target_test[i]).to(self.device)
            selection = torch.argmax(res)
            if selection == target:
                pass_iter += 1

            confmat[selection][target] += 1

            print('Target: {}'.format(target))
            print('Guess: {}'.format(selection))

        print('Confidence Matrix [actual x predicted]')
        print(confmat)

        # test accuracy
        accuracy = pass_iter / test_iter
        # print('train_err {}'.format(train_error))
        # print('val_err {}'.format(validate_error))
        # print('val accuracy {}'.format(validate_accuracy))
        # print('epoch times {}'.format(epoch_time))
        print('Test Accuracy {}'.format(accuracy))

        # plot results, save to file
        plt.figure(0)
        plt.plot(train_error)

        # plt.plot(validate_error)
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.legend(['Training', 'Validation'])
        plt.title('Batch = {}, eta = {}, Test Acc = {}'.format(self.batch_size, self.eta, accuracy))
        plt.suptitle('Validate and Training Error')
        plt.savefig('my_err_b25_eta25')

        # plt.figure(1)
        # plt.plot(validate_accuracy)
        # plt.ylabel('Validation Accuracy')
        # plt.xlabel('Epochs')
        # plt.title('Batch = {}, eta = {}, Test Acc = {}'.format(self.batch_size, self.eta, accuracy))
        # plt.suptitle('Validation Accuracy vs Epochs')
        # plt.savefig('my_acc_b25_eta25')

        # plt.figure(2)
        # plt.plot(epoch_time)
        # plt.ylabel('Time (s)')
        # plt.xlabel('Epochs')
        # plt.title('Batch = {}, eta = {}, Test Acc = {}'.format(self.batch_size, self.eta, accuracy))
        # plt.suptitle('Train and Validation Time per Epoch')
        # plt.savefig('my_time_b25_eta25')

        plt.show()
