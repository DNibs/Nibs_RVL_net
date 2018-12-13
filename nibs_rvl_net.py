# nibs_rvl_net.py
# RVL Lab Meeting
# Author: David Niblick
# Date: 14DEC18

# Class that creates simple neural network with ability to conduct forward pass

import numpy as np
import torch as tch


# May want to create two additional attributes (dictionaries) a and z in class

class NeuralNetwork:

    # Creates dictionary of matrices theta
    def __init__(self, lst=None, no_cuda=True ):

        # dictionary to allow access to matrix of differences used in backpropogation
        self.de_dTheta = {}

        # dictionary to allow access to matrix of weights in layer x
        self.theta = {}

        self.error_total = 0.

        for i in range(0, len(lst) - 1):
            # NOTE: Adding one in dim0 for bias term weight!
            # Normalizing layer weights by mean=0 std=1/sqrt(#nodes)
            self.theta[i] = tch.ones([lst[i]+1, lst[i+1]], dtype=tch.float).normal_(mean=0, std=1/np.sqrt(lst[i]))

        # Select as False if you want to use gpu
        self.no_cuda = no_cuda

        self.use_cuda = not self.no_cuda and tch.cuda.is_available()
        self.device = tch.device("cuda" if self.use_cuda else "cpu")

    # Returns the theta layer
    def getLayer(self, layer):

        # returns the weight matrices per the dictionary
        # example - layer = x.getLayer(0) returns the weight tensor just after "in"

        return self.theta[layer]

    # Returns a snapshot of total error
    def getError(self):

        return self.error_total

    # Perform forward pass on network previously built using sigmoid non-linearities
    def forward(self, in_arr):

        # Tensor convention: first axis (vert) is previous inputs, second axis (hor) is outputs

        # Transform inputs to a tensor

        prev_arr = tch.tensor(in_arr, dtype=tch.float)
        prev_arr = tch.t(prev_arr)
        prev_arr = prev_arr.to(self.device)

        for i in self.theta:

            # Create tensor of ones to represent bias
            bias = tch.ones((1, prev_arr.size(1)), dtype=tch.float).to(self.device)

            # Add bias to beginning of input matrix
            bias_prev_arr = tch.cat((bias, prev_arr), 0)

            # Store inputs into de_dTheta dictionary for use during backprop
            self.de_dTheta[i] = bias_prev_arr # MIGHT BE INCORRECT!!!

            # Get weights for layer
            w_arr = tch.t(self.theta[i]).to(self.device)

            # Multiply input (including bias) by weights
            z_arr = tch.mm(w_arr, bias_prev_arr)

            # Calculate output of neurons using sigmoid activation
            prev_arr = tch.sigmoid(z_arr)

        # Save result of forward pass
        self.result = prev_arr

        return self.result

    def backward(self, target):

        # Calculate overall Error of forward pass
        error_vec = 0.5*((target - self.result)**2)
        self.error_total = tch.sum(error_vec)

        # Calculate delta for output layer
        delta = (-1) * (target - self.result) * (1 - self.result)
        delta_cut = delta.to(self.device)

        length = len(self.theta)
        # Step backwards through network
        for i in range(0, length):

            # get layer inputs (from last to first, working backwards)
            out_layer = self.de_dTheta[length-i-1].to(self.device)
            # print('outlayer {}'.format(out_layer))

            # get layer weights
            w_arr = self.theta[length-i-1].to(self.device)
            # print('warr {}'.format(w_arr))

            # Calculate error in respect to input weights
            de_dw = tch.mm(delta_cut, tch.t(out_layer))
            # print('de_dw {}'.format(de_dw))

            # Set value in dictionary
            self.de_dTheta[length - i - 1] = tch.t(de_dw).to(self.device)
            # print('de_dTheta {}'.format(self.de_dTheta[length - i - 1]))

            # Calculate new delta = w_arr * delta * out_layer ( 1 - out_layer)
            tmp1 = tch.mm(w_arr, delta_cut)
            delta = tmp1 * out_layer * (1 - out_layer)

            # Remove bias term from delta
            delta_cut = delta.narrow(0, 1, delta.size(0) - 1)
            # print('delta_cut {}'.format(delta_cut))

    def updateParams(self, eta):

        # Updates Parameters after backwards pass
        for i in self.theta:
            self.theta[i] = self.theta[i] - eta * self.de_dTheta[i].to('cpu')
            # self.de_dTheta[i] = tch.zeros(self.de_dTheta[i].size())

