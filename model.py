'''

RNN network

'''

import numpy as np
from numpy import loadtxt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import string

class RNN(nn.Module):
    def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units, 
        nonlinearity="tanh", device="cpu", which_init=None):

        super(RNN, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_num_units = hidden_num_units
        self.num_layers = num_layers

        # RNN layers
        self.rnn = nn.RNN(input_num_units, hidden_num_units, num_layers, nonlinearity=nonlinearity)

        # Fully connected layer
        self.fc = nn.Linear(hidden_num_units, output_num_units)

        self.device = device
        self.to(self.device)

        # custom (re-)initialization of the parameters of the network
        if which_init is not None:
            self.init_weights(which_init)        

    def init_weights(self, scaling, seed=None):
        
        if seed is not None:
            torch.manual_seed(seed)

        if scaling == "Rich":
            # initialisation of the weights -- N(0, 1/n)
            scaling_f = lambda f_in: 1./f_in
        elif scaling == "Lazy":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            scaling_f = lambda f_in: 1./np.sqrt(f_in)
        elif scaling == "Const":
            # initialisation of the weights independent of n
            scaling_f = lambda f_in: 0.001
        elif isinstance(scaling, float) and scaling > 0:
            # initialisation of the weights -- N(0, 1/n**alpha)
            '''
            UNTESTED
            '''
            scaling_f = lambda f_in: 1./np.power(f_in, scaling)
        else:
            raise ValueError(
                f"Invalid scaling option '{scaling}'\n" + \
                 "Choose either 'sqrt', 'lin' or a float larger than 0")
        
        for name, pars in self.named_parameters():
            if "weight" in name:
                f_in = 1.*pars.data.size()[1]
                std = scaling_f(f_in)
                pars.data.normal_(0., std)

    def forward(self, x):
        '''
        x
        ---
        seq_length, batch_size, input_num_units
        or
        seq_length, input_num_units
        '''

        '''
        h0 -- initial network state
        ---
        num_layers, batch_size, hidden_num_units
        or
        num_layers, hidden_num_units
        '''
        h0 = torch.randn(self.num_layers, x.shape[1], self.hidden_num_units).to(self.device)
        
        # h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_num_units).to(self.device)

        # ht = sequence of hidden states
        # hT = last hidden state

        # ht, hT = self.rnn(x, h0)

        ht, hT = self.rnn(x)


        # whole sequence of hidden states, linearly transformed
        y = self.fc(ht)
        y = F.softmax(self.fc(ht), dim=-1)
        
        return ht, hT, y


    def get_activity(self, x):

        with torch.no_grad():

            _x = x.clone().detach().to(self.device)

            ht, hT, _  = self.forward(_x)

            # print('ht', np.shape(ht)) # activity for whole sequence
            # print('hT', np.shape(hT)) # activity for last element of sequence

            y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

        return y