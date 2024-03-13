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


# class RNN(nn.Module):
#     def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units, 
#         nonlinearity="tanh", device="cpu", which_init=None):

#         super(RNN, self).__init__()

#         # Defining the number of layers and the nodes in each layer
#         self.hidden_num_units = hidden_num_units
#         self.num_layers = num_layers

#         # RNN layers
#         self.rnn = nn.RNN(input_num_units, hidden_num_units, num_layers, nonlinearity=nonlinearity)

#         # Fully connected layer
#         self.fc = nn.Linear(hidden_num_units, output_num_units)

#         self.device = device
#         self.to(self.device)

#         # custom (re-)initialization of the parameters of the network
#         if which_init is not None:
#             self.init_weights(which_init)        

#     def init_weights(self, scaling, seed=None):
        
#         if seed is not None:
#             torch.manual_seed(seed)

#         if scaling == "Rich":
#             # initialisation of the weights -- N(0, 1/n)
#             scaling_f = lambda f_in: 1./f_in
#         elif scaling == "Lazy":
#             # initialisation of the weights -- N(0, 1/sqrt(n))
#             scaling_f = lambda f_in: 1./np.sqrt(f_in)
#         elif scaling == "Const":
#             # initialisation of the weights independent of n
#             scaling_f = lambda f_in: 0.001
#         elif isinstance(scaling, float) and scaling > 0:
#             # initialisation of the weights -- N(0, 1/n**alpha)
#             '''
#             UNTESTED
#             '''
#             scaling_f = lambda f_in: 1./np.power(f_in, scaling)
#         else:
#             raise ValueError(
#                 f"Invalid scaling option '{scaling}'\n" + \
#                  "Choose either 'sqrt', 'lin' or a float larger than 0")
        
#         for name, pars in self.named_parameters():
#             if "weight" in name:
#                 f_in = 1.*pars.data.size()[1]
#                 std = scaling_f(f_in)
#                 pars.data.normal_(0., std)

#     def forward(self, x):
#         '''
#         x
#         ---
#         seq_length, input_num_units
#         '''

#         '''
#         h0 -- initial network state
#         ---
#         num_layers, batch_size, hidden_num_units
#         or
#         num_layers, hidden_num_units
#         '''
        
#         if len(x.shape) == 2:
#             # h0 = torch.randn(self.num_layers, self.hidden_num_units).to(self.device)
#             h0 = torch.zeros(self.num_layers, self.hidden_num_units).to(self.device)
#         elif len(x.shape) == 3:
#             # h0 = torch.randn(self.num_layers, x.shape[1], self.hidden_num_units).to(self.device)
#             h0 = torch.zeros(self.num_layers, x.shape[1], self.hidden_num_units).to(self.device)

#         # ht = sequence of hidden states
#         # hT = last hidden state

#         ht, hT = self.rnn(x, h0)

#         # ht, hT = self.rnn(x)

#         # whole sequence of hidden states, linearly transformed
#         y = self.fc(ht)
#         y = F.softmax(self.fc(ht), dim=-1)
        
#         return ht, hT, y


#     def get_activity(self, x):

#         with torch.no_grad():

#             _x = x.clone().detach().to(self.device)

#             ht, hT, _  = self.forward(_x)

#             # print('ht', np.shape(ht)) # activity for whole sequence
#             # print('hT', np.shape(hT)) # activity for last element of sequence

#             y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

#         return y

#############################################################################

class Net(nn.Module):
    '''
    Base class for network models.
    Attribute function `init_weights` is a custom weight initialisation.
    '''

    def init_weights (self, scaling):
        torch.manual_seed(1871)

        scaling_arr = scaling.split(",")
        assert len(scaling_arr) in [1, len(list(self.named_parameters()))], \
            "The `scaling` parameter must be a string with one of the available options, "+\
            "or multiple available options comma-separated (as many as the number of layers)"
        
        for l, (name, pars) in enumerate(self.named_parameters()):
            if len(scaling_arr) == 1:
                scaling = scaling_arr[0]
            else:
                scaling = scaling_arr[l]

            if "weight" in name:
                f_in = 1.*pars.data.size()[1]
                if scaling == "lin":
                    # initialisation of the weights -- N(0, 1/n)
                    init_f = lambda f_in: (0., 1./f_in)
                elif scaling == "lin+":
                    # initialisation of the weights -- N(0, 1/n)
                    init_f = lambda f_in: (1./f_in, 1./f_in)
                elif scaling == "sqrt":
                    # initialisation of the weights -- N(0, 1/sqrt(n))
                    init_f = lambda f_in: (0., 1./np.sqrt(f_in))
                elif scaling == "const":
                    # initialisation of the weights independent of n
                    init_f = lambda f_in: (0., 0.001)
                elif scaling == "const+":
                    # initialisation of the weights independent of n
                    init_f = lambda f_in: (0.001, 0.001)
                elif isinstance(scaling, float) and scaling > 0:
                    # initialisation of the weights -- N(0, 1/n**alpha)
                    '''
                    UNTESTED
                    '''
                    init_f = lambda f_in: (0., 1./np.power(f_in, scaling))
                else:
                    raise ValueError(
                        f"Invalid scaling option '{scaling}'\n" + \
                         "Choose either 'sqrt', 'lin' or a float larger than 0")

                mu, sigma = init_f(f_in)
                pars.data.normal_(mu, sigma)

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename, map_location=self.device))

    def __len__ (self):
        return len(self._modules.items())



class RNN (Net):

    def __init__(self, d_input, d_hidden, num_layers, d_output, 
            activation='relu',
            output_activation = None, # for classification vs regression tasks
            drop_l=None,
            nonlinearity=F.tanh,
            layer_type=nn.Linear,
            which_init='Const',
            bias=False,
            device="cpu"
        ):

        super(RNN, self).__init__()
        scaling=which_init
        if scaling is None:
            scaling = "Const"

        self.device = device
        # Defining the number of layers and the nodes in each layer
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden

        self.i2h = nn.Linear (d_input, d_hidden, bias=False)
        self.h2h = layer_type (d_hidden, d_hidden, bias=bias)
        self.h2o = nn.Linear (d_hidden, d_output, bias=bias)

        if activation in [None, 'linear']:
            self.phi = lambda x: x
        elif activation == 'relu':
            self.phi = lambda x: F.relu(x)
        elif activation == 'sigmoid':
            self.phi = lambda x: F.sigmoid(x)
        else:
            raise NotImplementedError("activation function " + \
                            f"\"{activation}\" not implemented")

        if output_activation in [None, 'linear']:
            self.out_phi = lambda x: x
        elif output_activation == 'softmax':
            self.out_phi = lambda x: F.softmax(x, dim=-1)
        else:
            raise NotImplementedError("output activation function " + \
                            f"\"{output_activation}\" not implemented")

        # convert drop_l into a list of strings
        if drop_l == None:
            drop_l = ""
        elif drop_l == "all":
            drop_l = ",".join([str(i+1) for i in range(self.n_layers)])
        drop_l = drop_l.split(",")

        self.init_weights (scaling)


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

    # def forward (self, x, h0):
    def forward (self, x):
        '''
        x
        ---
        seq_length, batch_size, input_num_units
        or
        seq_length, input_num_units
        '''
        _shape = *x.shape[:-1], self.d_hidden

        # if batch dimension missing, add it
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]) )
        
        ht = torch.zeros(x.shape[1], self.d_hidden)
        # hidden = ht
        hidden = []
        # print(40*"==")
        # print("x.shape=", x.shape)
        for t, xt in enumerate(x):
            # print("xt.shape=", xt.shape)
            z = self.i2h (xt);      # print("step 1, z.shape=", z.shape)
            z = self.h2h (ht + z);  # print("step 2, z.shape=", z.shape)
            z = self.phi (z);       # print("step 3, z.shape=", z.shape)
            hidden.append(z)
            ht = z

        hidden = torch.reshape(torch.stack(hidden), _shape)
        y = self.h2o(hidden)
        y = self.out_phi(y)

        return hidden, hidden[-1].view(1,-1), y

    def get_activity(self, x):

        with torch.no_grad():

            _x = x.clone().detach().to(self.device)

            ht, hT, _  = self.forward(_x)

            # print('ht', np.shape(ht)) # activity for whole sequence
            # print('hT', np.shape(hT)) # activity for last element of sequence

            y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

        return y