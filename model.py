'''

RNN network

'''

import numpy as np
from numpy import loadtxt
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import string
import functools


# class RNN(nn.Module):
#     def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units, 
#         nonlinearity="tanh", device="cpu", which_init=None):

#         super(RNN, self).__init__()

#         # Defining the number of layers and the nodes in each layer
#         self.hidden_num_units = hidden_num_units
#         self.input_num_units = input_num_units
#         self.output_num_units = output_num_units
#         self.num_layers = num_layers

#         # RNN layer
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

#         # whole sequence of hidden states, linearly transformed
#         y = self.fc(ht)

#         # np.savetxt('hT_v.txt', hT[0].detach().numpy())
#         # np.save('hT_v.npy', hT[0].detach().numpy())

#         return ht, hT, y

#     def save(self, filename):
#         torch.save(self.state_dict(), filename)

#     def load(self, filename):
#         self.load_state_dict(torch.load(filename, map_location=self.device))

#     def grad_dict (self):
#         return OrderedDict({name:pars.grad for name, pars in self.named_parameters()})

#     def get_activity(self, x):

#         with torch.no_grad():

#             _x = x.clone().detach().to(self.device)

#             ht, hT, _  = self.forward(_x)

#             # print('ht', np.shape(ht)) # activity for whole sequence
#             # print('hT', np.shape(hT)) # activity for last element of sequence

#             y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

#         return y




############################################################################

def freeze(module):
    for name, pars in module.named_parameters():
        pars.requires_grad = False
        

class LinearWeightDropout(nn.Linear):
    '''
    Linear layer with weights dropout (synaptic failures)
    '''
    def __init__(self, in_features, out_features, drop_p=0.0, **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.drop_p = drop_p

    def forward(self, input):
        new_weight = (torch.rand((input.shape[0], *self.weight.shape), device=input.device) > self.drop_p) * self.weight[None, :, :]
        output = torch.bmm(new_weight, input[:, :, None])[:, :, 0] / (1 - self.drop_p)
        if self.bias is None:
            return output
        return output + self.bias


class Net(nn.Module):
    '''
    Base class for network models.
    Attribute function `init_weights` is a custom weight initialisation.
    '''

    def init_weights (self, scaling):

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
                print('f_in', f_in)
                exit()
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
            output_activation=None, # choose btw softmax for classification vs linear for regression tasks
            drop_l=None,
            nonlinearity='relu',
            layer_type=nn.Linear,
            which_init=None,
            model_filename=None, # file with model parameters
            to_freeze = [], # parameters to keep frozen; list with elements in ['i2h', 'h2h', 'h2o']
            from_file = [], # parameters to set from file; list with elements in ['i2h', 'h2h', 'h2o']
            bias=True,
            device="cpu",
            train_i2h = True,
        ):

        super(RNN, self).__init__()
        init=which_init
        self._from_file = []
        self._model_filename=model_filename

        self.device = device
        # Defining the number of layers and the nodes in each layer
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden

        # if train_i2h:
        #     self.i2h = nn.Linear (d_input, d_hidden, bias=bias)

        # else:
        #     # MANUALLY DEFINE INPUT WEIGHTS HERE
        #     self._input_weights = torch.cat([torch.eye(self.d_input), torch.zeros(self.d_hidden - self.d_input, self.d_input)]) 

        #     # _n_repeats = self.d_hidden // self.d_input
        #     # assert self.d_hidden % self.d_input == 0, \
        #     #     "Hidden layer size should be integer multiple of input size"
        #     # self._input_weights = torch.repeat_interleave(_input_weights, _n_repeats, dim=0)
            
        #     self._input_weights.requires_grad = False

        #     self.i2h = lambda x: torch.matmul( x, self._input_weights.T )

        self.i2h = nn.Linear (d_input, d_hidden, bias=bias)
        if 'i2h' in to_freeze:
            freeze(self.i2h)
        if 'i2h' in from_file:
            self._from_file += ['i2h.'+n for n,_ in self.i2h.named_parameters()]

        self.h2h = layer_type (d_hidden, d_hidden, bias=bias)
        if 'h2h' in to_freeze:
            freeze(self.h2h)
        if 'h2h' in from_file:
            self._from_file += ['h2h.'+n for n,_ in self.h2h.named_parameters()]

        self.h2o = nn.Linear (d_hidden, d_output, bias=bias)
        if 'h2o' in to_freeze:
            freeze(self.h2o)
        if 'h2o' in from_file:
            self._from_file += ['h2o.'+n for n,_ in self.h2o.named_parameters()]

        if nonlinearity in [None, 'linear']:
            self.phi = lambda x: x
        elif nonlinearity == 'relu':
            self.phi = lambda x: F.relu(x)
        elif nonlinearity == 'sigmoid':
            self.phi = lambda x: F.sigmoid(x)
        elif nonlinearity == 'tanh':
            self.phi = lambda x: F.tanh(x)
        else:
            raise NotImplementedError("activation function " + \
                            f"\"{nonlinearity}\" not implemented")

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

        self.init_weights (init)

    def init_weights(self, init, seed=None):
        
        _pars_dict = {}
        if self._model_filename is not None:
            # The parameters to be set from file are removed from the list of
            # parameters to which the initialisation rule is applied
            print(f"Loading parameters from file '{self._model_filename}'")
            _pars_dict = torch.load(self._model_filename)

        if seed is not None:
            torch.manual_seed(seed)

        if init is None:
            pass
        elif init == "Rich":
            # initialisation of the weights -- N(0, 1/n)
            init_f = lambda f_in: 1./f_in
        elif init == "Lazy":
            # initialisation of the weights -- N(0, 1/sqrt(n))
            init_f = lambda f_in: 1./np.sqrt(f_in)
        elif init == "Const":
            # initialisation of the weights independent of n
            init_f = lambda f_in: 0.001
        else:
            raise ValueError(
                f"Invalid init option '{init}'\n" + \
                 "Choose either None, 'Rich', 'Lazy' or 'Const'")
        
        for name, pars in self.named_parameters():
            if name in self._from_file:
                # print(pars)
                # print(_pars_dict[name])
                pars.data = _pars_dict[name]
            elif init is not None:
                if "weight" in name:
                    f_in = 1.*pars.data.size()[1]
                    std = init_f(f_in)
                    pars.data.normal_(0., std)

        # # check
        # for name, pars in self.named_parameters():
        #     print(name, "\t", torch.max(pars - _pars_dict[name]))
        # exit()
        # # end check



    def forward(self, x, mask=None, delay=0):
        '''
        x
        ---
        seq_length, batch_size, input_num_units
        or
        seq_length, input_num_units
        '''

        # print("x.shape (input) ", x.shape)
        # print("mask.shape ", mask.shape)

        if mask is not None:
            assert isinstance(mask, torch.Tensor) and mask.shape == (self.d_hidden,), \
                f"`mask` must be a 1D torch tensor with the same size as the hidden layer"
            mask = mask.to(self.device)
            _masking = lambda h: h * mask[None,:]
        else:
            _masking = lambda h: h

        # _shape = seq_length, batch_size, n_hidden
        # or 
        # _shape = seq_length, n_hidden
        # This is saved here to remove the batch dimension
        # after the processing, if it is not present in input
        _shape = *x.shape[:-1], self.d_hidden
        # print("_shape ", _shape)

        # If batch dimension missing, add it (in the middle -- as in the
        # default implementation of pytorch RNN module).
        # This allows to treat an input containing a batch of sequences
        # in the same way as a single sequence.
        # print(x.shape)
        if len(x.shape) == 2:
            x = torch.reshape(x, (x.shape[0], 1, x.shape[1]) )
        # print("x.shape (reshaped) ", x.shape)

        # initialization of net        
        # h0 = torch.randn(x.shape[1], self.d_hidden)
        h0 = torch.zeros(x.shape[1], self.d_hidden)
        
        # batch_size, n_hidden
        ht = _masking(h0)
        hidden = []
        
        # pad input with 0 along the time axis for `delay` time steps
        if delay != 0:
            assert isinstance(delay, int), "delay must be an integer"
            x = torch.cat([x, torch.zeros((delay,*x.shape[1:]))], dim=0)
            _shape = _shape[0]+delay, *_shape[1:]

        # t is the sequence of time-steps
        for t, xt in enumerate(x):

            # xt = batch_size, n_input
            # _input_weights = n_hidden, n_input
            # zt = batch_size, n_hidden

            # process input to feed into recurrent network
            zi = self.i2h (xt)

            zh = self.h2h (ht)
            z = self.phi (zh + zi)
            z = _masking( z )

            hidden.append(z)
            ht = z 

        hidden = torch.reshape(torch.stack(hidden), _shape)

        y = self.h2o(hidden)

        hT = torch.reshape(hidden[-1], (1, *hidden.shape[1:]))

        return hidden, hT, y

    def get_activity(self, x):

        with torch.no_grad():

            _x = x.clone().detach().to(self.device)

            ht, hT, _  = self.forward(_x)

            # print('ht', np.shape(ht)) # activity for whole sequence
            # print('hT', np.shape(hT)) # activity for last element of sequence

            y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

        return y


# if __name__ == "__main__":

#     rnn = RNN(4, 8, 1, 3)

#     x = torch.randn(2, 5, 4)
#     print("x", x)

#     rnn.forward(x)
