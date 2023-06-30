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


class RNN(nn.Module):
    def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units):
        super(RNN, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_num_units = hidden_num_units
        self.num_layers = num_layers

        # RNN layers
        self.rnn = nn.RNN(input_num_units, hidden_num_units, num_layers)

        # Fully connected layer
        self.fc = nn.Linear(hidden_num_units, output_num_units)

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
        h0 = torch.randn(self.num_layers, x.shape[1], self.hidden_num_units)

        # ht = sequence of hidden states
        # hT = last hidden state
        # ht, hT = self.rnn(x, h0)
        ht, hT = self.rnn(x)

        # whole sequence of hidden states, linarly transformed
        y = self.fc(ht)

        return ht, hT, y


if __name__ == "__main__":

    # load the number of inputs
    alphabet = loadtxt('input/alphabet.txt', dtype='str')
    print(alphabet)
    nb_classes=len(alphabet)
    print(nb_classes)

    # input_num_units, hidden_num_units, num_layers, output_num_units
    model = RNN(nb_classes, 10, 1, 10)

    print(model)

    # for name, vals in model.rnn.named_parameters():
    #     print(name, "-->", vals.shape)
    #     print("-------------------------------")
    #     print(vals)
    #     print("\n")

    # exit()

    # x = torch.randn(5, 3, 10)
    # ht, hT, output = model(x)

    # print(ht[-1] - hT)

    # print("x = ", x.shape)
    # print("ht = ", ht.shape)
    # print("output = ", output.shape)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # cycles = 10
    # x0 = torch.sin(torch.arange(50)/50*np.pi*2 * cycles)
    # exit()

    train_losses = []
    for _ in tqdm(range(1000)):

        # sequence of 50 i.i.d. Gaussian numbers
        # x = torch.randn(50, 64, 10) #+ x0[:,None,None]
        # x = F.softmax(x, dim=-1)

        # load inputs
        structure = loadtxt('input/structures.txt', dtype='str')[0]
        print(structure)

        one_hot_targets = np.eye(nb_classes)[targets]
        print(one_hot_targets)

        exit()

        optimizer.zero_grad()

        '''
        train the network to reproduce the input. Expecting:
        - recurrent weights = 0 (no memory, as inputs are i.i.d.)
        - product W_out.W_in  prop to identity matrix
        '''
        ht, hT, y = model(x)
        loss = F.mse_loss(y, 3*x)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())





    fig, ax = plt.subplots()
    ax.plot(train_losses)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(2,2, figsize=(8,8))

    ax[0][0].set_title("weight . weight_ih_l0")
    im = ax[0][0].imshow(np.dot(
        model.fc.state_dict()["weight"].detach().cpu().numpy(),
        model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy()
        ))
    plt.colorbar(im, ax=ax[0][0])

    ax[0][1].set_title("weight_hh_l0")
    im = ax[0][1].imshow(model.rnn.state_dict()["weight_hh_l0"].detach().cpu().numpy())
    plt.colorbar(im, ax=ax[0][1])

    ax[1,0].set_title("weight_ih_l0")
    im = ax[1,0].imshow(model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy())
    plt.colorbar(im, ax=ax[1][0])

    ax[1,1].set_title("weight")
    im = ax[1,1].imshow(model.fc.state_dict()["weight"].detach().cpu().numpy())
    plt.colorbar(im, ax=ax[1][1])
    plt.show()
    plt.close(fig)
