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
    def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units, nonlinearity="tanh"):
        super(RNN, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.hidden_num_units = hidden_num_units
        self.num_layers = num_layers

        # RNN layers
        self.rnn = nn.RNN(input_num_units, hidden_num_units, num_layers, nonlinearity=nonlinearity)

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

        # whole sequence of hidden states, linearly transformed
        y = self.fc(ht)

        return ht, hT, y


if __name__ == "__main__":

    # load the number of inputs
    alphabet = loadtxt('input/alphabet.txt', dtype='str')
    nb_classes=len(alphabet)
    
    L=5
    m=5

    # make a dictionary
    dicts = {}
    keys = list(string.ascii_lowercase)[:nb_classes]
    values = np.arange(nb_classes)
    for i, k in enumerate(keys):
        dicts[k] = values[i]

    # input_num_units, hidden_num_units, num_layers, output_num_units
    model = RNN(nb_classes, 100, 1, 10)

    

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # load types
    types = np.array(loadtxt('input/structures_L%d_m%d.txt'%(L, m), dtype='str')).reshape(-1)

    all_tokens=[]
    # load all the tokens corresponding to that type
    for t, type_ in enumerate(types):
        tokens = loadtxt('input/%s.txt'%type_, dtype='str')
        tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
        all_tokens.append(tokens_arr)
    all_tokens = np.vstack(all_tokens)

    
    # count how many transitions of each kind we have
    count=np.zeros((nb_classes, nb_classes))
    for i in range(len(all_tokens)):
        for j in range(L-1):
            x = dicts[all_tokens[i,j]]
            y = dicts[all_tokens[i,j+1]]
            count[x,y]+=1
    count /= np.sum(count, axis=1)[:,None]

    # create train data
    x = torch.zeros((L, len(all_tokens), nb_classes), dtype=torch.float32)
    for i, token in enumerate(tokens):
        pos = []
        for letter in token:
            pos = np.append(pos, dicts[letter])
        x[:,i,:] = F.one_hot(torch.tensor(pos.astype(int)), nb_classes)
    
    # train network
    train_losses = []
    for _ in tqdm(range(100)):
        
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

    Wio=np.dot( model.fc.state_dict()["weight"].detach().cpu().numpy(),
         model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy())
    
    fig, axs = plt.subplots(1,3, figsize=(14,4))
    
    axs[0].plot(train_losses)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Training loss')
    im1 = axs[1].imshow(Wio.T)#, vmin=0, vmax=1)
    axs[1].set_xticks(np.arange(nb_classes))
    axs[1].set_yticks(np.arange(nb_classes))
    axs[1].set_xticklabels([string.ascii_lowercase[i] for i in range(nb_classes)])
    axs[1].set_yticklabels([string.ascii_lowercase[i] for i in range(nb_classes)])

    im2 = axs[2].imshow(count)#, vmin=0, vmax=1)
    axs[2].set_xticks(np.arange(nb_classes))
    axs[2].set_yticks(np.arange(nb_classes))
    axs[2].set_xticklabels([string.ascii_lowercase[i] for i in range(nb_classes)])
    axs[2].set_yticklabels([string.ascii_lowercase[i] for i in range(nb_classes)])

    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
    fig.savefig('loss.png')


    '''
    plot SVD decomposition of weights
    Rows: 0 - input weights, 1 - recurrent weights, 2 - output weights
    Cols: 0 - raw, 1 - singular values S, 2 - left singular vectors U, 3 - right singular vectors V
    '''

    fig, axs = plt.subplots(3,4, figsize=(13,8))
    plt.tight_layout()

    axs[0,0].set_title("Weights")
    axs[0,1].set_title("Singular values")
    axs[0,2].set_title("Left singular vectors, U")
    axs[0,3].set_title("Right singular vectors, V")

    def plot_svd (weights, title, axs):
        axs[0].set_ylabel(title)
        im = axs[0].imshow(weights)
        plt.colorbar(im, ax=axs[0])

        U, S, Vh = np.linalg.svd(weights)

        # check that Uh.U and Vh.V are identity matrices
        print(title)
        # print("U = ", U.shape, np.linalg.norm(np.dot(U.T, U) -  np.eye(len(U)) ) )
        # print("Vh = ", Vh.shape, np.linalg.norm(np.dot(Vh, Vh.T) -  np.eye(len(Vh)) ) )

        axs[1].plot(S)
        im = axs[2].imshow(U)
        plt.colorbar(im, ax=axs[2])
        im = axs[3].imshow(Vh.T)
        plt.colorbar(im, ax=axs[3])

    weights = model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy()
    plot_svd(weights, "weight_ih_l0", axs[0])

    weights = model.rnn.state_dict()["weight_hh_l0"].detach().cpu().numpy()
    plot_svd(weights, "weight_hh_l0", axs[1])
    
    weights = model.fc.state_dict()["weight"].detach().cpu().numpy()
    plot_svd(weights, "weight_out", axs[2])

    fig.savefig('weights.png')
