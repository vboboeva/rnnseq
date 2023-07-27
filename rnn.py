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
from train import train
from train import predict


class RNN(nn.Module):
    def __init__(self, input_num_units, hidden_num_units, num_layers, output_num_units, \
            nonlinearity="tanh", device="cpu"):
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

        # ht = sequence of hidden states
        # hT = last hidden state
        # ht, hT = self.rnn(x, h0)
        # print('x',np.shape(x))
        ht, hT = self.rnn(x)

        # whole sequence of hidden states, linearly transformed
        y = self.fc(ht)
        y = F.softmax(self.fc(ht), dim=-1)

        return ht, hT, y


if __name__ == "__main__":

    # sequence parameters 
    L=5
    m=2
    whichloss='CE'

    # network parameters
    n_hidden = 100
    n_layers = 1

    n_epochs = 300
    batch_size = 10

    # fraction of data used to train
    frac_train=.5

    # load the number of inputs
    alpha = len(loadtxt('input/alphabet.txt', dtype='str'))

    print(alpha)

    # number of initial letters to cue net with
    start=4

    # make a dictionary
    letter_to_index = {}
    keys = list(string.ascii_lowercase)[:alpha]
    values = np.arange(alpha)
    for i, k in enumerate(keys):
        letter_to_index[k] = values[i]

    index_to_letter = {}
    keys = np.arange(alpha)
    values = list(string.ascii_lowercase)[:alpha]
    for i, k in enumerate(keys):
        index_to_letter[k] = values[i]

    print(index_to_letter)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # input_num_units, hidden_num_units, num_layers, output_num_units
    model = RNN(alpha, n_hidden, n_layers, alpha, device=device)

    # load types
    types = np.array(loadtxt('input/structures_L%d_m%d.txt'%(L, m), dtype='str')).reshape(-1)

    all_tokens=[]
    # load all the tokens corresponding to that type
    for t, type_ in enumerate(types[:1]):
        print('type_', type_)
        tokens = loadtxt('input/%s.txt'%type_, dtype='str')
        tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
        all_tokens.append(tokens_arr)
    all_tokens = np.vstack(all_tokens)
    
    # count how many transitions of each kind we have
    count=np.zeros((alpha, alpha))
    for i in range(len(all_tokens)):
        for j in range(L-1):
            x = letter_to_index[all_tokens[i,j]]
            y = letter_to_index[all_tokens[i,j+1]]
            count[x,y]+=1
    count /= np.sum(count, axis=1)[:, None]

    # turn letters into one hot vectors
    x = torch.zeros((L, len(all_tokens), alpha), dtype=torch.float32)
    for i, token in enumerate(all_tokens):
        # print(token)
        pos = []
        for letter in token:
            pos = np.append(pos, letter_to_index[letter])
        x[:,i,:] = F.one_hot(torch.tensor(pos.astype(int)), alpha)

    '''
    make train and test data

    '''    
    n_train = int(frac_train*len(all_tokens))
    n_test = len(all_tokens) - n_train

    print(f"{n_train} training sequences")
    print(f"{n_test} testing sequences")

    # torch.manual_seed(0)
    # print("seed", torch.manual_seed(0))
    ids = torch.randperm(len(all_tokens))
    # print('ids', ids)
    train_ids = ids[:n_train]
    test_ids = ids[n_train:]

    tokens_train=all_tokens[train_ids,:]
    tokens_test=all_tokens[test_ids,:]

    '''
    train and test network

    '''

    n_batches = len(train_ids)//batch_size
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

    ###################################################
    
    train_losses, test_losses, train_accuracies, test_accuracies = train(x, train_ids, test_ids, tokens_train, tokens_test, model, optimizer, whichloss, L, n_epochs, n_batches, batch_size, alpha, letter_to_index, index_to_letter, start)

    ###################################################
    # X_train:  L x len(trainingdata) x alpha


    Wio=np.dot( model.fc.state_dict()["weight"].detach().cpu().numpy(),
         model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy() )

    A=np.vstack((train_losses, test_losses))
    B=np.vstack((train_accuracies, test_accuracies))

    np.savetxt('output/loss_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), A.T)
    np.savetxt('output/accuracy_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), B.T)
    np.savetxt('output/Wio_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), Wio)
    np.savetxt('output/count_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), count)

