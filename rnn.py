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
import random
import sys
from train import train
from train import predict
from model import RNN

# count how many transitions of each kind we have
def count(M):    
    count=np.zeros((alpha, alpha))
    for i in range(len(M)):
        for j in range(L-1):
            x = letter_to_index[M[i,j]]
            y = letter_to_index[M[i,j+1]]
            count[x,y]+=1
    count /= np.sum(count, axis=1)[:, None]
    return count

# make a dictionary
def make_dicts(alpha):
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

    return letter_to_index, index_to_letter

def load_tokens(types, n_types):
    all_tokens=[]
    # load all the tokens corresponding to that type
    for t, type_ in enumerate(types[:n_types]):
        print('type_', type_)
        tokens = loadtxt('input/%s.txt'%type_, dtype='str')
        tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
        all_tokens.append(tokens_arr)
    all_tokens = np.vstack(all_tokens)

    # turn letters into one hot vectors
    x = torch.zeros((L, len(all_tokens), alpha), dtype=torch.float32)
    for i, token in enumerate(all_tokens):
        # print(token)
        pos = []
        for letter in token:
            pos = np.append(pos, letter_to_index[letter])
        x[:,i,:] = F.one_hot(torch.tensor(pos.astype(int)), alpha)

    return x, all_tokens


if __name__ == "__main__":

    # sequence parameters 
    L=int(sys.argv[1])+2
    m=2

    # network parameters
    n_hidden = 100
    n_layers = 1

    # training
    
    whichloss='CE'
    n_simulations = 10
    n_epochs = 300
    batch_size = 20
    learning_rate = 0.001
    frac_train = 0.8 # fraction of data to train net with
    start = L-1   # number of initial letters to cue net with
    n_repeats = 1 # max number of repeats of a given sequence
    n_types = -1 # number of types to train net with: 1 takes just the first, -1 takes all

    # load the number of inputs
    alpha = len(loadtxt('input/alphabet.txt', dtype='str'))
    print(alpha)

    letter_to_index, index_to_letter = make_dicts(alpha)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device =", device)


    # load types
    types = np.array(loadtxt('input/structures_L%d_m%d.txt'%(L, m), dtype='str')).reshape(-1)

    x, all_tokens = load_tokens(types, n_types)

    # make train and test data

    n_train = int(frac_train*len(all_tokens))
    n_test = len(all_tokens) - n_train

    print('n_train', n_train)

    train_losses=np.ndarray((n_simulations, n_epochs))
    test_losses=np.ndarray((n_simulations, n_epochs))
    train_accuracies=np.ndarray((n_simulations, n_epochs))
    test_accuracies=np.ndarray((n_simulations, n_epochs))

    for sim in np.arange(n_simulations):

        # input_num_units, hidden_num_units, num_layers, output_num_units
        model = RNN(alpha, n_hidden, n_layers, alpha, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

        print('SIMULATION NO', sim)
        # take all sequences and randomize them, split into train and test sets
        ids = torch.randperm(len(all_tokens))
        train_ids = ids[:n_train]
        test_ids = ids[n_train:]
        X_train = x[:,train_ids,:]
        X_test = x[:,test_ids,:]

        print('len train before repeats', np.shape(X_train))

        tokens_train = all_tokens[train_ids,:]
        tokens_test = all_tokens[test_ids,:]

        # take only training sequences and repeat some of them 
        tokens_train_repeated=[]
        X_train_repeated=[]
    
        for i in range(len(tokens_train)):

            random_number=random.randint(1, n_repeats)

            X_tostack = (np.repeat(X_train[:, i, :, np.newaxis], random_number, axis=2)).permute(0,2,1)
            
            if i == 0:
                tokens_train_repeated = np.tile(tokens_train[i], (random_number,1))
                X_train_repeated = X_tostack

            else:
                tokens_train_repeated = np.vstack((tokens_train_repeated, np.tile(tokens_train[i], (random_number,1))))
                X_train_repeated = np.concatenate((X_train_repeated, X_tostack), axis = 1)
            
        # count=count(tokens_train_repeated)
        X_train_repeated=torch.tensor(X_train_repeated)

        print('len train after repeats', np.shape(X_train_repeated))

        # make sure batch size is not larger than total amount of data

        if len(tokens_train_repeated) <= batch_size:
            batch_size = len(tokens_train_repeated)    

        n_batches = len(tokens_train_repeated)//batch_size

        ###################################################
        # train and test network
        
        # train_losses, test_losses, train_accuracies, test_accuracies = train(X_train, X_test, train_ids, test_ids, tokens_train, tokens_test, model, optimizer, whichloss, L, n_epochs, n_batches, batch_size, alpha, letter_to_index, index_to_letter, start)

        # X_train is dimension L x len(trainingdata) x alpha
        train_losses[sim,:], test_losses[sim,:], train_accuracies[sim,:], test_accuracies[sim,:] = train(X_train_repeated, X_test, tokens_train_repeated, tokens_test, model, optimizer, whichloss, L, n_epochs, n_batches, batch_size, alpha, letter_to_index, index_to_letter, start)

        # print(train_losses[sim,:])

    ###################################################

    Wio=np.dot( model.fc.state_dict()["weight"].detach().cpu().numpy(),
         model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy() )

    np.savetxt('output/loss_train_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), train_losses)
    np.savetxt('output/loss_test_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), test_losses)

    np.savetxt('output/accuracy_train_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), train_accuracies)
    np.savetxt('output/accuracy_test_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), test_accuracies)
    
    # np.savetxt('output/Wio_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), Wio)
    # np.savetxt('output/count_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), count)

