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
        # exit()
        # y = F.softmax(self.fc(ht), dim=2)

        return ht, hT, y


def predict(letter_to_index, index_to_letter, seq_start, next_letters):
    # model.eval()

    # print('seqstart', seq_start)
    with torch.no_grad():

    # starts with a sequence of words of given length, initializes network

        # goes through each of the seq_start we want to predict
        for i in range(0, next_letters):
            x = torch.zeros((len(seq_start), alpha), dtype=torch.float32)
            pos = [letter_to_index[w] for w in seq_start[i:]]
            # print('pos',pos)
            for k, p in enumerate(pos):
                # print(k)
                x[k,:]= F.one_hot(torch.tensor(p), alpha)
            # print('heree', x)
            # y_pred should have dimensions 1 x L-1 x alpha, ours has dimension L x 1 x alpha, so permute

            # x has to have dimensions (L, sizetrain, alpha)

            a, b, y_pred = model(x)#.permute(1,0,2)
            # print('y_pred shape', np.shape(y_pred))
            # print('y_pred', y_pred)
            

            # last_letter_logits has dimension alpha
            last_letter_logits = y_pred[-1,:]
            # print('logit', last_letter_logits)
            # print('shape logit', np.shape(last_letter_logits))

            # applies a softmax to transform activations into a proba, has dimensions alpha
            proba = torch.nn.functional.softmax(last_letter_logits, dim=0).detach().numpy()
            # print('proba tensor', proba)
            # print('sum=', np.sum(proba))

            # then samples randomly from that proba distribution 
            letter_index = np.random.choice(len(last_letter_logits), p=proba)

            # print(letter_index)

            # appends it into the sequence produced
            seq_start.append(index_to_letter[letter_index])

    # print(seq_start)
    return seq_start

if __name__ == "__main__":

    # sequence parameters 
    L=5
    m=2
    whichloss='CE'

    # network parameters
    n_hidden = 100
    n_layers = 1

    n_epochs = 400
    batch_size = 10

    # fraction of data used to train
    frac_train=.95

    # load the number of inputs
    alpha = len(loadtxt('input/alphabet.txt', dtype='str'))

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
    for t, type_ in enumerate(types[:2]):
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
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    n_batches = len(train_ids)//batch_size
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

    X_train = x[:,train_ids,:]
    X_test = x[:,test_ids,:]

    # for _ in range(n_epochs):
    for _ in tqdm(range(n_epochs)):
        '''
        Calculate test error and accuracy
        '''
        with torch.no_grad():
            X_test = X_test.to(model.device)
            ht, hT, y_test = model(X_test)
            y_test = y_test.to(model.device)
            if whichloss == 'CE':
                loss = F.cross_entropy(y_test[:-1].permute(1,2,0), X_test[1:].permute(1,2,0).softmax(dim=1), reduction='mean')
            elif whichloss == 'MSE':
                loss = F.mse_loss(y_test[:-1], X_test[1:], reduction='mean')
            else:
                print('Loss function not recognized!')
            test_losses.append(loss.item())
            label = torch.argmax(X_test, dim=-1)
            pred = torch.argmax(y_test, dim=-1)
            test_accuracies.append( pred.eq(label).sum().item() / (n_test*L) )
        '''
        Calculate train error
        '''
        with torch.no_grad():
            X_train = X_train.to(model.device)
            ht, hT, y_train = model(X_train)
            y_train = y_train.to(model.device)
            
            if whichloss == 'CE':
                loss = F.cross_entropy(y_train[:-1].permute(1,2,0), X_train[1:].permute(1,2,0).softmax(dim=1), reduction='mean')
            elif whichloss == 'MSE':
                loss = F.mse_loss(y_train[:-1], X_train[1:], reduction='mean')
            else:
                print('Loss function not recognized!')

            train_losses.append(loss.item())
            label = torch.argmax(X_train, dim=-1)
            pred = torch.argmax(y_train, dim=-1)
            train_accuracies.append( pred.eq(label).sum().item() / (n_train*L) )

        '''
        train the network to produce the next letter
        '''

        # shuffle training data so that in each epoch data is split randomly in batches for training

        _ids = torch.randperm(train_ids.size(0))
        train_ids = train_ids[_ids]
        # np.random.shuffle(train_ids)

        # we are training in batches
        accuracy = 0
        for batch in range(n_batches):
            optimizer.zero_grad()

            batch_start = batch * batch_size
            batch_end = (batch + 1) * batch_size

            X_batch = x[:, train_ids[batch_start:batch_end], :]
            X_batch = X_batch.to(model.device)
            # print(np.shape(X_batch))

            ht, hT, y_batch = model(X_batch)
            y_batch = y_batch.to(model.device)

            if whichloss == 'CE':
                loss = F.cross_entropy(y_batch[:-1].permute(1,2,0), X_batch[1:].permute(1,2,0).softmax(dim=1), reduction='mean')
            elif whichloss == 'MSE':
                loss = F.mse_loss(y_batch[:-1], X_batch[1:], reduction='mean')
            else:
                print('Loss function not recognized!')

            loss.backward()
            optimizer.step()

    start=4

    # X_train:  L x len(trainingdata) x alpha

    in_train=[]
    in_test=[]
    in_none=[]
    for i in range(len(X_train[0,:])):
        se= tokens_train[i]
        seq = [se[j] for j in range(len(se))]
        pred_seq = predict(letter_to_index, index_to_letter, seq[:start], L-start)
        if (tokens_train == pred_seq).all(axis=1).any():
            print('in train', pred_seq)
            in_train += [pred_seq]
        elif (tokens_test == pred_seq).all(axis=1).any():
            print('in test', pred_seq)
            in_test += [pred_seq]
        else:
            print('in none', pred_seq)
            in_none +=[pred_seq]
    print('len in_train', len(in_train))
    print('len in_test', len(in_test))
    print('len in_none', len(in_none))

    Wio=np.dot( model.fc.state_dict()["weight"].detach().cpu().numpy(),
         model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy() )

    A=np.vstack((train_losses, test_losses))
    B=np.vstack((train_accuracies, test_accuracies))

    np.savetxt('output/loss_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), A.T)
    np.savetxt('output/accuracy_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), B.T)
    np.savetxt('output/Wio_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), Wio)
    np.savetxt('output/count_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), count)

