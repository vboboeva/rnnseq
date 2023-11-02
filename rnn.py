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
from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
from plot_utils import plot_fps_subspace
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import seaborn as sns
import scipy
import scipy.cluster.hierarchy as sch
from itertools import permutations


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

def z_score(X):
    # X: ndarray, shape (n_features, n_samples)
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X.T).T
    print(Xz)
    return Xz

###################################################################################
# find fixed points (adapted from https://github.com/mattgolub/fixed-point-finder #
###################################################################################

def find_plot_fixed_points(model, valid_predictions):
    ''' Find, analyze, and visualize the fixed points of the trained RNN.

    Args:
        model: FlipFlop object.

            Trained RNN model, as returned by train_FlipFlop().

        valid_predictions: dict.

            Model predictions on validation trials, as returned by
            train_FlipFlop().

    Returns:
        None.
    '''

    NOISE_SCALE = 0.2 # Standard deviation of noise added to initial states
    N_INITS =  20000 # 1024 The number of initial states to provide

    n_bits = alpha

    '''Fixed point finder hyperparameters. See FixedPointFinder.py for detailed
    descriptions of available hyperparameters.'''
    fpf_hps = {
        'max_iters': 1000,
        'lr_init': 1.,
        'outlier_distance_scale': 10.0,
        'verbose': True, 
        'super_verbose': True}

    # Setup the fixed point finder
    fpf = FixedPointFinder(model.rnn, **fpf_hps)

    '''Draw random, noise corrupted samples of those state trajectories
    to use as initial states for the fixed point optimizations.'''
    initial_states = fpf.sample_states(valid_predictions,
        n_inits=N_INITS,
        noise_scale=NOISE_SCALE)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    inputs = np.zeros([1, n_bits])

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)


    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the the example RNN states.
    fig = plot_fps_subspace(unique_fps, valid_predictions,
        plot_batch_idx=list(range(30)),
        plot_start_time=10)
    plt.tight_layout()
    fig.savefig('test.jpg')

    return(unique_fps)

#########################################################################
# do PCA (adapted from https://pietromarchesi.net/pca-neural-data.html) #
#########################################################################

def apply_PCA():


    # valid predictions is of size num_trials x L x N, we want to transform it to N x L x num_trials
    # Xl = valid_predictions.reshape(len(valid_predictions),-1)
    Xl = np.array(valid_predictions.permute(2, 1, 0)) 
    Xl = Xl.reshape(len(Xl), -1) # to make the array of shape N x L num_trials

    print('shape=', np.shape(Xl))

    # We then standardize the resulting matrix Xl, and fit and apply PCA to it.

    n_components=5
    trial_unique_types=types[:n_types]
    trial_types=np.repeat(trial_unique_types, n_train)
    trial_size=L

    Xl = z_score(Xl)
    pca = PCA() # PCA(n_components=n_components)
    Xl_p = pca.fit_transform(Xl.T).T

    # Our projected data Xl_p is in the form of a Q×TK array (number of components by number of time points times number of trials). To plot the components, I rearrange it into a dictionary:

    gt = {comp : {t_type : [] for t_type in trial_unique_types} for comp in range(n_components)}

    for comp in range(n_components):
        for i, t_type in enumerate(trial_types):
            t = Xl_p[comp, trial_size * i: trial_size * (i + 1)]
            gt[comp][t_type].append(t)

    for comp in range(n_components):
        for t_type in trial_unique_types:
            gt[comp][t_type] = np.vstack(gt[comp][t_type])

    # Now, accessing the dictionary as gt[component][orientation] returns an array of all trials of the selected orientation projected on the selected component. We then plot the projections of each trial on the first three components.

    pal = sns.color_palette('husl', 9)
    
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.show()

    fig, axes = plt.subplots(1, 5, figsize=[20, 6], sharey=True, sharex=True)
    for comp in range(5):
        print(comp)
        ax = axes[comp]
        for t, t_type in enumerate(trial_unique_types):
            data_ = gt[comp][t_type]
            for i in range(len(trial_types))[30:40]:
                ax.plot(np.arange(L), data_[i], label='%s'%tokens_train_repeated[i])
        ax.set_ylabel('PC {}'.format(comp+1))
    axes[1].set_xlabel('Time (s)')
    ax.legend()
    # fig.tight_layout()
    fig.savefig('PCA.jpg')

def cluster_correlation_matrix(which_clustering, W_hh):

    overlap_tokens=np.ndarray((n_train, n_train))

    fig, ax = plt.subplots(1, L, figsize=[30, 10])

    maxcorrval=0
    mincorrval=0

    maxconnval=0
    minconnval=0

    for index_timestep in range(L):
        for index_token1 in range(n_train):
            for index_token2 in range(n_train):
                overlap_tokens[index_token1, index_token2] = np.cumsum(np.dot(valid_predictions[index_token1,index_timestep,:], valid_predictions[index_token2,index_timestep,:]))/n_hidden

        if np.max(overlap_tokens) > maxcorrval:
            maxcorrval = np.max(overlap_tokens)
        if np.min(overlap_tokens) <= mincorrval:
            mincorrval = np.min(overlap_tokens)

        if np.max(overlap_tokens) > maxconnval:
            maxconnval = np.max(overlap_tokens)
        if np.min(overlap_tokens) <= minconnval:
            minconnval = np.min(overlap_tokens)

    for index_timestep in range(L):
        for index_token1 in range(n_train):
            for index_token2 in range(n_train):
                overlap_tokens[index_token1, index_token2] = np.cumsum(np.dot(valid_predictions[index_token1,index_timestep,:], valid_predictions[index_token2,index_timestep,:]))/n_hidden

        if which_clustering == 'timestep':

            idx=np.argsort(tokens_train_repeated[:,index_timestep])
            overlap_tokens_sorted=overlap_tokens[idx,:][:,idx]
            tokens_train_repeated_sorted=tokens_train_repeated[idx]
            ticklabels=["".join(i) for i in tokens_train_repeated_sorted.astype(str)]

        elif which_clustering == 'distance':

            pairwise_distances = sch.distance.pdist(overlap_tokens)
            linkage = sch.linkage(pairwise_distances, method='complete')
            cluster_distance_threshold = pairwise_distances.max()/2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
            idx = np.argsort(idx_to_cluster_array)

        tokens_train_repeated_sorted = tokens_train_repeated[idx]

        ticklabels=["".join(i) for i in tokens_train_repeated_sorted.astype(str)]

        overlap_tokens_sorted=overlap_tokens[idx, :][:, idx]

        im=ax[index_timestep].imshow(overlap_tokens_sorted, vmin=mincorrval, vmax=maxcorrval)
        ax[index_timestep].set_xticks(np.arange(n_train))
        ax[index_timestep].set_xticklabels(ticklabels, rotation=90)            
        ax[index_timestep].set_yticks(np.arange(n_train))
        ax[index_timestep].set_yticklabels(ticklabels)

        # im1=ax[1,index_timestep].imshow(W_hh, vmin=minconnval, vmax=maxconnval)

        fig.colorbar(im, ax=ax[index_timestep], fraction=0.046, pad=0.04)
        # fig.colorbar(im1, ax=ax[1, index_timestep])

    fig.savefig('Overlap_L%d_m%d_nepochs%d_ntypes%d_loss%s.png'%(L,m,n_epochs,n_types,whichloss))


def connectivity_matrix(valid_predictions, W_hh):
    fig, ax = plt.subplots(1, L, figsize=[30, 10])

    valid_predictions = valid_predictions.numpy()

    for index_t in range(L):
        list_indices = []
        list_tokens = []     
        # Wmean_hh = np.ndarray((alpha, alpha))       
        
        for i, a in enumerate(alphabet):
            
            # find the indices of the sequences corresponding to a given token
            index_token = np.where(tokens_train_repeated[:, index_t] == a)[0]

            # find the neuron indices that are active 
            # take only those neurons
            index_neuron = np.unique(np.where(valid_predictions[index_token, index_t, :] > 0.4)[1])

            # append these to the big list
            list_indices += list(index_neuron)
            list_tokens += list(np.repeat(a, len(index_neuron)))

            # ii, jj = np.meshgrid(index_neuron, index_neuron)
            
            # Wmean_hh[i,j] = np.mean(W_hh[index_neuron, index_neuron])
            # print('mesh', ii, jj)

        len_M = len(list_indices)

        # make a giant empty matrix the size of the big list
        M = np.ndarray((len_M, len_M))

        for i, j in enumerate(list_indices):
            for k, l in enumerate(list_indices):
                M[i,k] = W_hh[j, l]

        ax[index_t].set_xticks(np.arange(len_M))
        ax[index_t].set_xticklabels(list_tokens, rotation=90)            
        ax[index_t].set_yticks(np.arange(len_M))
        ax[index_t].set_yticklabels(list_tokens)

        im = ax[index_t].imshow(M)
        fig.colorbar(im, ax=ax[index_t], fraction=0.046, pad=0.04)

    fig.savefig('Connectivity_L%d_m%d_nepochs%d_ntypes%d_loss%s.png'%(L,m,n_epochs,n_types,whichloss))    

###########################################
################## M A I N ################
###########################################

if __name__ == "__main__":

    # sequence parameters 
    L=int(sys.argv[1])+1
    m=2

    # network parameters
    n_hidden = 40
    n_layers = 1

    # training
    
    whichloss='CE'
    n_simulations = 10
    n_epochs = 100
    batch_size = 10
    learning_rate = 0.001
    frac_train = 0.9 # fraction of data to train net with
    start = 1   # number of initial letters to cue net with
    n_repeats = 1 # max number of repeats of a given sequence
    n_types = -1 # number of types to train net with: 1 takes just the first, -1 takes all

    # load the number of inputs
    alphabet = loadtxt('input/alphabet.txt', dtype='str')
    alpha = len(alphabet)
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

    # print('n_train', n_train)

    losses_train=np.ndarray((n_simulations, n_epochs))
    losses_test=np.ndarray((n_simulations, n_epochs))
    performances_train=np.ndarray((n_simulations, n_epochs))
    performances_test=np.ndarray((n_simulations, n_epochs))
    performances_other=np.ndarray((n_simulations, n_epochs))

    seq_retrieved_train=np.ndarray((n_simulations, n_train))
    seq_retrieved_test=np.ndarray((n_simulations, n_test))

    for sim in np.arange(n_simulations):

        torch.manual_seed(1987)

        # input_num_units, hidden_num_units, num_layers, output_num_units
        model = RNN(alpha, n_hidden, n_layers, alpha, device=device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

        print('SIMULATION NO', sim)
        # take all sequences and randomize them, split into train and test sets
        ids = np.arange(len(all_tokens)) # torch.randperm(len(all_tokens))
        train_ids = ids[:n_train]
        test_ids = ids[n_train:]
        X_train = x[:,train_ids,:]
        X_test = x[:,test_ids,:]

        tokens_train = all_tokens[train_ids,:]
        tokens_test = all_tokens[test_ids,:]

        # take only training sequences and repeat some of them 
        tokens_train_repeated=[]
        X_train_repeated=[]

        for i in range(len(tokens_train)):

            random_number=random.randint(1, n_repeats)

            X_tostack = (np.repeat(X_train[:, i, :, np.newaxis], random_number, axis = 2)).permute(0,2,1)
            
            if i == 0:
                tokens_train_repeated = np.tile(tokens_train[i], (random_number, 1))
                X_train_repeated = X_tostack

            else:
                tokens_train_repeated = np.vstack((tokens_train_repeated, np.tile(tokens_train[i], (random_number, 1))))
                X_train_repeated = np.concatenate((X_train_repeated, X_tostack), axis = 1)
            
        # count=count(tokens_train_repeated)
        X_train_repeated=torch.tensor(X_train_repeated)

        # make sure batch size is not larger than total amount of data

        if len(tokens_train_repeated) <= batch_size:
            batch_size = len(tokens_train_repeated)    

        n_batches = len(tokens_train_repeated)//batch_size

        ##########################
        # train and test network #
        ##########################
        
        # X_train is dimension L x len(trainingdata) x alpha
        losses_train[sim,:], losses_test[sim,:], performances_train[sim,:], performances_test[sim,:], performances_other[sim,:], seq_retrieved_train[sim,:], seq_retrieved_test[sim,:], seq_retrieved_other, tokens_other = train(X_train_repeated, X_test, tokens_train_repeated, tokens_test, model, optimizer, whichloss, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, start)

        print(seq_retrieved_train)
        print(seq_retrieved_test)
        print(seq_retrieved_other)
        print(tokens_other)

        valid_predictions, W_hh = model.get_activity(X_train)
        
        ##########################
        # Connectivity           #
        ##########################

        connectivity_matrix(valid_predictions, W_hh)

        ##########################
        # Fixed Points           #
        ##########################

        # find fixed points
        # fps = find_plot_fixed_points(model, valid_predictions)
        # print(fps)

        # do PCA on activity of hidden layer
        # apply_PCA()

        ###################################
        # Dot product of the hidden layer #
        ###################################

        cluster_correlation_matrix('timestep', W_hh.detach().numpy() ) # either timestep or distance  

    ###################################################

    Wio=np.dot( model.fc.state_dict()["weight"].detach().cpu().numpy(),
         model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy() )

    np.savetxt('output/tokens_train_L%d_m%d_ntypes%d.txt'%(L, m, n_types), tokens_train, fmt="%s")
    np.savetxt('output/tokens_test_L%d_m%d_ntypes%d.txt'%(L, m, n_types), tokens_test, fmt="%s")
    np.savetxt('output/tokens_other_L%d_m%d_ntypes%d.txt'%(L, m, n_types), tokens_other, fmt="%s")

    np.savetxt('output/loss_train_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), losses_train)
    np.savetxt('output/loss_test_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), losses_test)

    np.savetxt('output/performance_train_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), performances_train)
    np.savetxt('output/performance_test_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), performances_test)
    np.savetxt('output/performance_other_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), performances_other)

    np.savetxt('output/seq_retrieved_train_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), seq_retrieved_train)
    np.savetxt('output/seq_retrieved_test_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), seq_retrieved_test)
    np.savetxt('output/seq_retrieved_other_L%d_m%d_nepochs%d_ntypes%d_loss%s.txt'%(L, m, n_epochs, n_types, whichloss), seq_retrieved_other)
    
    # np.savetxt('output/Wio_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), Wio)
    # np.savetxt('output/count_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,whichloss), count)