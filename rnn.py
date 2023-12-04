import sys
import os
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
from train import train
from train import predict
from model import RNN
#from FixedPointFinderTorch import FixedPointFinderTorch as FixedPointFinder
#from plot_utils import plot_fps_subspace
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy
import scipy.cluster.hierarchy as sch
from itertools import permutations
from itertools import product
from matplotlib import rc
from pylab import rcParams

# # the axes attributes need to be set before the call to subplot
# rc('font',**{'family':'sans-serif','sans-serif':['Arial']}, size=10)
# rc('text', usetex=True)
# rc('axes', edgecolor='black', linewidth=0.5)
# rc('legend', frameon=False)
# rcParams['ytick.direction'] = 'in'
# rcParams['xtick.direction'] = 'in'
# rcParams['text.latex.preamble'] = r'\usepackage{sfmath}' # \boldmath


###################################################################################
# find fixed points (adapted from https://github.com/mattgolub/fixed-point-finder #
###################################################################################


def find_plot_fixed_points(model, Z):
    ''' Find, analyze, and visualize the fixed points of the trained RNN.

    Args:
        model: FlipFlop object.

            Trained RNN model, as returned by train_FlipFlop().

        Z: dict.

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
    initial_states = fpf.sample_states(Z,
        n_inits=N_INITS,
        noise_scale=NOISE_SCALE)

    # Study the system in the absence of input pulses (e.g., all inputs are 0)
    inputs = np.zeros([1, n_bits])

    # Run the fixed point finder
    unique_fps, all_fps = fpf.find_fixed_points(initial_states, inputs)

    # Visualize identified fixed points with overlaid RNN state trajectories
    # All visualized in the 3D PCA space fit the the example RNN states.
    fig = plot_fps_subspace(unique_fps, Z,
        plot_batch_idx=list(range(30)),
        plot_start_time=10)
    plt.tight_layout()
    fig.savefig('figs/test.jpg')
    return(unique_fps)

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

def load_tokens(L, m, n_types, letter_to_index):
    # load types
    types = np.array(loadtxt('input/structures_L%d_m%d.txt'%(L, m), dtype='str')).reshape(-1)

    all_tokens=[]
    # load all the tokens corresponding to that type
    if n_types > 0:
        types=types[:n_types]

    for t, type_ in enumerate(types):
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
    
    return x, all_tokens, np.shape(tokens)[0]

def generate_configurations(L, alphabet):
    configurations = list(product(alphabet, repeat=L))
    configurations = np.vstack([np.array(list(config)) for config in configurations])    
    return configurations

def remove_subset(configurations, subset):
    subset_as_arrays = [np.array(item) for item in subset]
    filtered = [config for config in configurations if not any(np.array_equal(config, sub) for sub in subset_as_arrays)]
    return np.array(filtered)

def savefiles(output_folder_name, sim, losses_train, losses_test, tokens_train, tokens_test, tokens_other,seq_retrieved_train, seq_retrieved_test, seq_retrieved_other, yh_train, yh_test, Whh):

    np.save('%s/loss_train_sim%d'%(output_folder_name, sim), losses_train)
    np.save('%s/loss_test_sim%d'%(output_folder_name, sim), losses_test)

    # tokens are not simulation dependent, only datasplit dependent!
    np.save('%s/tokens_train'%(output_folder_name), tokens_train)
    np.save('%s/tokens_test'%(output_folder_name), tokens_test)
    np.save('%s/tokens_other'%(output_folder_name), tokens_other)

    np.save('%s/seq_retrieved_train_sim%d'%(output_folder_name, sim), seq_retrieved_train)
    np.save('%s/seq_retrieved_test_sim%d'%(output_folder_name, sim), seq_retrieved_test)
    np.save('%s/seq_retrieved_other_sim%d'%(output_folder_name, sim), seq_retrieved_other)

    np.save('%s/yh_train_sim%d'%(output_folder_name, sim), yh_train)
    np.save('%s/yh_test_sim%d'%(output_folder_name, sim), yh_test)
    np.save('%s/Whh_sim%d'%(output_folder_name, sim), Whh)

    # np.savetxt('output/Wio_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,which_objective), Wio)
    # np.savetxt('output/count_L%d_m%d_nepochs%d_loss%s.txt'%(L,m,n_epochs,which_objective), count)

# take only training sequences and repeat some of them 
def make_repetitions(tokens_train, X_train, n_repeats):

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

    return tokens_train_repeated, X_train_repeated

###########################################
################## M A I N ################
###########################################

def main(
    L, m, sim, sim_datasplit,
    # network parameters
    n_hidden = 40,
    n_layers = 1,
    # training
    which_objective='CE',
    which_init=None,
    n_epochs = 10,
    batch_size = 10,
    learning_rate = 0.01,
    frac_train = 0.7, # fraction of data to train net with
    start = 1,   # number of initial letters to cue net with
    n_repeats = 1, # max number of repeats of a given sequence
    n_types = -1, # number of types to train net with: 1 takes just the first, -1 takes all
    alpha = 5, # length of alphabet
    ):

    output_folder_name = 'N%d_L%d_m%d_nepochs%d_lr%.5f_ntypes%d_obj%s_init%s_datasplit%s'%(n_hidden, L, m, n_epochs, learning_rate, n_types, which_objective, which_init, sim_datasplit)

    # this if statement creates problems
    # if not os.path.exists(output_folder_name):
    #     os.makedirs(output_folder_name)

    os.makedirs(output_folder_name, exist_ok=True)    

    print('DATASPLIT NO', sim_datasplit)
    print('SIMULATION NO', sim)

    letter_to_index, index_to_letter = make_dicts(alpha)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("device =", device)

    x, all_tokens, num_tokens_onetype = load_tokens(L, m, n_types, letter_to_index)

    # make train and test data

    n_train = int(frac_train*len(all_tokens))
    n_test = len(all_tokens) - n_train
    n_other = alpha**L - n_train - n_test

    torch.manual_seed(sim_datasplit)
    
    # no random splitting
    # ids = np.arange(len(all_tokens)) 
    
    # take all sequences and randomize them, split into train and test sets BALANCED
    # ids in a 2d array, with
    # - row -> type
    # - column -> token in type

    ids = torch.arange(len(all_tokens)).reshape(-1, num_tokens_onetype)
    for i, ids_type in enumerate(ids):
        ids[i] = torch.take(ids_type, torch.randperm(len(ids_type)))

    num_types = int(len(all_tokens)/num_tokens_onetype)
    n_train_type = n_train//num_types
    train_ids = ids[:,:n_train_type].reshape(-1)
    test_ids = ids[:,n_train_type:].reshape(-1)

    X_train = x[:,train_ids,:]
    X_test = x[:,test_ids,:]

    tokens_train = all_tokens[train_ids, :]
    tokens_test = all_tokens[test_ids, :]

    all_configurations = generate_configurations(L, np.array(alphabet))

    tokens_other = remove_subset(all_configurations, all_tokens)

    ######### if using repetitions of train data ####
    # tokens_train_repeated, X_train_repeated = make_repetitions(tokens_train, X_train, n_repeats)
        
    # X_train_repeated=torch.tensor(X_train_repeated)

    # # make sure batch size is not larger than total amount of data
    # if len(tokens_train_repeated) <= batch_size:
    #     batch_size = len(tokens_train_repeated)    

    # n_batches = len(tokens_train_repeated)//batch_size
    ##################################################

    ##########################
    # train and test network #
    ##########################
    torch.manual_seed(sim)

    n_batches = len(tokens_train)//batch_size

    # input_num_units, hidden_num_units, num_layers, output_num_units
    
    model = RNN(alpha, n_hidden, n_layers, alpha, device=device, which_init=which_init)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

    # X_train is dimension L x len(trainingdata) x alpha
    losses_train, losses_test, seq_retrieved_train, seq_retrieved_test, seq_retrieved_other = train(X_train, X_test, tokens_train, tokens_test, tokens_other, model, optimizer, which_objective, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, start)

    yh_train = model.get_activity(X_train)
    yh_test = model.get_activity(X_test)

    Whh = model.rnn.weight_hh_l0
    
    ##########################
    # Fixed Points           #
    ##########################

    # find fixed points
    # fps = find_plot_fixed_points(model, Z)
    # print(fps)

    # Wio=np.dot(model.fc.state_dict()["weight"].detach().cpu().numpy(),
    #  model.rnn.state_dict()["weight_ih_l0"].detach().cpu().numpy() )

    return output_folder_name, losses_train, losses_test, tokens_train, tokens_test, tokens_other, seq_retrieved_train, seq_retrieved_test, seq_retrieved_other, yh_train.detach().cpu().numpy(), yh_test.detach().cpu().numpy(), Whh.detach().cpu().numpy()

##################################################

if __name__ == "__main__":

    # parameters
    # load the number of inputs
    alpha=5
    alphabet = [string.ascii_lowercase[i] for i in range(alpha)]

    params=loadtxt("params.txt", dtype='int')

    main_kwargs = dict(# network parameters
        n_hidden = 50,
        n_layers = 1,

        # training
        which_objective='CE',
        which_init=None,
        n_epochs = 300,
        batch_size = 10,
        learning_rate = 0.01,
        frac_train = 0.7, # fraction of data to train net with
        start = 1,   # number of initial letters to cue net with
        n_repeats = 1, # max number of repeats of a given sequence
        n_types = -1, # number of types to train net with: 1 takes just the first, -1 takes all
        alpha = alpha,
    )

    # sequence parameters 
    L_col_index=0
    m_col_index=1
    sim_datasplit_col_index=2
    sim_col_index=3        
    index = int(sys.argv[1])-1
    
    size = 20
    for i in range(size):
        row_index = index * size + i
        # print(index, row_index)
        sim = params[row_index, sim_col_index]
        L = params[row_index, L_col_index]
        m = params[row_index, m_col_index]
        sim_datasplit = params[row_index, sim_datasplit_col_index]
        
        output_folder_name, losses_train, losses_test, tokens_train, tokens_test, tokens_other, seq_retrieved_train, seq_retrieved_test, seq_retrieved_other, yh_train, yh_test, Whh = main(L, m, sim, sim_datasplit, **main_kwargs)

        savefiles(output_folder_name, sim, losses_train, losses_test, tokens_train, tokens_test, tokens_other, seq_retrieved_train, seq_retrieved_test, seq_retrieved_other, yh_train, yh_test, Whh)