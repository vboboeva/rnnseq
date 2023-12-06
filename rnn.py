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

def load_tokens(alpha, L, m, n_types, letter_to_index):
	# load types
	types = np.array(loadtxt('input/structures_L%d_m%d.txt'%(L, m), dtype='str')).reshape(-1)

	all_tokens=[]
	all_labels=[]
	# load all the tokens corresponding to that type
	if n_types > 0:
		types=types[:n_types]

	for t, type_ in enumerate(types):
		print('type_', type_)
		tokens = loadtxt('input/%s.txt'%type_, dtype='str')
		tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
		all_tokens.append(tokens_arr)
		all_labels.append(np.array(len(tokens_arr)*[t]))

	all_tokens = np.vstack(all_tokens)
	all_labels = np.hstack(all_labels)#.reshape(-1,1)

	# turn letters into one hot vectors
	n_types = np.max(all_labels) + 1
	X = torch.zeros((L, len(all_tokens), alpha), dtype=torch.float32)
	y = torch.zeros((len(all_labels), n_types), dtype=torch.float32)
	for i, (token, label) in enumerate(zip(all_tokens, all_labels)):
		pos = [letter_to_index[letter] for letter in token]
		X[:,i,:] = F.one_hot(torch.tensor(pos, dtype=int), alpha)
		y[i,:] = F.one_hot(torch.tensor([label]), n_types)

	return X, y, all_tokens, all_labels, np.shape(tokens)[0]

def generate_configurations(L, alphabet):
	configurations = list(product(alphabet, repeat=L))
	configurations = np.vstack([np.array(list(config)) for config in configurations])    
	return configurations

def remove_subset(configurations, subset):
	subset_as_arrays = [np.array(item) for item in subset]
	filtered = [config for config in configurations if not any(np.array_equal(config, sub) for sub in subset_as_arrays)]
	return np.array(filtered)

def savefiles(output_folder_name, sim, which_task, model, results):
	
	# Save the model state
	torch.save(model.state_dict(), '%s/model_state_sim%d.pth' % (output_folder_name, sim))
	# Save the connectivity
	np.save('%s/%s_sim%d' % (output_folder_name, 'Whh', sim), results['Whh'])

	# Save files not sim specific
	for key in results['Tokens'].keys():
		np.save('%s/%s_%s' % (output_folder_name, 'Tokens', key), results['Tokens'][key])	

	sub_results = {k: v for k, v in results.items() if k not in ['Tokens', 'Whh']}

	for key in sub_results.keys():
		nested_keys = [k for k in sub_results[key].keys()]
		for nk in nested_keys:
			if results[key][nk] != []:
				np.save('%s/%s_%s_sim%d' % (output_folder_name, key, nk, sim), results[key][nk])

###########################################
################## M A I N ################
###########################################

def main(
	L, m, sim, sim_datasplit,
	# network parameters
	n_hidden=40,
	n_layers=1,
	which_task=None,
	which_objective='CE',
	which_init=None,
	n_epochs=10,
	batch_size=10,
	learning_rate=0.01,
	frac_train=0.7,  # fraction of data to train net with
	n_repeats=1,  # max number of repeats of a given sequence
	n_types=-1,  # number of types to train net with: 1 takes just the first, -1 takes all
	alpha=5,  # length of alphabet
):

	print('DATASPLIT NO', sim_datasplit)
	print('SIMULATION NO', sim)

	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	X, y, all_tokens, all_labels, num_tokens_onetype = load_tokens(alpha, L, m, n_types, letter_to_index)

	# make train and test data
	n_train = int(frac_train * len(all_tokens))
	n_test = len(all_tokens) - n_train
	n_other = alpha ** L - n_train - n_test

	torch.manual_seed(sim_datasplit)
	ids = torch.arange(len(all_tokens)).reshape(-1, num_tokens_onetype)
	for i, ids_type in enumerate(ids):
		ids[i] = torch.take(ids_type, torch.randperm(len(ids_type)))

	num_types = int(len(all_tokens) / num_tokens_onetype)
	n_train_type = n_train // num_types
	train_ids = ids[:, :n_train_type].reshape(-1)
	test_ids = ids[:, n_train_type:].reshape(-1)

	X_train = X[:, train_ids, :]
	X_test = X[:, test_ids, :]
	y_train = y[train_ids, :]
	y_test = y[test_ids, :]

	tokens_train = all_tokens[train_ids, :]
	tokens_test = all_tokens[test_ids, :]
	labels_train = all_labels[train_ids]
	labels_test = all_labels[test_ids]

	all_configurations = generate_configurations(L, np.array(alphabet))
	tokens_other = remove_subset(all_configurations, all_tokens)

	# Train and test network
	torch.manual_seed(sim)
	n_batches = len(tokens_train) // batch_size

	# Dynamically determine the output size of the model
	task_to_output_size = {
		'Pred': alpha,
		'Class': num_types
	}

	output_size = task_to_output_size.get(which_task)
	if output_size is None:
		raise ValueError(f"Invalid task: {which_task}")

	# Create the model
	model = RNN(alpha, n_hidden, n_layers, output_size, device=device, which_init=which_init)

	# Set up the optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

	results = { 'Tokens': {
			'train': tokens_train,
			'test': tokens_test,
			'other': tokens_other
		},
	    'Loss': {
	        'train': [],
	        'test': []
	    },
	    'Retrieval': {
	        'train': [],
	        'test': [],
	        'other': []
	    },
	    'Accuracy': {
	        'train': [],
	        'test': []
	    }
	}

	# if which_task in task_functions:
	if which_task in ['Pred', 'Class']:
		task_results = train(X_train, X_test, y_train, y_test, model, optimizer, which_objective, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, results, which_task=which_task)
	else:
		print("Task not recognized!")
		return

	# Post-training operations

	results.update({
		'yh': {'train': model.get_activity(X_train).detach().cpu().numpy(),
		'test': model.get_activity(X_test).detach().cpu().numpy() 
		},
	'Whh': model.rnn.weight_hh_l0.detach().cpu().numpy()
	})

	return model, results

##################################################

if __name__ == "__main__":

	params = loadtxt("params.txt", dtype='int')

	main_kwargs = dict(
		# network parameters
		n_hidden = 50,
		n_layers = 1,
		which_task='Pred',  # Directly specify the task here
		which_objective='CE',
		which_init=None,
		n_epochs = 3000,
		batch_size = 10,
		learning_rate = 0.001,
		frac_train = 0.7,
		n_repeats = 1,
		n_types = -1,
		alpha = 5,
	)
	# parameters
	alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]

	L_col_index = 0
	m_col_index = 1
	sim_datasplit_col_index = 2
	sim_col_index = 3        
	index = int(sys.argv[1]) - 1

	size = 20
	for i in range(size):
		row_index = index * size + i
		sim = params[row_index, sim_col_index]
		L = params[row_index, L_col_index]
		m = params[row_index, m_col_index]
		sim_datasplit = params[row_index, sim_datasplit_col_index]

		output_folder_name = 'Task%s_N%d_L%d_m%d_nepochs%d_lr%.5f_ntypes%d_obj%s_init%s_datasplit%s' % (
		main_kwargs['which_task'], main_kwargs['n_hidden'], L, m, main_kwargs['n_epochs'], main_kwargs['learning_rate'], main_kwargs['n_types'], main_kwargs['which_objective'], main_kwargs['which_init'], sim_datasplit)

		os.makedirs(output_folder_name, exist_ok=True)

		model, results = main(L, m, sim, sim_datasplit, **main_kwargs)
		savefiles(output_folder_name, sim, main_kwargs['which_task'], model, results)


######### if using repetitions of train data ####
# tokens_train_repeated, X_train_repeated = make_repetitions(tokens_train, X_train, n_repeats)
	
# X_train_repeated=torch.tensor(X_train_repeated)

# # make sure batch size is not larger than total amount of data
# if len(tokens_train_repeated) <= batch_size:
#     batch_size = len(tokens_train_repeated)    

# n_batches = len(tokens_train_repeated)//batch_size
##################################################
