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
import scipy
from itertools import permutations
from itertools import product
from matplotlib import rc
from pylab import rcParams

from train import train
import pickle

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

	print('total number of tokens=', np.shape(all_tokens)[0])
	return X, y, all_tokens, all_labels, np.shape(tokens)[0]

def generate_configurations(L, alphabet):
	configurations = list(product(alphabet, repeat=L))
	configurations = np.vstack([np.array(list(config)) for config in configurations])    
	return configurations

def remove_subset(configurations, subset):
	subset_as_arrays = [np.array(item) for item in subset]
	filtered = [config for config in configurations if not any(np.array_equal(config, sub) for sub in subset_as_arrays)]
	return np.array(filtered)

def make_results_dict(tokens_train, tokens_test, tokens_other):
	# Set up the dictionary that will contain results for each token
	results = {}
	for which_result in ['Loss', 'Retrieval', 'Accuracy', 'yh']:
		results.update({which_result:{}}) 
		for myset, label in (zip([tokens_train, tokens_test, tokens_other], ['train','test','other'])):
			results[which_result].update({label:{}}) 
			for idx_token, token in enumerate(myset):
				temp = ''.join(token)
				results[which_result][label].update({temp:[]})

	results.update({'Whh':[]})
	return results


def savefiles(output_folder_name, sim, which_task, model, results):
	
	# Save the model state
	torch.save(model.state_dict(), '%s/model_state_sim%d.pth' % (output_folder_name, sim))

	with open('%s/results_sim%d.pkl'% (output_folder_name, sim), 'wb') as handle:
	    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# Save snapshots of connectivity across training 
	# np.save('%s/%s_sim%d' % (output_folder_name, 'Whh', sim), results['Whh'])

	# sub_results = {k: v for k, v in results.items() if k not in ['Whh']}

	# for key in sub_results.keys():
	# 	nested_keys = [k for k in sub_results[key].keys()]
	# 	for nk in nested_keys:
	# 		if len(results[key][nk]) != 0:
	# 			if which_task == 'Class':
	# 				if key != 'Retrieval':
	# 					if nk != 'other':
	# 						print('entered if')
	# 						np.save('%s/%s_%s_sim%d' % (output_folder_name, key, nk, sim), results[key][nk])
	# 			elif which_task == 'Pred':
	# 				if key != 'Accuracy':
	# 						np.save('%s/%s_%s_sim%d' % (output_folder_name, key, nk, sim), results[key][nk])

def quick_plot(results):
	fig, ax = plt.subplots(1,2, figsize=(13,8))

	# mean_acc=np.zeros(len(results['Accuracy']['train'][next(iter(results['Accuracy']['train']))]))
	mean_loss=np.zeros(len(results['Loss']['train'][next(iter(results['Loss']['train']))]))

	count=0
	for tok in results['Accuracy']['train'].keys():
		ax[0].plot(results['Loss']['train'][tok])
		# ax[1].plot(results['Accuracy']['train'][tok])
		
		mean_loss+=results['Loss']['train'][tok]
		# mean_acc+=results['Accuracy']['train'][tok]
		count+=1
	
	ax[0].plot(mean_loss/count, color='black', lw=5)
	# ax[1].plot(mean_acc/count, color='black', lw=5)
	plt.show()

###########################################
################## M A I N ################
###########################################

from model import RNN
# from model_v import RNN

def main(
	learning_rate, n_hidden, sim, sim_datasplit,
	# network parameters
	n_layers=1,
	L = 4,
	m = 2,
	which_task=None,
	which_objective='CE',
	which_init=None,
	which_transfer='relu',
	n_epochs=10,
	batch_size=8,
	frac_train=0.7,  # fraction of data to train net with
	n_repeats=1,  # max number of repeats of a given sequence
	n_types=-1,  # number of types to train net with: 1 takes just the first, -1 takes all
	alpha=5,  # length of alphabet
	snap_freq=2
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
	n_batches = n_train // batch_size

	# Dynamically determine the output size of the model
	task_to_output_size = {
		'Pred': alpha,
		'Class': num_types
	}

	output_size = task_to_output_size.get(which_task)
	if output_size is None:
		raise ValueError(f"Invalid task: {which_task}")

	# n_epochs for which take a snapshot of neural activity
	epochs_snapshot = [snap_freq*i for i in range(0, int(n_epochs/snap_freq)+1)]

	# Create the model
	model = RNN(alpha, n_hidden, n_layers, output_size, nonlinearity=which_transfer, device=device, which_init=which_init)

	# Set up the optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

	# Set up the results dictionary
	results=make_results_dict(tokens_train, tokens_test, tokens_other)

	print('TRAINING NETWORK')
	if which_task in ['Pred', 'Class']:
		task_results = train(X_train, X_test, y_train, y_test, model, optimizer, which_objective, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, results, epochs_snapshot, which_task=which_task)
	else:
		print("Task not recognized!")
		return

	# Quick and dirty plot of loss (comment when running on cluster, for local use)
	quick_plot(results)

	return model, results

##################################################

if __name__ == "__main__":

	params = loadtxt("params_test.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1,
		L = 4,
		m = 2,
		which_task = 'Class',  # Specify task
		which_objective = 'CE',
		which_init = None,
		which_transfer='relu',
		n_epochs = 2000,
		batch_size = 98,
		frac_train = 0.7,
		n_repeats = 1,
		n_types = -1,
		alpha = 5,
		snap_freq = 10 # snapshot of net activity every snap_freq epochs
	)
	# parameters
	alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]

	lr_col_index = 0
	n_hidden_col_index = 1
	sim_datasplit_col_index = 2
	sim_col_index = 3        
	index = int(sys.argv[1]) - 1

	size = 1
	for i in range(size):
		row_index = index * size + i
		learning_rate = params[row_index, lr_col_index]
		
		n_hidden = int(params[row_index, n_hidden_col_index])
		sim_datasplit = int(params[row_index, sim_datasplit_col_index])
		sim = int(params[row_index, sim_col_index])

		output_folder_name = 'Task%s_N%d_L%d_m%d_nepochs%d_lr%.5f_bs%d_ntypes%d_obj%s_init%s_transfer%s_datasplit%s' % (
		main_kwargs['which_task'], n_hidden, main_kwargs['L'], main_kwargs['m'], main_kwargs['n_epochs'], learning_rate, main_kwargs['batch_size'], main_kwargs['n_types'], main_kwargs['which_objective'], main_kwargs['which_init'],  main_kwargs['which_transfer'], sim_datasplit)

		os.makedirs(output_folder_name, exist_ok=True)

		model, results = main(learning_rate, n_hidden, sim, sim_datasplit, **main_kwargs)

		print('SAVING FILES')
		savefiles(output_folder_name, sim, main_kwargs['which_task'], model, results)