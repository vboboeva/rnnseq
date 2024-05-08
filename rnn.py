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
from functools import partial
from matplotlib import rc
from pylab import rcParams
import pickle

from train import train
from train import test
from model import RNN, LinearWeightDropout
from quick_plot import plot_weights
from quick_plot import plot_loss
from quick_plot import plot_accuracy_ablation
import json

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
	return X, y, all_tokens, np.shape(tokens)[0]

def generate_configurations(L, alphabet):
	configurations = list(product(alphabet, repeat=L))
	configurations = np.vstack([np.array(list(config)) for config in configurations])    
	return configurations

def remove_subset(configurations, subset):
	subset_as_arrays = [np.array(item) for item in subset]
	filtered = [config for config in configurations if not any(np.array_equal(config, sub) for sub in subset_as_arrays)]
	return np.array(filtered)

def make_tokens(all_tokens, sim_datasplit, num_tokens_onetype, L, alpha, frac_train, X, y):
	# make train and test data
	n_train = int(frac_train * len(all_tokens))
	n_test = len(all_tokens) - n_train
	n_other = alpha ** L - n_train - n_test

	print('number of train', n_train)
	print('number of test', n_test)

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

	return X_train, X_test, y_train, y_test, tokens_train, tokens_test, n_train, n_test, n_other, num_types

def make_results_dict(which_task, tokens_train, tokens_test, tokens_other, ablate):

	# Set up the dictionary that will contain results for each token
	results = {}

	if which_task == 'Pred':
		metric='Retrieval'
	if which_task == 'Class':
		metric='Accuracy'

	for which_result in ['Loss', metric, 'yh']:
		results.update({which_result:{}}) 
		for myset, label in (zip([tokens_train, tokens_test, tokens_other], ['train','test','other'])):
			results[which_result].update({label:{}}) 
			for idx_token, tok in enumerate(myset):
				token = ''.join(tok)
				results[which_result][label].update({token:{}})
				results[which_result][label][token].update({0:[]})
				if ablate == True: 
					for unit_ablated in range(1, n_hidden+1):
						results[which_result][label][token].update({unit_ablated:[]})
	results['Whh']=[]

	return results

def tokenwise_test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, ablate):

	for (whichset, X, y, tokens) in zip(['train', 'test'], [X_train, X_test], [y_train, y_test], [tokens_train, tokens_test]):

		X = X.permute((1,0,2))
		# print("X after ", X.shape)

		if ablate == False:
			range_ablate=1
		else:
			range_ablate=n_hidden+1

		# ablate units one by one
		for idx_ablate in range(range_ablate):

			for (_X, _y, token) in zip(X, y, tokens):
				token = ''.join(token)

				metric, loss, yh = test(_X, _y, token, whichset, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task,
					idx_ablate=idx_ablate, n_hidden=n_hidden)
			
				if which_task == 'Pred':

					for sett in ['train', 'test', 'other']:

						if metric in results['Retrieval'][sett]:
							results['Retrieval'][sett][metric][idx_ablate].append(1)
							# print(metric, sett)

				elif which_task == 'Class':
					results['Accuracy'][whichset][token][idx_ablate].append(metric)

				results['Loss'][whichset][token][idx_ablate].append(loss)
				results['yh'][whichset][token][idx_ablate].append(yh)


# def Hypo1():
# 	# HYPOTHESIS 1
# 	mydict = {}
# 	# Identify all sequences with letter in a given position
# 	for letter in alphabet:
# 		mydict.update({letter:{}})
# 		# print('letter', letter)
# 		for position in range(L):
# 			# print('position', position)
# 			mydict[letter].update({position:{}})
# 			where = np.where(tokens_train[:, position] == letter)
			
# 			X = X_train.permute((1,0,2))
# 			y = y_train
# 			tokens = tokens_train

# 			X = X[where]
# 			y = y[where]
# 			tokens = tokens_train[where]
# 			# print('tokens', tokens)

# 			mydict[letter][position].update({'Loss':[]})
# 			mydict[letter][position].update({'Accuracy':[]})
# 			# ablate units one by one, the zeroth element is with no ablation

# 			for idx_ablate in range(n_hidden+1):

# 				# print('ablating unit %s'%idx_ablate)
# 				accuracy_mean=0
# 				loss_mean=0
# 				# Without and without ablation evaluate classification accuracy on this set of sequences
# 				for (_X, _y, _token) in zip(X, y, tokens):
# 					accuracy, loss, yh = test(_X, _y, _token, whichset, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task, idx_ablate, ablate, n_hidden)
# 					accuracy_mean=accuracy_mean+accuracy
# 					loss_mean=loss_mean+loss
# 				accuracy_mean=accuracy_mean/np.shape(tokens)[0]
# 				loss_mean=loss_mean/np.shape(tokens)[0]
# 				# print(accuracy_mean)

# 				mydict[letter][position]['Loss'].append(loss_mean)
# 				mydict[letter][position]['Accuracy'].append(accuracy_mean)				


def savefiles(output_folder_name, sim, which_task, model, results):
	
	# Save the model state
	torch.save(model.state_dict(), '%s/model_state_sim%d.pth' % (output_folder_name, sim))

	with open('%s/results_sim%d.pkl'% (output_folder_name, sim), 'wb') as handle:
	    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# # Convert and write JSON object to file
	# with open('%s/results_sim%d.json'% (output_folder_name, sim), 'wb') as handle:
	# 	_dumps = json.dumps(results)
	# 	json.dump(_dumps, handle)

###########################################
################## M A I N ################
###########################################

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
	batch_size=7,
	frac_train=0.7,  # fraction of data to train net with
	n_repeats=1,  # max number of repeats of a given sequence
	n_types=-1,  # number of types to train net with: 1 takes just the first, -1 takes all
	alpha=5,  # length of alphabet
	snap_freq=2,
	drop_connect = 0.,
	weight_decay = 0.,
	ablate=True
):
	print('DATASPLIT NO', sim_datasplit)
	print('SIMULATION NO', sim)

	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	X, y, all_tokens, num_tokens_onetype = load_tokens(alpha, L, m, n_types, letter_to_index)

	X_train, X_test, y_train, y_test, tokens_train, tokens_test, n_train, n_test, n_other, num_types = make_tokens(all_tokens, sim_datasplit, num_tokens_onetype, L, alpha, frac_train, X, y)

	all_configurations = generate_configurations(L, np.array(alphabet))
	tokens_other = remove_subset(all_configurations, all_tokens)

	# Train and test network
	torch.manual_seed(sim)
	n_batches = n_train // batch_size

	# print('n_batches', n_batches)

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

	if drop_connect != 0.:
		layer_type = partial(LinearWeightDropout, drop_p=drop_connect)
	else:
		layer_type = nn.Linear

	# Create the model
	model = RNN(alpha, n_hidden, n_layers, output_size, nonlinearity=which_transfer, device=device, which_init=which_init, layer_type=layer_type)
	
	print(model)
	print(n_hidden)

	# Set up the optimizer
	optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

	# Set up the results dictionary
	results = make_results_dict(which_task, tokens_train, tokens_test, tokens_other, ablate)

	def stats(x):
		try:
			_res = torch.mean(x), torch.std(x)
		except:
			_res = None
		return _res

	print('TRAINING NETWORK')

	if which_task in ['Pred', 'Class']:

		for epoch in range(n_epochs + 1):
			# print("X_train.shape (before) ", X_train.shape)
			if epoch in epochs_snapshot:
				print(epoch)
				# COPY THE WEIGHTS WHEN YOU SAVE THEM
				results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				tokenwise_test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, ablate)
					
			train(X_train, y_train, model, optimizer, which_objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, which_task=which_task, weight_decay=weight_decay)

			# After training: looking for motifs
			# Hypo1()	
	else:
		print("Task not recognized!")
		return

	# Quick and dirty plot of l:qoss (comment when running on cluster, for local use)
	# plot_weights(results, 5)
	plot_loss(n_types, n_hidden, ablate, results)
	# plot_accuracy_ablation(n_hidden, alphabet, L, mydict)

	return model, results

##################################################

if __name__ == "__main__":

	_L=4
	_m=2
	params = loadtxt('params_L%d_m%d.txt'%(_L,_m))
	# params = loadtxt("params_test.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1,
		L = 4,
		m = 2,
		which_task = 'Pred',  # Specify task
		which_objective = 'CE',
		which_init = None,
		which_transfer='relu',
		n_epochs = 2000,
		batch_size = 1, #16, # GD if = size(training set), SGD if = 1
		frac_train = 0.7,
		n_repeats = 1,
		n_types = -1, # set minimum 2 for task to make sense
		alpha = 5,
		snap_freq = 20, # snapshot of net activity every snap_freq epochs
		drop_connect = 0.,
		# weight_decay = 0.2, # weight of L1 regularisation
		ablate = False
	)

	# parameters
	alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]

	lr_col_index = 0
	n_hidden_col_index = 1
	sim_datasplit_col_index = 2
	sim_col_index = 3        
	index = int(sys.argv[1]) - 1

	# size is the number of serial simulations running on a single node of the cluster, set this accordingly with the number of arrays in order to cover all parameters in the parameters.txt file

	size = 1
	for i in range(size):
		row_index = index * size + i
		learning_rate = params[row_index, lr_col_index]
		
		n_hidden = int(params[row_index, n_hidden_col_index])
		sim_datasplit = int(params[row_index, sim_datasplit_col_index])
		sim = int(params[row_index, sim_col_index])

		output_folder_name = 'Task%s_N%d_L%d_m%d_nepochs%d_lr%.5f_bs%d_ntypes%d_obj%s_init%s_transfer%s_datasplit%s_ablate%s' % (
		main_kwargs['which_task'], n_hidden, main_kwargs['L'], main_kwargs['m'], main_kwargs['n_epochs'], learning_rate, main_kwargs['batch_size'], main_kwargs['n_types'], main_kwargs['which_objective'], main_kwargs['which_init'],  main_kwargs['which_transfer'], sim_datasplit, main_kwargs['ablate'])

		os.makedirs(output_folder_name, exist_ok=True)

		model, results = main(learning_rate, n_hidden, sim, sim_datasplit, **main_kwargs)

		print('SAVING FILES')
		savefiles(output_folder_name, sim, main_kwargs['which_task'], model, results)