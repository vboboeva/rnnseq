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
import json

from train import tokenwise_test

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

def make_tokens(all_tokens, all_labels, sim_datasplit, num_tokens_onetype, L, alpha, frac_train, X, y):
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

	labels_train = all_labels[train_ids]
	labels_test = all_labels[test_ids]

	return X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, n_train, n_test, n_other, num_types

def make_results_dict(which_task, tokens_train, tokens_test, tokens_other, labels_train, labels_test, labels_other, ablate, epochs_snapshot):

	# Set up the dictionary that will contain results for each token
	results = {}
	token_to_type = {}
	token_to_set = {}

	if which_task == 'RNNClass':
		for measure in ['Loss', 'Retrieval', 'yh']:
			results.update({measure:{}}) 

			for set_, tokens, labels in (zip(['train','test'], [tokens_train, tokens_test], [labels_train, labels_test])):

				tokens = [''.join(token) for token in tokens]
			
				for token, label in (zip(tokens, labels)):
					token_to_set.update({token:set_}) 
					token_to_type.update({token:label})

					results[measure].update({token:{}})

					for epoch in epochs_snapshot:
						results[measure][token].update({epoch:{}})
						results[measure][token][epoch].update({0:[]})
				
						if ablate == True: 		
							for unit_ablated in range(1, n_hidden+1):
								results[measure][token][epoch].update({unit_ablated:[]})

	if which_task == 'RNNPred':
		for measure in ['Retrieval', 'yh']:
			results.update({measure:{}}) 

			for set_, tokens, labels in (zip(['train','test','other'], [tokens_train, tokens_test, tokens_other], [labels_train, labels_test, labels_other])):

				tokens = [''.join(token) for token in tokens]
			
				for token, label in (zip(tokens, labels)):
					token_to_set.update({token:set_}) 
					token_to_type.update({token:label})

			for epoch in epochs_snapshot:
				results[measure].update({epoch:{}})
				results[measure][epoch].update({0:[]})
				
				if ablate == True: 		
					for unit_ablated in range(1, n_hidden+1):
						results[measure][epoch].update({unit_ablated:[]})


	if which_task == 'RNNAuto':
		for measure in ['Loss', 'Retrieval', 'yh']:
			results.update({measure:{}}) 

			for set_, tokens, labels in (zip(['train','test'], [tokens_train, tokens_test], [labels_train, labels_test])):

				tokens = [''.join(token) for token in tokens]
			
				for token, label in (zip(tokens, labels)):
					token_to_set.update({token:set_}) 
					token_to_type.update({token:label})

					results[measure].update({token:{}})

					for epoch in epochs_snapshot:
						results[measure][token].update({epoch:{}})
						results[measure][token][epoch].update({0:[]})
				
						if ablate == True: 		
							for unit_ablated in range(1, n_hidden+1):
								results[measure][token][epoch].update({unit_ablated:[]})

	results['Whh']=[]
	return results, token_to_type, token_to_set

def test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, alphabet, ablate, delay, epoch, cue_size):
	# print('Testing')
	for (whichset, X, y, tokens, labels) in zip(['train', 'test'], [X_train, X_test], [y_train, y_test], [tokens_train, tokens_test], [labels_train, labels_test]):

		X = X.permute((1,0,2))
		# print("X after ", X.shape)

		if ablate == False:
			range_ablate=1
		else:
			range_ablate=n_hidden+1

		# ablate units one by one
		for idx_ablate in range(range_ablate):

			for (_X, _y, token, label) in zip(X, y, tokens, labels):
				token = ''.join(token)

				# For the classification task, Z is the output class
				# For the prediction task, Z is what has been predicted
				# For reconstruction task, Z is the reconstructed sequence

				Z, loss, yh = tokenwise_test(_X, _y, token, label, whichset, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task, idx_ablate=idx_ablate, n_hidden=n_hidden, delay=delay, cue_size=cue_size)

				if which_task == 'RNNClass':
					results['Loss'][token][epoch][idx_ablate]=loss
					results['Retrieval'][token][epoch][idx_ablate]=Z # how token was classified
					results['yh'][token][epoch][idx_ablate]=yh	 # hidden layer activity throughout sequence: L by n_hidden
					results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNPred':
					results['Retrieval'][epoch][idx_ablate].append(Z) # which token retrieved (Z)
					results['yh'][epoch][idx_ablate].append(yh)  # collect statistics of hidden layer activity in sequence that gave rise to retrieval of token Z: (num_Z, L, N)
					results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNAuto':
					results['Loss'][token][epoch][idx_ablate]=loss
					results['Retrieval'][token][epoch][idx_ablate]=Z # how input token was reconstructed
					results['yh'][token][epoch][idx_ablate]=yh	 # latent layer activity throughout sequence: n_latent

def savefiles(output_folder_name, sim, which_task, model, results, token_to_type, token_to_set):
	
	# Save the model state
	model.save('%s/model_state_sim%d.pth' % (output_folder_name, sim))

	with open('%s/results_sim%d.pkl'% (output_folder_name, sim), 'wb') as handle:
	    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('%s/token_to_set.pkl'% (output_folder_name), 'wb') as handle:
	    pickle.dump(token_to_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('%s/token_to_type.pkl'% (output_folder_name), 'wb') as handle:
	    pickle.dump(token_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)

	# # Convert and write JSON object to file
	# with open('%s/results_sim%d.json'% (output_folder_name, sim), 'wb') as handle:
	# 	_dumps = json.dumps(results)
	# 	json.dump(_dumps, handle)

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

