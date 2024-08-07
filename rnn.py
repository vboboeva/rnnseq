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

from functions import * 
from train import train
from model import RNN, LinearWeightDropout
# from quick_plot import plot_weights
# from quick_plot import plot_loss
# from quick_plot import plot_accuracy_ablation


###########################################
################## M A I N ################
###########################################

def main(
	learning_rate, n_hidden, sim, sim_datasplit,
	# network parameters
	n_layers=1,
	# L = 4,
	m = 2,
	which_task=None,
	which_objective='CE',
	model_filename=None, # string with parameters filename
	from_file = [], # layers to set from file
	to_freeze = [], # layers not to be updated 
	which_init=None, # 'const', 'lazy', 'rich' 
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
	ablate=True,
	delay=0,
	cue_size=1
):
	print('TASK', which_task)
	print('DATASPLIT NO', sim_datasplit)
	print('SIMULATION NO', sim)
	print('L=', L)

	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	X, y, all_tokens, all_labels, num_tokens_onetype = load_tokens(alpha, L, m, n_types, letter_to_index)

	X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, n_train, n_test, n_other, num_types = make_tokens(all_tokens, all_labels, sim_datasplit, num_tokens_onetype, L, alpha, frac_train, X, y)

	all_configurations = generate_configurations(L, np.array(alphabet))
	tokens_other = remove_subset(all_configurations, all_tokens)
	labels_other = -1*np.ones(len(tokens_other))

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

	if drop_connect != 0.:
		layer_type = partial(LinearWeightDropout, drop_p=drop_connect)
	else:
		layer_type = nn.Linear

	# Create the model
	model = RNN(alpha, n_hidden, n_layers, output_size, 
		nonlinearity=which_transfer, device=device, 
		model_filename=model_filename, from_file=from_file,
		to_freeze=to_freeze, which_init=which_init, layer_type=layer_type)
	
	# Set up the optimizer
	optimizer = optim.Adam(
			model.parameters(),
			# filter(lambda p: p.requires_grad, model.parameters()), # may not be necessary
			lr=learning_rate, weight_decay=0,
		)

	# Set up the results dictionary
	results, token_to_type, token_to_set = make_results_dict(which_task, tokens_train, tokens_test, tokens_other, labels_train, labels_test, labels_other, ablate, epochs_snapshot)


	print('TRAINING NETWORK')

	if which_task in ['Pred', 'Class']:

		for epoch in range(n_epochs + 1):
			# print("X_train.shape (before) ", X_train.shape)
			if epoch in epochs_snapshot:
				# COPY THE WEIGHTS WHEN YOU SAVE THEM
				results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, alphabet, ablate, delay, epoch, cue_size)
					
			train(X_train, y_train, model, optimizer, which_objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, which_task=which_task, weight_decay=weight_decay, delay=delay)

			# After training: looking for motifs
			# Hypo1()	
	else:
		print("Task not recognized!")
		return

	# Quick and dirty plot of l:qoss (comment when running on cluster, for local use)
	# plot_weights(results, 5)
	# plot_loss(n_types, n_hidden, ablate, results)
	# plot_accuracy_ablation(n_hidden, alphabet, L, mydict)

	return model, results, token_to_type, token_to_set

##################################################

if __name__ == "__main__":

	# params = loadtxt('params_L4_m2.txt')
	params = loadtxt("params_test.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1,
		# L = 4,
		m = 2,
		which_task = 'Pred',  # Specify task
		which_objective = 'CE',
		model_filename = 'model_state_datasplit3956437760_sim603726602.pth',
		from_file = [], #, ['i2h', ['h2h']] 
		to_freeze = [], #, ['i2h','h2h'] 
		which_init = None,
		which_transfer = 'relu',
		n_epochs = 500,
		batch_size = 1, #16, # GD if = size(training set), SGD if = 1
		frac_train = 0.7,
		n_repeats = 1,
		n_types = 2, # set minimum 2 for class task to make sense
		alpha = 5,
		snap_freq = 1, # snapshot of net activity every snap_freq epochs
		drop_connect = 0.,
		# weight_decay = 0.2, # weight of L1 regularisation
		ablate = False,
		delay=0,
		cue_size=1 # only used for prediction task: number of letters to cue net with
	)

	# parameters
	alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]
	if main_kwargs['from_file'] == []:
		transfer=False
	else:
		transfer=True

	L_col_index = 0
	lr_col_index = 1
	n_hidden_col_index = 2
	sim_datasplit_col_index = 3
	sim_col_index = 4
	index = int(sys.argv[1]) - 1

	# size is the number of serial simulations running on a single node of the cluster, set this accordingly with the number of arrays in order to cover all parameters in the parameters.txt file

	size = 5
	for i in range(size):
		row_index = index * size + i

		L = int(params[row_index, L_col_index])
		learning_rate = params[row_index, lr_col_index]
		n_hidden = int(params[row_index, n_hidden_col_index])
		sim_datasplit = int(params[row_index, sim_datasplit_col_index])
		sim = int(params[row_index, sim_col_index])

		output_folder_name = 'Task%s_N%d_L%d_m%d_nepochs%d_lr%.5f_bs%d_ntypes%d_obj%s_init%s_transfer%s_datasplit%s_delay%d_ablate%s_cuesize%d_transferlearn%s' % (
		main_kwargs['which_task'], n_hidden, L, main_kwargs['m'], main_kwargs['n_epochs'], learning_rate, main_kwargs['batch_size'], main_kwargs['n_types'], main_kwargs['which_objective'], main_kwargs['which_init'],  main_kwargs['which_transfer'], sim_datasplit, main_kwargs['delay'], main_kwargs['ablate'], main_kwargs['cue_size'], transfer)

		os.makedirs(output_folder_name, exist_ok=True)

		model, results, token_to_type, token_to_set = main(learning_rate, n_hidden, sim, sim_datasplit, **main_kwargs)

		print('SAVING FILES')
		savefiles(output_folder_name, sim, main_kwargs['which_task'], model, results, token_to_type, token_to_set)