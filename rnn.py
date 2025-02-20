import sys
import os
import numpy as np
from numpy import loadtxt
import torch.nn as nn
import torch.optim as optim
import string
from functools import partial
from functions import * 
from train import train
from model import RNN, RNNAutoencoder, RNNMulti, LinearWeightDropout


###########################################
################## M A I N ################
###########################################

def main(
	learning_rate, n_hidden, sim, sim_datasplit,
	# network parameters
	n_layers=1,
	n_latent=7,
	# L = 4,
	m = 2,
	task=None,
	objective='CE',
	model_filename=None, 
	from_file = [], 
	to_freeze = [], 
	init_weights=None, 
	transfer_func='relu',
	n_epochs=10,
	batch_size=7,
	frac_train=0.7,  
	n_repeats=1,  
	n_types=-1,  
	alpha=5,
	snap_freq=2,
	drop_connect = 0.,
	weight_decay = 0.,
	ablate=True,
	delay=0,
	cue_size=1,
	data_balance='class',
	teacher_forcing_ratio=0.5  # Add teacher forcing ratio parameter
):
	print('TASK', task)
	print('DATASPLIT NO', sim_datasplit)
	print('SIMULATION NO', sim)
	print('L=', L)

	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	X, y, all_tokens, all_labels, num_tokens_onetype = load_tokens(alpha, L, m, n_types, letter_to_index)

	X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, num_classes = make_tokens(data_balance, all_tokens, all_labels, sim_datasplit, num_tokens_onetype, L, alpha, frac_train, X, y)

	all_configurations = generate_configurations(L, np.array(alphabet))
	tokens_other = remove_subset(all_configurations, all_tokens)
	labels_other = -1*np.ones(len(tokens_other))

	# Train and test network
	torch.manual_seed(sim)
	n_batches = len(tokens_train) // batch_size

	# n_epochs for which take a snapshot of neural activity
	epochs_snapshot = np.arange(0, int(n_epochs)+1, snap_freq)

	if drop_connect != 0.:
		layer_type = partial(LinearWeightDropout, drop_p=drop_connect)
	else:
		layer_type = nn.Linear

	# Create the model
	if task in ['RNNClass', 'RNNPred']:
		if task == 'RNNClass':
			output_size = num_classes
		else:
			output_size = alpha
		model = RNN(alpha, n_hidden, n_layers, output_size, 
		nonlinearity=transfer_func, device=device, 
		model_filename=model_filename, from_file=from_file,
		to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)
	
	elif task == 'RNNAuto':
		model = RNNAutoencoder(alpha, n_hidden, n_layers, n_latent, L, device=device)
	
	elif task == 'RNNMulti':
		model = RNNMulti(alpha, n_hidden, n_layers, n_latent, num_classes, L, device=device, model_filename=model_filename, from_file=from_file, to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)

	else:
		raise ValueError(f"Model not recognized: {task}")

	# Set up the optimizer
	optimizer = optim.Adam(
			model.parameters(),
			lr=learning_rate, weight_decay=0.) # Putting weight_decay nonzero here will apply it to all the weights in the model, not what we want

	if task != 'RNNMulti':
		test_tasks = [task]
		results, token_to_type, token_to_set = make_results_dict(test_tasks[0], tokens_train, tokens_test, tokens_other, labels_train, labels_test, labels_other, ablate, epochs_snapshot)
		results_list = [results]
	else:
		test_tasks = ['RNNClass', 'RNNPred', 'RNNAuto']
		results_list = []
		for test_task in test_tasks:
			results, token_to_type, token_to_set = make_results_dict(test_task, tokens_train, tokens_test, tokens_other, labels_train, labels_test, labels_other, ablate, epochs_snapshot)
			results_list.append(results)

	print('TRAINING NETWORK')

	for epoch in range(n_epochs + 1):
		if epoch in epochs_snapshot:
			for test_task, results in zip(test_tasks, results_list):
				test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, letter_to_index, index_to_letter, test_task, objective, n_hidden, L, alphabet, ablate, delay, epoch, cue_size)

		train(X_train, y_train, model, optimizer, objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter,  task=task, weight_decay=weight_decay, delay=delay, teacher_forcing_ratio=teacher_forcing_ratio)

		#  Decrease teacher forcing ratio
		# if teacher_forcing_ratio:
		# 	teacher_forcing_ratio = max(0.1, teacher_forcing_ratio * 0.99)

		# Print loss
		if epoch in epochs_snapshot:
			print(f'Epoch {epoch}', end='   ')
			for test_task, results in zip(test_tasks, results_list):
				if test_task == 'RNNClass' or test_task == 'RNNAuto':
					meanval_train=np.mean([results['Loss'][k][epoch][0] for k in results['Loss'].keys() if token_to_set[k] == 'train'])
					meanval_test=np.mean([results['Loss'][k][epoch][0] for k in results['Loss'].keys() if token_to_set[k] == 'test'])
					print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f}', end='   ')
		   
				elif test_task == 'RNNPred':
					losses = results['Loss'][epoch][0]
					predicted_tokens = results['Retrieval'][epoch][0]

					# Define ANSI escape codes for colors
					GREEN = '\033[92m'
					BLUE = '\033[94m'
					RED = '\033[91m'
					RESET = '\033[0m'

					# Print predicted tokens with colors
					for token in predicted_tokens:
						if token in tokens_train:
							print(f"{GREEN}{token}{RESET}", end=' ')
						elif token in tokens_test:
							print(f"{BLUE}{token}{RESET}", end=' ')
						else:
							print(f"{RED}{token}{RESET}", end=' ')
					print()

					tokens_train = [''.join(p) for p in tokens_train]
					tokens_test = [''.join(p) for p in tokens_test]
					tokens_other = [''.join(p) for p in tokens_other]
					retrieved_train = len([s for s in predicted_tokens if s in tokens_train])/len(predicted_tokens)
					retrieved_test = len([s for s in predicted_tokens if s in tokens_test])/len(predicted_tokens)
					retrieved_other = len([s for s in predicted_tokens if s in tokens_other])/len(predicted_tokens)
					meanval_train=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_train])
					meanval_test=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_test])
					meanval_other=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_other])
					# print(f'{test_task} Loss Tr {meanval:.2f} frac_train {retrieved_train:.2f} frac_test {retrieved_test:.2f} frac_other{retrieved_other:.4f}', end='   ')
					print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f} Loss NonPatt {meanval_other:.2f}', end='   ')
			print('\n')
						
	# Quick and dirty plot of loss (comment when running on cluster, for local use)
	# plot_weights(results, 5)
	# plot_loss(n_types, n_hidden, ablate, results)
	# plot_accuracy_ablation(n_hidden, alphabet, L, mydict)
	return model, results_list, test_tasks, token_to_type, token_to_set

##################################################

if __name__ == "__main__":

	# params = loadtxt('params_L4_m2.txt')
	params = loadtxt("params.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1, # number of RNN layers
		n_latent = 10, # size of latent layer (autoencoder only!!)
		# L = 4, # length of sequence
		m = 2, # number of unique letters in each sequence
		task = 'RNNPred',  # choose btw 'RNNPred', 'RNNClass', RNNAuto', or 'RNNMulti' 
		objective = 'CE', # choose btw cross entr (CE) and mean sq error (MSE)
		model_filename = None, # choose btw None or file of this format ('model_state_datasplit0_sim0.pth') if initializing state of model from file
		from_file = [], # choose one or more of ['i2h', 'h2h'], if setting state of layers from file
		to_freeze = [], # choose one or more of ['i2h','h2h'], those  layers not to be updated   
		init_weights = None, # choose btw None, 'const', 'lazy', 'rich' , weight initialization
		transfer_func = 'relu', # transfer function of RNN units only
		n_epochs = 300, # number of training epochs
		batch_size = 1, #16, # GD if = size(training set), SGD if = 1
		frac_train = 110./140., # fraction of dataset to train on
		n_repeats = 1, # number of repeats of each sequence for training
		n_types = 1, # # number of types to train net with: 1 takes just the first, -1 takes all types. Set minimum 2 for class task to make sense
		alpha = 10, # size of alphabet
		snap_freq = 5, # snapshot of net activity every snap_freq epochs
		drop_connect = 0., # fraction of dropped connections (reg)
		# weight_decay = 0.2, # weight of L1 regularisation
		ablate = False, # whether to test net with ablated units
		delay = 0, # number of zero-padding steps at end of input
		cue_size = 2, # number of letters to cue net with (prediction task only!!)
		data_balance = 'class', # choose btw 'class' and 'whatwhere'
		teacher_forcing_ratio = 1.  # Add teacher forcing ratio parameter
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

	size = 25
	for i in range(size):
		row_index = index * size + i

		L = int(params[row_index, L_col_index])
		learning_rate = params[row_index, lr_col_index]
		n_hidden = int(params[row_index, n_hidden_col_index])
		sim_datasplit = int(params[row_index, sim_datasplit_col_index])
		sim = int(params[row_index, sim_col_index])

		output_folder_name = 'Task%s_N%d_nlatent%d_L%d_m%d_alpha%d_nepochs%d_lr%.5f_bs%d_ntypes%d_fractrain%.1f_obj%s_init%s_transfer%s_cuesize%d_delay%d_datasplit%s' % ( main_kwargs['task'], n_hidden, main_kwargs['n_latent'], L, main_kwargs['m'], main_kwargs['alpha'], main_kwargs['n_epochs'], learning_rate, main_kwargs['batch_size'], main_kwargs['n_types'], main_kwargs['frac_train'], main_kwargs['objective'], main_kwargs['init_weights'],  main_kwargs['transfer_func'], main_kwargs['cue_size'], main_kwargs['delay'], sim_datasplit )

		os.makedirs(output_folder_name, exist_ok=True)

		model, results_list, test_tasks, token_to_type, token_to_set = main(learning_rate, n_hidden, sim, sim_datasplit, **main_kwargs)

		print('SAVING FILES')
		savefiles(output_folder_name, sim, model, results_list, test_tasks, token_to_type, token_to_set)