import sys
import os
import numpy as np
from numpy import loadtxt
import torch.nn as nn
import torch.optim as optim
import string
from functools import partial
import itertools
import pickle
from functions import * 
from train import train
from model import RNN, RNNAutoencoder, RNNMulti, LinearWeightDropout

###########################################
################## M A I N ################
###########################################

def main(
	L, n_types, n_hidden, sim, sim_datasplit,
	# network parameters
	n_layers=1,
	n_latent=7,
	m = 2,
	task=None,
	objective='CE',
	# model_filename=None, 
	from_file = [], 
	to_freeze = [], 
	init_weights=None, 
	learning_rate = 0.001,
	transfer_func='relu',
	n_epochs=10,
	batch_size=7,
	frac_train=0.7,  
	n_repeats=1,  
	alpha=5,
	snap_freq=2,
	drop_connect = 0.,
	weight_decay = 0.,
	ablate=True,
	delay=0,
	cue_size=1,
	data_balance='class',
	teacher_forcing_ratio=0.5,  # Add teacher forcing ratio parameter
	train_test_letters = 'Overlapping',
	letter_permutations_class = 'Random',	
	noise_level=0.0
):
	print('TASK', task)
	print('DATASPLIT NO', sim_datasplit)
	print('SIMULATION NO', sim)
	print('L=', L)

	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	types = np.array(loadtxt('input/types_L%d_m%d.txt'%(cue_size, m), dtype='str')).reshape(-1)

	# load all the tokens corresponding to that type
	if len(types) < n_types:
		raise ValueError('Not enough types! Please adjust cue_size.')

	np.random.seed(sim_datasplit)
	types_suffix = generate_random_strings(m, len(types), L, sim_datasplit)
	combined = [s1 + s2 for s1, s2 in zip(types, types_suffix)]
	types = combined

	if n_types > 0:
			# Get all ntype-element combinations
			type_combinations = list(itertools.combinations(types, n_types))

	# if the number of possible combinations is too large, we just consider the first 100
	if len(type_combinations) > 10:
		np.random.shuffle(type_combinations)
		type_combinations = type_combinations[:10]
	else:
		pass

	with open('%s/classes.pkl'% (output_folder_name), 'wb') as handle:
		pickle.dump(list(type_combinations), handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	print(f'number of {n_types}-tuple combinations)', len(type_combinations))

	for t, types_chosen in enumerate(list(type_combinations)):
		num_classes = len(types_chosen)		
		if from_file != []:
			model_filename = '%s/model_state_sim%d_classcomb%d.pth'%(input_folder_name, sim, t) # choose btw None or file of this format ('model_state_datasplit0_sim0.pth') if initializing state of model from file
		else:
			model_filename=None

		print('types_chosen', types_chosen)

		X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test = make_tokens(sim_datasplit, types_chosen, alpha, cue_size, L, m, frac_train, letter_to_index, train_test_letters, letter_permutations_class, noise_level)

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
			model = RNNAutoencoder(alpha, n_hidden, n_layers, n_latent, L+cue_size, device=device)
		
		elif task == 'RNNMulti':
			model = RNNMulti(alpha, n_hidden, n_layers, n_latent, num_classes, L+cue_size, device=device, model_filename=model_filename, from_file=from_file, to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)

		else:
			raise ValueError(f"Model not recognized: {task}")

		# Set up the optimizer
		optimizer = optim.Adam(
				model.parameters(),
				lr = learning_rate, weight_decay=0.) # Putting weight_decay nonzero here will apply it to all the weights in the model, not what we want

		if task == 'RNNMulti':
			test_tasks = ['RNNClass', 'RNNPred', 'RNNAuto']
			results_list = []
			for test_task in test_tasks:
				results, token_to_type, token_to_set = make_results_dict(test_task, tokens_train, tokens_test, labels_train, labels_test, ablate, epochs_snapshot)
				results_list.append(results)
		else:
			test_tasks = [task]
			results, token_to_type, token_to_set = make_results_dict(test_tasks[0], tokens_train, tokens_test, labels_train, labels_test, ablate, epochs_snapshot)
			results_list = [results]

		print('TRAINING NETWORK')
		for epoch in range(n_epochs + 1):
			
			if epoch in epochs_snapshot:
				print('epoch', epoch, test_tasks)

				for test_task, results in zip(test_tasks, results_list):
					test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, letter_to_index, index_to_letter, test_task, objective, n_hidden, L, alphabet, ablate, delay, epoch, cue_size)

					if test_task == 'RNNClass' or test_task == 'RNNAuto':
						meanval_train = np.mean([results['Loss'][k][epoch][0] for k in results['Loss'].keys() if token_to_set[k] == 'train'])
						meanval_test=np.mean([results['Loss'][k][epoch][0] for k in results['Loss'].keys() if token_to_set[k] == 'test'])
						print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f}', end='   ')
			
					elif test_task == 'RNNPred':
						losses = results['Loss'][epoch][0]
						predicted_tokens = results['Retrieval'][epoch][0]

						print_retrieval_color(test_task, losses, predicted_tokens, tokens_train, tokens_test)
					else:
						raise ValueError(f"Invalid task: {test_task}")
					print('\n')

			train(X_train, y_train, model, optimizer, objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter,  task=task, weight_decay=weight_decay, delay=delay)
	
		print('SAVING RESULTS')
		# Save the model state
		torch.save(model.state_dict(), '%s/model_state_sim%d_classcomb%d.pth' % (output_folder_name, sim, t))

		for results, test_task in zip(results_list, test_tasks):
			with open('%s/results_task%s_sim%d_classcomb%d.pkl'% (output_folder_name, test_task, sim, t), 'wb') as handle:
				pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open('%s/token_to_set_classcomb%d.pkl'% (output_folder_name, t), 'wb') as handle:
			pickle.dump(token_to_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open('%s/token_to_type_classcomb%d.pkl'% (output_folder_name, t), 'wb') as handle:
			pickle.dump(token_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)

##################################################

if __name__ == "__main__":

	# params = loadtxt('params_L4_m2.txt')
	params = loadtxt("params_new.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1, # number of RNN layers
		n_latent = 10, # size of latent layer (autoencoder only!!)
		m = 2, # number of unique letters in each sequence
		task = 'RNNClass',  # choose btw 'RNNPred', 'RNNClass', RNNAuto', or 'RNNMulti' 
		objective = 'CE', # choose btw cross entr (CE) and mean sq error (MSE)
		# model_filename = 'model_state_sim0_classcomb0.pth', # choose btw None or file of this format ('model_state_datasplit0_sim0.pth') if initializing state of model from file
		from_file = ['i2h', 'h2h'], # choose one or more of ['i2h', 'h2h'], if setting state of layers from file
		to_freeze = ['i2h', 'h2h'], # choose one or more of ['i2h','h2h'], those  layers not to be updated   
		init_weights = None, # choose btw None, 'const', 'lazy', 'rich' , weight initialization
		learning_rate = 0.001,
		transfer_func = 'relu', # transfer function of RNN units only
		n_epochs = 30, # number of training epochs
		batch_size = 1, #16, # GD if = size(training set), SGD if = 1
		frac_train = 110./140., # fraction of dataset to train on
		n_repeats = 1, # number of repeats of each sequence for training
		alpha = 15, # size of alphabet
		snap_freq = 1, # snapshot of net activity every snap_freq epochs
		drop_connect = 0., # fraction of dropped connections (reg)
		# weight_decay = 0.2, # weight of L1 regularisation
		ablate = False, # whether to test net with ablated units
		delay = 0, # number of zero-padding steps at end of input
		cue_size = 4, # number of letters to cue net with (prediction task only!!)
		data_balance = 'class', # choose btw 'class' and 'whatwhere'
		teacher_forcing_ratio = 1.,  # Add teacher forcing ratio parameter
		train_test_letters = 'Overlapping', # choose btw 'Disjoint' and 'Overlapping' and 'SemiOverlapping'
		letter_permutations_class = 'Random', # choose btw 'Same' and 'Random'
		noise_level = 0.0 # probability of finding an error in a given train sequence
	)

	# parameters
	alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]
	if main_kwargs['from_file'] == []:
		transfer=False
	else:
		transfer=True

	L_col_index = 0
	n_types_col_index = 1
	n_hidden_col_index = 2
	sim_datasplit_col_index = 3
	sim_col_index = 4
	index = int(sys.argv[1]) - 1

	# size is the number of serial simulations running on a single node of the cluster, set this accordingly with the number of arrays in order to cover all parameters in the parameters.txt file

	size = 1
	for i in range(size):
		row_index = index * size + i

		L = 2 #int(params[row_index, L_col_index])
		n_types = 5 #int(params[row_index, n_types_col_index])
		n_hidden = int(params[row_index, n_hidden_col_index])
		sim_datasplit = int(params[row_index, sim_datasplit_col_index])
		sim = int(params[row_index, sim_col_index])

		output_folder_name = 'Task%s_N%d_nlatent%d_L%d_m%d_alpha%d_nepochs%d_ntypes%d_fractrain%.1f_obj%s_init%s_transfer%s_cuesize%d_delay%d_datasplit%s' % ( main_kwargs['task'], n_hidden, main_kwargs['n_latent'], L, main_kwargs['m'], main_kwargs['alpha'], main_kwargs['n_epochs'], n_types, main_kwargs['frac_train'], main_kwargs['objective'], main_kwargs['init_weights'],  main_kwargs['transfer_func'], main_kwargs['cue_size'], main_kwargs['delay'], sim_datasplit )

		input_folder_name = 'TaskRNNPred_N%d_nlatent%d_L%d_m%d_alpha%d_nepochs30_ntypes%d_fractrain%.1f_obj%s_init%s_transfer%s_cuesize%d_delay%d_datasplit%s' % (n_hidden, main_kwargs['n_latent'], L, main_kwargs['m'], main_kwargs['alpha'], n_types, main_kwargs['frac_train'], main_kwargs['objective'], main_kwargs['init_weights'],  main_kwargs['transfer_func'], main_kwargs['cue_size'], main_kwargs['delay'], sim_datasplit )

		os.makedirs(output_folder_name, exist_ok=True)

		main(L, n_types, n_hidden, sim, sim_datasplit, **main_kwargs)