import sys
import os
from os.path import join
import numpy as np
from numpy import loadtxt
import torch.nn as nn
import torch.optim as optim
import string
from functools import partial
import itertools
import pickle
import json
import random
from functions import * 
from train import train
from train import test_save
from model import RNN, RNNAutoencoder, RNNMulti, LinearWeightDropout, LowRankLinear, \
				print_parameters, print_parameters_comp, state_dicts_equal
from pprint import pprint

from data_utils import augment_data

###########################################
################## M A I N ################
###########################################

def main(
	L, n_types, n_hidden, split_id, sim_id,
	# network parameters
	n_layers=1,
	n_latent=7,
	m = 2,
	task=None,
	k_steps=1,		# only used for task='RNNPred'
	k_steps_scheduling=False,
	objective='CE',
	from_file = [], 
	to_freeze = [],
	max_rank=None,
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
	ablate=False,
	delay=0,
	cue_size=1,
	data_balance='class',
	teacher_forcing_ratio=0.5,  # Add teacher forcing ratio parameter
	train_test_letters = 'Overlapping',
	letter_permutations_class = 'Random',	
	noise_level=0.0,
	input_folder_name = '',
	output_folder_name = '',
):
	if k_steps_scheduling:
		k_steps_max = k_steps
		k_steps_schedule = lambda epoch: 1 + int(epoch / n_epochs * k_steps_max)

	print('TASK', task)
	print('DATASPLIT NO', split_id)
	print('L=', L)
		
	letter_to_index, index_to_letter = make_dicts(alpha)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	types = np.array(loadtxt(f"input/types_L{cue_size}_m{m}.txt", dtype="str")).reshape(-1)

	# load all the tokens corresponding to that type
	if len(types) < n_types:
		raise ValueError('Not enough types! Please adjust cue_size.')

	types_suffix = generate_random_strings(m, len(types), L)
	combined = [s1 + s2 for s1, s2 in zip(types, types_suffix)]
	types = combined

	if n_types > 0:
		# Get all ntype-element combinations
		type_combinations = list(itertools.combinations(types, n_types))

	# if the number of possible combinations is too large, we just consider the first 20
	if len(type_combinations) > 20:
		np.random.shuffle(type_combinations)
		type_combinations = type_combinations[:20]
	else:
		pass

	with open(f"{output_folder_name}/classes.pkl", "wb") as handle:
		pickle.dump(list(type_combinations), handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	print(f'number of {n_types}-tuple combinations:', len(type_combinations))

	for t, types_chosen in enumerate(list(type_combinations)):

		num_classes = len(types_chosen)
		if from_file != []:
			model_filename = f'{input_folder_name}/model_state_filtered_classcomb{t}_sim{sim_id}.pth' # choose btw None or file of this format ('model_state.pth') if initializing state of model from file
		else:
			model_filename=None

		print('types_chosen', types_chosen)

		X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test = make_tokens(types_chosen, alpha, m, frac_train, letter_to_index, train_test_letters, letter_permutations_class, noise_level)

		# # Data augmentation
		# num_augmented = 3

		# X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test = augment_dataset(X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, num_augmented=num_augmented)

		# Train and test network
		n_batches = len(tokens_train) // batch_size

		# n_epochs for which take a snapshot of neural activity
		epoch_snapshots = np.arange(0, int(n_epochs), snap_freq)

		if drop_connect != 0.:
			layer_type = partial(LinearWeightDropout, drop_p=drop_connect)
		else:
			layer_type = LowRankLinear

		# Create the model
		if task in ['RNNClass', 'RNNPred']:
			if task == 'RNNClass':
				output_size = num_classes
			else:
				output_size = alpha
			model = RNN(alpha, n_hidden, n_layers, output_size,
						nonlinearity=transfer_func, device=device,
						model_filename=model_filename, from_file=from_file, max_rank=max_rank,
						to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type, sim_id=sim_id)
		
		elif task == 'RNNAuto':
			model = RNNAutoencoder(alpha, n_hidden, n_layers, n_latent, L+cue_size,
						nonlinearity=transfer_func, device=device,
						model_filename=model_filename, from_file=from_file,
						to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)
		
		elif task == 'RNNMulti':
			model = RNNMulti(alpha, n_hidden, n_layers, n_latent, num_classes, L+cue_size,
						device=device, model_filename=model_filename, from_file=from_file,
						to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)

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
				results, token_to_type, token_to_set = make_results_dict(tokens_train, tokens_test, labels_train, labels_test, epoch_snapshots)
				results_list.append(results)
		else:
			test_tasks = [task]
			results, token_to_type, token_to_set = make_results_dict(tokens_train, tokens_test, labels_train, labels_test, epoch_snapshots)
			results_list = [results]

		print('TRAINING NETWORK')
		for epoch in range(n_epochs):

			if k_steps_scheduling:
				k_steps = k_steps_schedule(epoch)
			
			if epoch in epoch_snapshots:
				print('epoch', epoch, test_tasks)

				for test_task, results in zip(test_tasks, results_list):

					test_save(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, test_task, objective, n_hidden, L, alphabet, delay, cue_size, epoch=epoch)
					
					meanval_train = np.mean([results['Loss'][k][epoch] for k in results['Loss'].keys() if token_to_set[k] == 'train'])

					meanval_test=np.mean([results['Loss'][k][epoch] for k in results['Loss'].keys() if token_to_set[k] == 'test'])

					print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f}', end = '   ')
			
					print('\n')

			try:
				train(X_train, y_train, model, optimizer, objective, n_batches, batch_size,
					task=task, k_steps=k_steps, weight_decay=weight_decay, delay=delay)
			except Exception as e:
				print(f"{type(e).__name__}: {e}.\nSkipping training step.")

		print('SAVING RESULTS')
		# Save the model state
		if task == 'RNNClass' or task == 'RNNPred':
			torch.save(model.state_dict(), f"{output_folder_name}/model_state_classcomb{t}_sim{sim_id}.pth")
		else:
			# Get the full state dictionary
			full_state_dict = model.state_dict()

			# Extract and rename only encoder-related keys
			renamed_encoder_state_dict = {
				k.replace("encoder.rnn.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.rnn.")
			}
			# Save the modified encoder weights
			torch.save(renamed_encoder_state_dict, f"{output_folder_name}/model_state_classcomb{t}.pth")

		for results, test_task in zip(results_list, test_tasks):
			with open(f"{output_folder_name}/results_task{test_task}_classcomb{t}_sim{sim_id}.pkl", 'wb') as handle:
				pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f"{output_folder_name}/token_to_set_classcomb{t}.pkl", 'wb') as handle:
			pickle.dump(token_to_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

		with open(f"{output_folder_name}/token_to_type_classcomb{t}.pkl", 'wb') as handle:
			pickle.dump(token_to_type, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# ablating model at end of training
		if ablate:
			print('epoch', epoch, test_tasks, 'TEST AFTER ABLATION')
			 
			if task == 'RNNMulti':
				test_tasks = ['RNNClass', 'RNNPred', 'RNNAuto']
				results_ablate_list = []
				for test_task in test_tasks:
					results_ablate = make_results_ablate_dict(tokens_train, tokens_test, labels_train, labels_test, types_chosen)
					results_ablate_list.append(results_ablate)
			else:
				test_tasks = [task]
				results_ablate = make_results_ablate_dict(tokens_train, tokens_test, labels_train, labels_test, types_chosen)
				results_ablate_list = [results_ablate]
				# print('results_ablate')
				# pprint(results_ablate)

			for test_task, results_ablate in zip(test_tasks, results_ablate_list):
				# ablate cluster and test
				ablate_dict = compute_feature_variance(token_to_type, types_chosen, results, task, n_hidden, epoch_snapshots[-1])

				for ablateclass in types_chosen:
					print('ablating cluster', ablateclass)
					idx_ablate = ablate_dict[ablateclass]
					
					test_save(results_ablate, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, test_task, objective, n_hidden, L, alphabet, delay, cue_size, idx_ablate = idx_ablate, class_ablate = ablateclass)

					meanval_ablate = np.mean([results_ablate['Loss'][k][ablateclass] for k in results_ablate['Loss'].keys() if token_to_type[k] == list(types_chosen).index(ablateclass)] )

					meanval_notablate = np.mean([results_ablate['Loss'][k][ablateclass] for k in results_ablate['Loss'].keys() if token_to_type[k] != list(types_chosen).index(ablateclass)] )

					print(f'{meanval_ablate:.2f} {meanval_notablate:.2f}', end = '   ')

					print('\n')

					# for testclass in types_chosen:
					# 	print('testing cluster', testclass)

						# meanval_train = np.mean([results_ablate['Loss'][k][ablateclass] for k in results_ablate['Loss'].keys() if token_to_type[k] == list(types_chosen).index(testclass) and token_to_set[k] == 'train'])

						# meanval_test = np.mean([results_ablate['Loss'][k][ablateclass] for k in results_ablate['Loss'].keys() if token_to_type[k] == list(types_chosen).index(testclass) and token_to_set[k] == 'test'])

						# print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f}', end = '   ')

				# with open(f"{output_folder_name}/results_ablate_task{test_task}_classcomb{t}.pkl", 'wb') as handle:
				# 	pickle.dump(results_ablate, handle, protocol=pickle.HIGHEST_PROTOCOL)


##################################################

if __name__ == "__main__":

	# params = loadtxt('params_L4_m2.txt')
	params = loadtxt("parameters.txt")

	main_kwargs = dict(
		# network parameters
		n_layers = 1, # number of RNN layers
		n_latent = 10, # size of latent layer (autoencoder only!!)
		m = 2, # number of unique letters in each sequence
		task = 'RNNPred',  # choose btw 'RNNPred', 'RNNClass', RNNAuto', or 'RNNMulti'
		k_steps=None,	# number of steps for the k-steps rollout (prediction only)
		k_steps_scheduling=True,
		objective = 'CE', # choose btw cross entr (CE) and mean sq error (MSE)
		from_file = [], # choose one or more of ['i2h', 'h2h'], if setting state of layers from file
		to_freeze = [], # choose one or more of ['i2h','h2h'], those  layers not to be updated   
		max_rank=None,
		init_weights = None, # choose btw None, 'Const', 'Lazy', 'Rich' , weight initialization
		learning_rate = 0.001,
		transfer_func = 'relu', # transfer function of RNN units only
		n_epochs = 50, # number of training epochs
		batch_size = 1, #16, # GD if = size(training set), SGD if = 1
		frac_train = 0.8, # fraction of dataset to train on
		n_repeats = 1, # number of repeats of each sequence for training
		alpha = 10, # size of alphabet
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
		noise_level = 0.0, # probability of finding an error in a given train sequence
		input_folder_name = '', # folder to load model state from if from_file != []
		output_folder_name = '', # folder to save results to
	)

	for sim_idx, (from_file, to_freeze) in enumerate(zip(
			[	[], 
				['i2h'], 
				['h2h'], 
				['h2o'], 
				['i2h', 'h2h'], 
				['h2h', 'h2o'], 
				['i2h', 'h2h', 'h2o'], 

				['i2h'], 
				['h2h'], 
				['h2o'], 
				['i2h', 'h2h'],
				['h2h', 'h2o'], 
				['i2h', 'h2h', 'h2o'] ],

			[	[], 
				['i2h'], 
				['h2h'], 
				['h2o'], 
				['i2h', 'h2h'], 
				['h2h', 'h2o'], 
				['i2h', 'h2h', 'h2o'], 

				[], 
				[], 
				[], 
				[],
				[],  
				[] ]
		)):

		if sim_idx > 0:
			continue
		
	
		# parameters
		alphabet = [string.ascii_lowercase[i] for i in range(main_kwargs['alpha'])]

		L_col_index = 0
		n_types_col_index = 1
		n_hidden_col_index = 2
		split_id_col_index = 3
		sim_id_col_index = 4
		index = int(sys.argv[1]) - 1

		k_steps = int(sys.argv[2])
		if k_steps == 0:
			k_steps = None
		main_kwargs['k_steps'] = k_steps
		main_kwargs['k_steps_scheduling'] = bool(int(sys.argv[3]))

		for key, val in main_kwargs.items():
			print(f"{str(key):<20}:  {val}")

		# size is the number of serial simulations running on a single node of the cluster, set this accordingly with the number of arrays in order to cover all parameters in the parameters.txt file
		
		size = 100
		for i in range(size):
			row_index = index * size + i
		# for split_id in range(40):

			L = int(params[row_index, L_col_index])
			n_types = int(params[row_index, n_types_col_index])
			n_hidden = int(params[row_index, n_hidden_col_index])
			split_id = int(params[row_index, split_id_col_index])
			sim_id = int(params[row_index, sim_id_col_index])

			# Set seeds
			random.seed(1990+split_id)
			np.random.seed(1990+split_id)

			output_folder_name = (
				f"test_rollout/Task{main_kwargs['task']}_N{n_hidden}_nlatent{main_kwargs['n_latent']}_"
				f"kSteps{main_kwargs['k_steps']}"+("scheduled" if main_kwargs['k_steps_scheduling'] else "")+"_"
				f"L{L}_m{main_kwargs['m']}_alpha{main_kwargs['alpha']}_nepochs{main_kwargs['n_epochs']}_"
				f"ntypes{n_types}_fractrain{main_kwargs['frac_train']:.1f}_obj{main_kwargs['objective']}_"
				f"init{main_kwargs['init_weights']}_transfer{main_kwargs['transfer_func']}_"
				f"cuesize{main_kwargs['cue_size']}_delay{main_kwargs['delay']}_datasplit{split_id}_{sim_idx}"
				f"max_rank{main_kwargs['max_rank']}"
			)

			input_folder_name = (
				f"test_avg/Task{main_kwargs['task']}_N{n_hidden}_nlatent{main_kwargs['n_latent']}_"
				f"L{L}_m{main_kwargs['m']}_alpha{main_kwargs['alpha']}_nepochs26_"
				f"ntypes{n_types}_fractrain{main_kwargs['frac_train']:.1f}_obj{main_kwargs['objective']}_"
				f"init{main_kwargs['init_weights']}_transfer{main_kwargs['transfer_func']}_"
				f"cuesize{main_kwargs['cue_size']}_delay{main_kwargs['delay']}_datasplit{split_id}"
			)
			
			os.makedirs(output_folder_name, exist_ok=True)
			main_kwargs['input_folder_name'] = input_folder_name
			main_kwargs['output_folder_name'] = output_folder_name

			main(L, n_types, n_hidden, split_id, sim_id, **main_kwargs)


# for c in range(0, 10):
# 	print(c)
# 	filename = f'model_state_classcomb{c}.pth'
# 	full_state_dict = torch.load(f'{input_folder_name}/{filename}')
# 	print(full_state_dict.keys())
# 	# Extract and rename only encoder-related keys
# 	renamed_encoder_state_dict = {
# 		k.replace("encoder.rnn.", ""): v for k, v in full_state_dict.items() if k.startswith("encoder.rnn.")
# 	}
# 	print(renamed_encoder_state_dict.keys())
# 	# Save the modified encoder weights
# 	torch.save(renamed_encoder_state_dict, f"{input_folder_name}/{filename}")
