import numpy as np
import torch
import torch.nn.functional as F
import string
import random
import itertools
from itertools import product
from train import tokenwise_test
# from find_flat_distribution_subset import *
import numpy as np
from collections import Counter, defaultdict
# from pulp import LpProblem, LpVariable, lpSum, LpBinary, LpStatus

# def check_feasibility(sequences, target_size):
#     """
#     Check if it's possible to achieve a perfectly flat distribution.
    
#     Parameters:
#         sequences (list of str): The list of input sequences.
#         target_size (int): The desired size of the subset.
    
#     Returns:
#         bool: True if feasible, False otherwise.
#     """
#     num_positions = len(sequences[0])  # Length of each sequence
#     alphabet = set(char for seq in sequences for char in seq)  # Unique letters
#     target_frequency = target_size // len(alphabet)  # Target frequency per letter per position

#     # Count letter occurrences at each position
#     position_counts = {i: Counter(seq[i] for seq in sequences) for i in range(num_positions)}

#     # Check if each letter has enough occurrences to meet the target frequency
#     for i in range(num_positions):
#         for char in alphabet:
#             if position_counts[i][char] < target_frequency:
#                 return False
#     return True


# def find_flat_distribution_subset_ip(sequences, target_size):
#     """
#     Finds a subset of sequences with a perfectly flat distribution using integer programming.
    
#     Parameters:
#         sequences (list of str): The list of input sequences.
#         target_size (int): The desired size of the subset.
    
#     Returns:
#         list of str: A subset of sequences with a perfectly flat letter distribution.
#     """
#     num_positions = len(sequences[0])
#     alphabet = set(char for seq in sequences for char in seq)
#     target_frequency = target_size // len(alphabet)
    
#     # Create decision variables
#     seq_vars = [LpVariable(f"seq_{i}", cat=LpBinary) for i in range(len(sequences))]
    
#     # Create the problem
#     problem = LpProblem("PerfectFlatSubset", sense=1)
    
#     # Add constraints for each position and letter
#     for i in range(num_positions):
#         for char in alphabet:
#             # Sum of occurrences of 'char' at position 'i' in selected sequences
#             problem += (
#                 lpSum(seq_vars[j] for j, seq in enumerate(sequences) if seq[i] == char) == target_frequency,
#                 f"Flat_{i}_{char}",
#             )
    
#     # Constraint to enforce the target size
#     problem += lpSum(seq_vars) == target_size, "TargetSize"
    
#     # Solve the problem
#     problem.solve()
    
#     # Check if a solution was found
#     if LpStatus[problem.status] != "Optimal":
#         raise ValueError("Cannot find a subset with a perfectly flat distribution.")
    
#     # Extract selected sequences
#     selected_indices = [i for i, var in enumerate(seq_vars) if var.value() == 1]
#     selected_sequences = np.array([sequences[i] for i in selected_indices])
#     # print(selected_sequences)
#     return selected_indices

# # Example usage
# sequences = ["ABCD", "BCDA", "ACBD", "DBCA", "CDBA", "DABC", "BACD", "CABD"]
# target_size = 4
# subset = find_perfect_flat_subset_ip(sequences, target_size)
# print("Selected subset:", subset)

def replace_symbols (sequence, symbols):
	newseq=np.array(list(sequence))

	n_sym_seq = len(np.unique(newseq))
	n_sym_repl = len(np.array(list(symbols)))

	assert n_sym_seq == n_sym_repl, \
		f"Trying to replace {n_sym_seq} symbols in sequence "+ \
		f"with {n_sym_repl} symbols"

	_, id_list = np.unique(newseq, return_index=True)

	symbols_list = [newseq[idx] for idx in sorted(id_list)]
	# print('symbols_list =', symbols_list)

	pos_symbols = [np.where(newseq == sym)[0] for sym in symbols_list]
	# print('pos_symbols =', pos_symbols)

	for i, pos in enumerate(pos_symbols):
		# print(i, symbols[i], pos)
		newseq[pos] = symbols[i]

	return "".join(list(newseq))

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

def generate_random_strings(n, length, sim_datasplit):
	random.seed(sim_datasplit)
	strings = []
	for _ in range(n):
		# Generate a string by randomly choosing 'a' or 'b' for each character
		s = ''.join(random.choice(['a', 'b']) for _ in range(length))
		strings.append(s)
	return strings

def letter_to_onehot(all_tokens, all_labels, letter_to_index, cue_size, L, alpha, n_types):
	# turn letters into one hot vectors
	print(np.shape(all_tokens))
	X = torch.zeros((cue_size+L, len(all_tokens), alpha), dtype=torch.float32)
	y = torch.zeros((len(all_labels), n_types), dtype=torch.float32)

	for i, (token, label) in enumerate(zip(all_tokens, all_labels)):
		pos = [letter_to_index[letter] for letter in token]
		X[:,i,:] = F.one_hot(torch.tensor(pos, dtype=int), alpha)
		y[i,:] = F.one_hot(torch.tensor([label]), n_types)
	return X, y

def make_tokens(types, data_balance, alpha, cue_size, L, m, n_types, frac_train, letter_to_index):
	# load types
	
	alphabet = [string.ascii_lowercase[i] for i in range(alpha)]
	all_tokens=[]
	all_labels=[]

	# all permutations of m letters in the alphabet
	list_permutations=list(itertools.permutations(alphabet, m))

	for t, type_ in enumerate(types):
		tokens=[]
		# loop over all permutations 
		for perm in list_permutations:
			newseq=replace_symbols(type_, perm)
			tokens.append(newseq)

		tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
		all_tokens.append(tokens_arr)
		all_labels.append(np.array(len(tokens_arr)*[t]))

	all_tokens = np.vstack(all_tokens)
	all_labels = np.hstack(all_labels)

	num_tokens_onetype = np.shape(tokens)[0]

	X, y = letter_to_onehot(all_tokens, all_labels, letter_to_index, cue_size, L, alpha, n_types)

	# make train and test data
	n_train = int(frac_train * len(all_tokens))
	n_test = len(all_tokens) - n_train

	print('number of train', n_train)
	print('number of test', n_test)

	ids = torch.arange(len(all_tokens)).reshape(-1, num_tokens_onetype)
	for i, ids_type in enumerate(ids):
		ids[i] = torch.take(ids_type, torch.randperm(len(ids_type)))

	num_classes = int(len(all_tokens) / num_tokens_onetype)
	n_train_type = n_train // num_classes

	if data_balance == 'class':
		train_ids = ids[:, :n_train_type].reshape(-1)
		test_ids = ids[:, n_train_type:].reshape(-1)

	# search a set of sequences out of all_tokens such that a criterion is reached
	elif data_balance == 'whatwhere':

		print("Feasibility:", check_feasibility(all_tokens, n_test))

		test_ids = find_flat_distribution_subset_ip(all_tokens, target_size=n_test)
		train_ids = np.setdiff1d(np.arange(len(all_tokens)), test_ids)
	
	X_train = X[:, train_ids, :]
	X_test = X[:, test_ids, :]
	y_train = y[train_ids, :]
	y_test = y[test_ids, :]

	tokens_train = all_tokens[train_ids, :]
	tokens_test = all_tokens[test_ids, :]

	labels_train = all_labels[train_ids]
	labels_test = all_labels[test_ids]

	return X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, num_classes

def generate_configurations(L, alphabet):
	configurations = list(product(alphabet, repeat=L))
	configurations = np.vstack([np.array(list(config)) for config in configurations])    
	return configurations

def remove_subset(configurations, subset):
	subset_as_arrays = [np.array(item) for item in subset]
	filtered = [config for config in configurations if not any(np.array_equal(config, sub) for sub in subset_as_arrays)]
	return np.array(filtered)

def make_results_dict(which_task, tokens_train, tokens_test, labels_train, labels_test, ablate, epochs_snapshot):

	def make_class_auto():
		results = {}
		token_to_type = {}
		token_to_set = {}

		for measure in ['Loss', 'Retrieval']:#, 'yh', 'latent']:
			results.update({measure:{}}) 

			for set_, tokens, labels in (zip(['train', 'test'], [tokens_train, tokens_test], [labels_train, labels_test])):

				tokens = [''.join(token) for token in tokens]
			
				for token, label in (zip(tokens, labels)):
					token_to_set.update({token:set_}) 
					token_to_type.update({token:label})

					results[measure].update({token:{}})

					for epoch in epochs_snapshot:
						results[measure][token].update({epoch:{}})
						results[measure][token][epoch].update({0:[]})
				
						if ablate == True: 		
							for unit_ablated in range(1, n_hidden + 1):
								results[measure][token][epoch].update({unit_ablated:[]})
			
		return results, token_to_type, token_to_set
	
	def make_pred():
		results = {}
		token_to_type = {}
		token_to_set = {}
		for measure in ['Loss', 'Retrieval']: #, 'yh']:
			results.update({measure:{}}) 

			for set_, tokens, labels in (zip(['train','test'], [tokens_train, tokens_test], [labels_train, labels_test])):

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
		
		return results, token_to_type, token_to_set

	if which_task == 'RNNClass' or which_task == 'RNNAuto':
		# Set up the dictionary that will contain results for each token
		results, token_to_type, token_to_set = make_class_auto()
	
	if which_task == 'RNNPred':
		# Set up the dictionary that will contain results
		results, token_to_type, token_to_set = make_pred()

	results['Whh'] = []
	return results, token_to_type, token_to_set

def test(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, alphabet, ablate, delay, epoch, cue_size):

	for (whichset, X, y, tokens, labels) in zip(['train', 'test'], [X_train, X_test], [y_train, y_test], [tokens_train, tokens_test], [labels_train, labels_test]):

		X = X.permute((1,0,2))

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
					results['Loss'][token][epoch][idx_ablate] = loss
					results['Retrieval'][token][epoch][idx_ablate] = Z # how token was classified
					# results['yh'][token][epoch][idx_ablate] = yh.detach().cpu().numpy()	 # hidden layer activity throughout sequence: L by n_hidden
					# results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNPred':
					results['Loss'][epoch][idx_ablate].append(loss) # loss for token retrieved
					results['Retrieval'][epoch][idx_ablate].append(Z) # which token retrieved (Z)
					# results['yh'][epoch][idx_ablate].append(yh.detach().cpu().numpy())  # collect statistics of hidden layer activity in sequence that gave rise to retrieval of token Z: (num_Z, L, N)
					# results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNAuto':
					results['Loss'][token][epoch][idx_ablate]=loss
					results['Retrieval'][token][epoch][idx_ablate]=Z # how input token was reconstructed
					# results['yh'][token][epoch][idx_ablate]=yh[0].detach().cpu().numpy() # latent layer activity throughout sequence: n_latent
					# results['latent'][token][epoch][idx_ablate]=yh[1].detach().cpu().numpy() # latent layer activity throughout sequence: n_latent

def print_retrieval_color(test_task, losses, predicted_tokens, tokens_train, tokens_test):

	tokens_train = [''.join(p) for p in tokens_train]
	tokens_test = [''.join(p) for p in tokens_test]
	tokens_all = np.append(tokens_train, tokens_test)

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

	retrieved_train = len([s for s in predicted_tokens if s in tokens_train])/len(predicted_tokens)
	retrieved_test = len([s for s in predicted_tokens if s in tokens_test])/len(predicted_tokens)
	retrieved_other = len([s for s in predicted_tokens if s not in tokens_all])/len(predicted_tokens)

	meanval_train=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_train])
	meanval_test=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_test])
	meanval_other=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] not in tokens_all])

	print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f} Loss NonPatt {meanval_other:.2f}', end='   ')

