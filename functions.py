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

def generate_random_strings(m, n, length, sim_datasplit):
	random.seed(sim_datasplit)
	strings = []
	for _ in range(n):
		# Generate a string by randomly choosing 'a' or 'b' for each character
		alphabet = [string.ascii_lowercase[i] for i in range(m)]
		s = ''.join(random.choice(alphabet) for _ in range(length))
		strings.append(s)
	return strings

def letter_to_seq(types, letters):

	the_tokens=[]
	the_labels=[]

	for t, type_ in enumerate(types):
		tokens=[]
		# loop over all permutations 
		for perm in letters:
			newseq = replace_symbols(type_, perm)
			tokens.append(newseq)
		tokens_arr = np.vstack([np.array(list(token_)) for token_ in tokens])
		the_tokens.append(tokens_arr)
		the_labels.append(np.array(len(tokens_arr)*[t]))

	the_tokens = np.vstack(the_tokens)
	the_labels = np.hstack(the_labels)

	return the_tokens, the_labels

def seq_to_vectors(tokens, labels, L, alpha, letter_to_index, cue_size, n_types, noise_level):
	if noise_level > 0.0:
		alphabet_new = np.unique(tokens)  # Return sorted letters as a string
		# print(alphabet_new)
		# Add noise to the input
		tokens_noisy = tokens.copy()
		for i, token in enumerate(tokens):
			for j, letter in enumerate(token):
				if random.random() < noise_level:  
					# Choose a new letter different from the current one
					new_letter = random.choice([l for l in alphabet_new if l != token[j]])
					tokens_noisy[i, j] = new_letter

		tokens = tokens_noisy
	# turn letters into one hot vectors
	X = torch.zeros((L + cue_size, len(tokens), alpha), dtype=torch.float32)
	y = torch.zeros((len(labels), n_types), dtype=torch.float32)
	for i, (token, label) in enumerate(zip(tokens, labels)):
		pos = [letter_to_index[letter] for letter in token]
		X[:,i,:] = F.one_hot(torch.tensor(pos, dtype=int), alpha)
		y[i,:] = F.one_hot(torch.tensor([label]), n_types)

	return X, y

def make_tokens(sim_datasplit, types, alpha, cue_size, L, m, frac_train, letter_to_index, train_test_letters, noise_level=0.0):
	# load types
	alphabet = [string.ascii_lowercase[i] for i in range(alpha)]
	torch.manual_seed(sim_datasplit)
	print('noise_level', noise_level)
	# split into training and testing
	if train_test_letters == 'Disjoint':
		alphabet_shuffled = alphabet.copy()
		random.shuffle(alphabet_shuffled)

		train_alphabet = alphabet_shuffled[:int(frac_train*len(alphabet))]
		test_alphabet = alphabet_shuffled[int(frac_train*len(alphabet)):]

		permutations_train=list(itertools.permutations(train_alphabet, m))
		permutations_test=list(itertools.permutations(test_alphabet, m))
		
		train_letters = [permutations_train[i:i+1] for i in range(0, len(permutations_train))]
		test_letters = [permutations_test[i:i+1] for i in range(0, len(permutations_test))]

		train_letters = [item for sublist in train_letters for item in sublist]
		test_letters = [item for sublist in test_letters for item in sublist]

	elif train_test_letters == 'SemiOverlapping':
		# all m-tuple permutations of letters in the alphabet
		list_permutations=list(itertools.permutations(alphabet, m))
		chunks = [list_permutations[i:i+1] for i in range(0, len(list_permutations))]
		random.shuffle(chunks)
		n = int(frac_train*len(chunks))
		train_letters = [item for sublist in chunks[:n] for item in sublist]
		test_letters = [item for sublist in chunks[n:] for item in sublist]

	elif train_test_letters == 'Overlapping':
		list_permutations=list(itertools.permutations(alphabet, m))
		chunks = [list_permutations[i:i + alpha-1] for i in range(0, len(list_permutations), alpha-1)]
		random.shuffle(chunks)
		n = int(len(chunks)) - 1

		train_letters = [item for sublist in chunks[:n] for item in sublist]
		test_letters = [item for sublist in chunks[n:] for item in sublist]
	else:
		raise ValueError('train_test_letters should be Disjoint, SemiOverlapping, or Overlapping')
	
	tokens_train, labels_train = letter_to_seq(types, train_letters)
		
	X_train, y_train = seq_to_vectors(tokens_train, labels_train, L, alpha, letter_to_index, cue_size, len(types), noise_level)
	
	tokens_test, labels_test = letter_to_seq(types, test_letters)

	X_test, y_test = seq_to_vectors(tokens_test, labels_test, L, alpha, letter_to_index, cue_size, len(types), noise_level=0.0)

	return X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test

def generate_configurations(L, alphabet):
	configurations = list(product(alphabet, repeat = L))
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

		for measure in ['Loss', 'Retrieval', 'yh', 'latent']:
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
		for measure in ['Loss', 'Retrieval', 'yh']:
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
					results['yh'][token][epoch][idx_ablate] = yh.detach().cpu().numpy()	 # hidden layer activity throughout sequence: L by n_hidden
					# results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNPred':
					results['Loss'][epoch][idx_ablate].append(loss) # loss for token retrieved
					results['Retrieval'][epoch][idx_ablate].append(Z) # which token retrieved (Z)
					results['yh'][epoch][idx_ablate].append(yh.detach().cpu().numpy())  # collect statistics of hidden layer activity in sequence that gave rise to retrieval of token Z: (num_Z, L, N)
					# results['Whh'].append(model.h2h.weight.detach().cpu().numpy().copy())

				elif which_task == 'RNNAuto':
					results['Loss'][token][epoch][idx_ablate]=loss
					results['Retrieval'][token][epoch][idx_ablate]=Z # how input token was reconstructed
					results['yh'][token][epoch][idx_ablate]=yh[0].detach().cpu().numpy() # latent layer activity throughout sequence: n_latent
					results['latent'][token][epoch][idx_ablate]=yh[1].detach().cpu().numpy() # latent layer activity throughout sequence: n_latent

def print_retrieval_color(test_task, losses, predicted_tokens, tokens_train, tokens_test):

	tokens_train = [''.join(p) for p in tokens_train]
	tokens_test = [''.join(p) for p in tokens_test]
	tokens_all = np.append(tokens_train, tokens_test)

	# # Define ANSI escape codes for colors
	# GREEN = '\033[92m'
	# BLUE = '\033[94m'
	# RED = '\033[91m'
	# RESET = '\033[0m'

	# # Print predicted tokens with colors
	# for token in predicted_tokens:
	# 	if token in tokens_train:
	# 		print(f"{GREEN}{token}{RESET}", end=' ')
	# 	elif token in tokens_test:
	# 		print(f"{BLUE}{token}{RESET}", end=' ')
	# 	else:
	# 		print(f"{RED}{token}{RESET}", end=' ')

	meanval_train=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_train])
	meanval_test=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_test])
	meanval_other=np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] not in tokens_all])

	print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f} Loss NonPatt {meanval_other:.2f}', end='   ')

