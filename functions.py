import numpy as np
import torch
import torch.nn.functional as F
import string
import random
import itertools
from train import tokenwise_test
import numpy as np
from collections import defaultdict
# from find_flat_distribution_subset import *

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

def replace_symbols(sequence, symbols):
	"""
	Replaces unique symbols in `sequence` with corresponding symbols from `symbols`.

	Args:
		sequence (str): The input sequence of characters.
		symbols (str): A string of replacement symbols.

	Returns:
		str: A new sequence where each unique character from `sequence` is replaced with the corresponding symbol.
	"""
	newseq = np.array(list(sequence))
	unique_symbols = np.unique(newseq)
	n_sym_seq = len(unique_symbols)
	n_sym_repl = len(symbols)

	assert n_sym_seq == n_sym_repl, \
		f"Trying to replace {n_sym_seq} symbols in sequence with {n_sym_repl} symbols"

	_, id_list = np.unique(newseq, return_index=True)
	symbols_list = [newseq[idx] for idx in sorted(id_list)]
	pos_symbols = [np.where(newseq == sym)[0] for sym in symbols_list]

	for i, pos in enumerate(pos_symbols):
		newseq[pos] = symbols[i]

	return "".join(newseq)

def make_repetitions(tokens_train, X_train, n_repeats, sim_datasplit):
	"""
	Efficiently repeats training tokens and tensors without using slow loops.

	Args:
		tokens_train (np.ndarray): Training tokens (num_samples, sequence_length).
		X_train (torch.Tensor): One-hot encoded training sequences.
		n_repeats (int): Maximum number of times to repeat each token.

	Returns:
		np.ndarray: Repeated tokens.
		torch.Tensor: Repeated tensor sequences.
	"""
	np.random.seed(sim_datasplit)

	repeat_counts = np.random.randint(1, n_repeats + 1, size=len(tokens_train))
	tokens_train_repeated = np.repeat(tokens_train, repeat_counts, axis=0)
	X_train_repeated = X_train.repeat_interleave(torch.tensor(repeat_counts), dim=1)

	return tokens_train_repeated, X_train_repeated

def make_dicts(alpha):
	"""
	Creates mappings between letters and indices.

	Args:
		alpha (int): Number of letters in the alphabet.

	Returns:
		dict: Mapping of letters to indices.
		dict: Mapping of indices to letters.
	"""
	keys = list(string.ascii_lowercase)[:alpha]
	values = np.arange(alpha)
	letter_to_index = dict(zip(keys, values))
	index_to_letter = dict(zip(values, keys))

	return letter_to_index, index_to_letter

def generate_random_strings(m, n, length, sim_datasplit):
	"""
	Generates `n` random strings of given `length` using an alphabet of size `m`.

	Args:
		m (int): Number of unique letters.
		n (int): Number of strings to generate.
		length (int): Length of each string.
		sim_datasplit (int): Random seed for reproducibility.

	Returns:
		list: List of randomly generated strings.
	"""
	np.random.seed(sim_datasplit)

	alphabet = np.array([string.ascii_lowercase[i] for i in range(m)])
	random_strings = np.random.choice(alphabet, size=(n, length))

	return [''.join(row) for row in random_strings]

def letter_to_seq(types, letters):
	"""
	Converts letter permutations into labeled token sequences.

	Args:
		types (list): List of type labels.
		letters (list): List of letter permutations.

	Returns:
		np.ndarray: Token sequences.
		np.ndarray: Corresponding labels.
	"""
	the_tokens = []
	the_labels = []

	for t, (type_, letters_) in enumerate(zip(types, letters)):
		tokens_arr = np.array([list(replace_symbols(type_, perm)) for perm in letters_])
		the_tokens.append(tokens_arr)
		the_labels.append(np.array(len(tokens_arr) * [t]))

	the_tokens = np.vstack(the_tokens)
	the_labels = np.hstack(the_labels)

	return the_tokens, the_labels

def seq_to_vectors(tokens, labels, alpha, letter_to_index, n_types, sim_datasplit, noise_level=0.0):
	"""
	Converts token sequences into one-hot encoded tensor representations efficiently.

	Args:
		tokens (np.ndarray): 2D array of shape (num_samples, sequence_length) containing letter tokens.
		labels (np.ndarray): 1D array of shape (num_samples,) containing integer labels.
		L (int): Sequence length.
		alpha (int): Alphabet size.
		letter_to_index (dict): Mapping from letter to index.
		cue_size (int): Cue size.
		n_types (int): Number of label categories.
		noise_level (float): Probability of a character being randomly replaced (default=0.0).

	Returns:
		torch.Tensor: One-hot encoded input tensor `X` of shape (sequence_length + cue_size, num_samples, alphabet_size).
		torch.Tensor: One-hot encoded label tensor `y` of shape (num_samples, n_types).
	"""
	np.random.seed(sim_datasplit)
	positions = np.vectorize(letter_to_index.get)(tokens)

	if noise_level > 0.0:
		alphabet_new = np.unique(tokens)
		mask = np.random.rand(*tokens.shape) < noise_level
		random_letters = np.random.choice(alphabet_new, size=tokens.shape)
		tokens[mask] = random_letters[mask]
		positions = np.vectorize(letter_to_index.get)(tokens)

	X = F.one_hot(torch.tensor(positions, dtype=torch.long), alpha).permute(1, 0, 2).float()
	y = F.one_hot(torch.tensor(labels, dtype=torch.long), n_types).float()

	return X, y

def make_tokens(sim_datasplit, types, alpha, cue_size, L, m, frac_train, letter_to_index, train_test_letters, letter_permutations_class, noise_level):
	"""
	Generates training and testing token sequences based on the specified split strategy.

	Args:
		sim_datasplit (int): Random seed for reproducibility.
		types (list): List of sequence types.
		alpha (int): Alphabet size.
		cue_size (int): Cue size.
		L (int): Sequence length.
		m (int): Length of permutations.
		frac_train (float): Fraction of data used for training.
		letter_to_index (dict): Mapping from letters to indices.
		train_test_letters (str): Strategy for splitting the alphabet ('Disjoint', 'SemiOverlapping', 'Overlapping').
		letter_permutations_class (str): Whether to shuffle permutations ('Random', 'Same').
		noise_level (float): Level of noise to introduce.

	Returns:
		tuple: Training and testing sets of `X`, `y`, tokens, and labels.
	"""
	print('letter_permutations_class', letter_permutations_class)
	alphabet = [string.ascii_lowercase[i] for i in range(alpha)]
	# Generate all m-length permutations from the full alphabet
	all_permutations = list(itertools.permutations(alphabet, m))
	
	np.random.seed(sim_datasplit) 

	# split into training and testing
	if train_test_letters == 'Disjoint':

	# Shuffle alphabet to ensure randomness in splitting
		np.random.shuffle(alphabet)

		# Split the alphabet into completely disjoint sets
		split_idx = int(len(alphabet) * frac_train)

		train_alpha = set(alphabet[:split_idx])  # Letters reserved for train
		test_alpha = set(alphabet[split_idx:])   # Letters reserved for test

		# Divide permutations into completely disjoint train and test sets
		train_letters = [[p for p in all_permutations if set(p).issubset(train_alpha)] for _ in types]
		test_letters = [[p for p in all_permutations if set(p).issubset(test_alpha)] for _ in types]

	elif train_test_letters == 'SemiOverlapping':
		np.random.shuffle(alphabet)
		split_idx = int(len(alphabet) * frac_train)
		train_alpha = set(alphabet[:split_idx])  # First letters for train set
		test_alpha = set(alphabet[split_idx:])   # First letters for test set

		# Initialize empty lists
		train_letters = []
		test_letters = []

		# Create the lists for each type
		train_letters = [
			[p for p in all_permutations if p[0] in train_alpha] if t == 0 else 
			[p for p in all_permutations if p[0] in train_alpha]
			for t in types
		]
		
		test_letters = [
			[p for p in all_permutations if p[0] in test_alpha] if t == 0 else
			[p for p in all_permutations if p[0] in test_alpha]
			for t in types
		]
		
	elif train_test_letters == 'Overlapping':
		# Shuffle list of all permutations
		list_permutations = [list(item) for item in all_permutations]
		np.random.shuffle(list_permutations)
		split_idx = int(frac_train * len(list_permutations))

		# Initialize empty lists
		train_letters = []
		test_letters = []

		# Create the lists for each type
		train_letters = [
			list_permutations[:split_idx] if t == 0 else list_permutations[:split_idx]
			for t in types
		]

		test_letters = [
			list_permutations[split_idx:] if t == 0 else list_permutations[split_idx:]
			for t in types
		]
	else:
		raise ValueError('train_test_letters should be Disjoint, SemiOverlapping, or Overlapping')

	tokens_train, labels_train = letter_to_seq(types, train_letters)
	X_train, y_train = seq_to_vectors(tokens_train, labels_train, alpha, letter_to_index, len(types), sim_datasplit, noise_level)
	tokens_test, labels_test = letter_to_seq(types, test_letters)
	X_test, y_test = seq_to_vectors(tokens_test, labels_test, alpha, letter_to_index, len(types), sim_datasplit, noise_level=0.0)

	return X_train, X_test, y_train, y_test, tokens_train, tokens_test, labels_train, labels_test

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
			range_ablate = 1
		else:
			range_ablate = n_hidden + 1

		for idx_ablate in range(range_ablate):
			for (_X, _y, token, label) in zip(X, y, tokens, labels):
				token = ''.join(token)

				Z, loss, yh = tokenwise_test(_X, _y, token, label, whichset, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task, idx_ablate=idx_ablate, n_hidden=n_hidden, delay=delay, cue_size=cue_size)

				if which_task == 'RNNClass':
					results['Loss'][token][epoch][idx_ablate] = loss
					results['Retrieval'][token][epoch][idx_ablate] = Z
					results['yh'][token][epoch][idx_ablate] = yh.detach().cpu().numpy()

				elif which_task == 'RNNPred':
					results['Loss'][epoch][idx_ablate].append(loss)
					results['Retrieval'][epoch][idx_ablate].append(Z)
					results['yh'][epoch][idx_ablate].append(yh.detach().cpu().numpy())

				elif which_task == 'RNNAuto':
					results['Loss'][token][epoch][idx_ablate] = loss
					results['Retrieval'][token][epoch][idx_ablate] = Z
					results['yh'][token][epoch][idx_ablate] = yh[0].detach().cpu().numpy()
					results['latent'][token][epoch][idx_ablate] = yh[1].detach().cpu().numpy()

def print_retrieval_color(test_task, losses, predicted_tokens, tokens_train, tokens_test):
	"""
	Prints retrieval loss values for different sets (train/test/non-patterned).
	
	Args:
		test_task (str): Name of the task.
		losses (list): Loss values corresponding to each predicted token.
		predicted_tokens (list): Tokens predicted by the model.
		tokens_train (list): List of training tokens.
		tokens_test (list): List of testing tokens.

	Returns:
		None
	"""
	tokens_train = [''.join(p) for p in tokens_train]
	tokens_test = [''.join(p) for p in tokens_test]
	tokens_all = np.append(tokens_train, tokens_test)

	meanval_train = np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_train])
	meanval_test = np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] in tokens_test])
	meanval_other = np.nanmean([losses[i] for i in range(len(losses)) if predicted_tokens[i] not in tokens_all])

	print(f'{test_task} Loss Tr {meanval_train:.2f} Loss Test {meanval_test:.2f} Loss NonPatt {meanval_other:.2f}', end='   ')
