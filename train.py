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
from itertools import product


##########################################
# 			train network 				 #
##########################################


def train(X_train, X_test, y_train, y_test, tokens_train, tokens_test, tokens_other, model, optimizer, which_objective, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, which_task):

	# Define task-specific operations
	def prediction_task():
		return test(X_train, X_test, y_train, y_test, tokens_train, tokens_test, tokens_other, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task)

	def classification_task():
		return test(X_train, X_test, y_train, y_test, tokens_train, tokens_test, tokens_other, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task)

	task_operations = {
		'Pred': prediction_task,
		'Class': classification_task
	}

	# Common setup for loss functions
	if which_objective == 'CE':
		loss_function = lambda output, target: F.cross_entropy(output, target, reduction="mean")

	elif which_objective == 'MSE':
		loss_function = lambda output, target: F.mse_loss(output, target, reduction="mean")
	else:
		print('Loss function not recognized!')
		return

	# Adjust the loss function based on the task
	if which_task == 'Pred':
		# Adjust the output and target tensor dimensions for prediction task
		adjusted_loss_function = lambda output, target: loss_function(output.permute(1, 2, 0), target.permute(1, 2, 0))
	
	elif which_task == 'Class':
		# Use the loss function as is for classification task
		adjusted_loss_function = loss_function
	else:
		print('Task not recognized!')
		return

	n_train = X_train.shape[1]

	for epoch in range(n_epochs):
		# Call the task-specific function
		task_output = task_operations[which_task]()
		losses_train, losses_test = task_output[:2]

		# shuffle training data
		_ids = torch.randperm(n_train)

		# training in batches
		for batch in range(n_batches):

			optimizer.zero_grad()
			batch_start = batch * batch_size
			batch_end = (batch + 1) * batch_size
			X_batch = X_train[:, _ids[batch_start:batch_end], :].to(model.device)

			if which_task == 'Pred':
				ht, hT, out_batch = model(X_batch)
				loss = loss_function(out_batch[:-1], X_batch[1:])
			
			elif which_task == 'Class':
				y_batch = y_train[_ids[batch_start:batch_end], :].to(model.device)
				ht, hT, out_batch = model(X_batch)
				loss = loss_function(out_batch[-1], y_batch)

			loss.backward()
			optimizer.step()

	return task_output

##########################################
# 				test network 			 #
##########################################	

def test(X_train, X_test, y_train, y_test, tokens_train, tokens_test, tokens_other, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task):

	# Define loss functions
	loss_functions = {
		'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	}
	loss_function = loss_functions.get(which_objective)
	if loss_function is None:
		print('Loss function not recognized!')
		return

	# Initialize results containers
	results = {
		'Pred': ([], [], [], [], []),
		'Class': ([], [], [], [])
	}

	if which_task not in results:
		print('Task not recognized!')
		return

	n_train = X_train.shape[1] 
	n_test = X_test.shape[1] 

	with torch.no_grad():
		# Process training data
		X_train = X_train.to(model.device)
		ht, hT, out_train = model(X_train)
		process_task_data(X_train, y_train, out_train, n_train, which_task, loss_function, results, tokens_train, tokens_test, tokens_other, alphabet, letter_to_index, index_to_letter, L, model, 'train')

		# Process test data
		X_test = X_test.to(model.device)
		ht, hT, out_test = model(X_test)
		process_task_data(X_test, y_test, out_test, n_test, which_task, loss_function, results, tokens_train, tokens_test, tokens_other, alphabet, letter_to_index, index_to_letter, L, model, 'test')

	return tuple(results[which_task])

def process_task_data(X, y, out, n, which_task, loss_function, results, tokens_train, tokens_test, tokens_other, alphabet, letter_to_index, index_to_letter, L, model, data_type):
	
	# Shared operations
	if which_task == 'Pred':
		loss = loss_function(out[:-1], X[1:])
		
		if data_type == 'test':
			predicted_lists = cued_retrieval(alphabet, tokens_train, tokens_test, tokens_other, model, letter_to_index, index_to_letter, L)

			results[which_task][2].append(predicted_lists[0])  # retr_train
			results[which_task][3].append(predicted_lists[1])  # retr_test
			results[which_task][4].append(predicted_lists[2])  # retr_other    

	elif which_task == 'Class':
		
		y = y.to(model.device)
		loss = loss_function(out[-1], y)
		labels = torch.argmax(y, dim=-1)
		preds = torch.argmax(out[-1], dim=-1)
		accuracy = preds.eq(labels).sum().item() / n
		
		if data_type == 'train':
			results[which_task][2].append(accuracy)  # accuracies_train
		else:
			results[which_task][3].append(accuracy)  # accuracies_test

	# Append loss to respective list
	if data_type == 'train':
		results[which_task][0].append(loss.item())  # losses_train
	else:
		results[which_task][1].append(loss.item())  # losses_test

def cued_retrieval(alphabet, tokens_train, tokens_test, tokens_other, model, letter_to_index, index_to_letter, L):

	alpha=len(alphabet)

	pred_seq_train = np.zeros(np.shape(tokens_train)[0])
	pred_seq_test = np.zeros(np.shape(tokens_test)[0])
	pred_seq_other = np.zeros(np.shape(tokens_other)[0])

	cues = np.append(np.repeat(tokens_train[:,0], 1), np.repeat(tokens_test[:,0], 1))

	# cue each letter
	for cue in cues:
		pred_seq = predict(alpha, model, letter_to_index, index_to_letter, [cue], L-1)

		where=np.where((tokens_train == pred_seq).all(axis=1))
		if where[0] != np.ndarray([]):
			index=where[0][0]
			pred_seq_train[index] +=1

		where=np.where((tokens_test == pred_seq).all(axis=1))
		if where[0] != np.ndarray([]):
			index=where[0][0]
			pred_seq_test[index] +=1

		where=np.where((tokens_other == pred_seq).all(axis=1))
		if where[0] != np.ndarray([]):
			index=where[0][0]
			pred_seq_other[index] +=1

	return pred_seq_train, pred_seq_test, pred_seq_other

def predict(alpha, model, letter_to_index, index_to_letter, seq_start, next_letters):
	with torch.no_grad():

		# goes through each of the seq_start we want to predict
		for i in range(0, next_letters):
			
			x = torch.zeros((len(seq_start), alpha), dtype=torch.float32).to(model.device)
			pos = [letter_to_index[w] for w in seq_start[i:]]
			for k, p in enumerate(pos):
				x[k,:] = F.one_hot(torch.tensor(p), alpha)
			# y_pred should have dimensions 1 x L-1 x alpha, ours has dimension L x 1 x alpha, so permute
			# x has to have dimensions (L, sizetrain, alpha)

			_, _, y_pred = model(x)#.permute(1,0,2)

			# last_letter_logits has dimension alpha
			last_letter_logits = y_pred[-1,:]
			# applies a softmax to transform activations into a proba, has dimensions alpha
			proba = temperature_scaled_softmax(last_letter_logits, temperature=1.).detach().cpu().numpy()
			# then samples randomly from that proba distribution 
			letter_index = np.random.choice(len(last_letter_logits), p= proba)

			# appends it into the sequence produced
			seq_start.append(index_to_letter[letter_index])

	return seq_start

def temperature_scaled_softmax(logits, temperature=1.0):
	logits = logits / temperature
	return torch.softmax(logits, dim=0)
