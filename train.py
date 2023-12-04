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
# train network to produce next letter   #
##########################################

def train(X_train, model, optimizer, whichloss, n_epochs, n_batches, batch_size):

	if whichloss == 'CE':
		loss_function = \
			lambda output, target: F.cross_entropy(
					output.permute(1,2,0), 
					target.permute(1,2,0), reduction="mean")

	elif whichloss == 'MSE':
		loss_function = \
			lambda output, target: F.mse_loss(output, target,
					reduction="mean")
	else:
		print('Loss function not recognized!')

	n_train = X_train.shape[1]

	for epoch in range(n_epochs): 	

		# shuffle training data so that in each epoch data is split randomly in batches for training
		_ids = torch.randperm(n_train)

		# we are training in batches
		for batch in range(n_batches):
			optimizer.zero_grad() # Resets the gradients of all optimized torch.Tensor s.

			batch_start = batch * batch_size
			batch_end = (batch + 1) * batch_size

			X_batch = X_train[:, _ids[batch_start:batch_end], :]
			X_batch = X_batch.to(model.device)

			ht, hT, y_batch = model(X_batch)
			y_batch = y_batch.to(model.device)

			loss = loss_function(y_batch[:-1], X_batch[1:])
			loss.backward()
			optimizer.step()

##########################################
# test network 						     #
##########################################			

def test(X_train, X_test, tokens_train, tokens_test, tokens_other, model, L, alphabet, letter_to_index, index_to_letter, start):

	losses_train = []
	losses_test = []

	retr_train = []
	retr_test = []
	retr_other = []	

	with torch.no_grad():

		'''
		Calculate train error and perf
		'''
		
		X_train = X_train.to(model.device)
		ht, hT, y_train = model(X_train)
		y_train = y_train.to(model.device)

		loss = loss_function(y_train[:-1], X_train[1:])
		losses_train.append(loss.item())

		'''
		Calculate test error and performance
		'''

		X_test = X_test.to(model.device)
		ht, hT, y_test = model(X_test)
		y_test = y_test.to(model.device)

		loss = loss_function(y_test[:-1], X_test[1:])
		losses_test.append(loss.item())

		predicted_list_train, predicted_list_test, predicted_list_other = cued_retrieval(alphabet, tokens_train, tokens_test, tokens_other, model, letter_to_index, index_to_letter, L)

		retr_train.append(predicted_list_train)
		retr_test.append(predicted_list_test)
		retr_other.append(predicted_list_other)

	return losses_train, losses_test, np.array(retr_train), np.array(retr_test), np.array(retr_other)


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
	
	# print(pred_seq_train)
	# print(pred_seq_test)
	# print(pred_seq_other)
	
	return pred_seq_train, pred_seq_test, pred_seq_other

def predict(alpha, model, letter_to_index, index_to_letter, seq_start, next_letters):
	# model.eval()

	with torch.no_grad():

	# starts with a sequence of words of given length, initializes network

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
