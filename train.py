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



def train(X_train, X_test, tokens_train, tokens_test, model, optimizer, whichloss, L, n_epochs, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, start):

	alpha=len(alphabet)

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

	train_losses = []
	test_losses = []
	train_accuracies = []
	test_accuracies = []

	n_train = X_train.shape[1] #len(train_ids)
	n_test = X_test.shape[1] #len(test_ids)

	for epoch in tqdm(range(n_epochs)):
		# print(epoch)
	
		with torch.no_grad():
			'''
			Calculate train error and accuracy
			'''
			X_train = X_train.to(model.device)
			ht, hT, y_train = model(X_train)
			y_train = y_train.to(model.device)

			loss = loss_function(y_train[:-1], X_train[1:])
			train_losses.append(loss.item())

			compute_accuracies(X_train, 'train', tokens_train, tokens_test, train_accuracies, alpha, model, letter_to_index, index_to_letter, L, start)

			'''
			Calculate test error and accuracy
			'''
			X_test = X_test.to(model.device)
			ht, hT, y_test = model(X_test)
			y_test = y_test.to(model.device)

			loss = loss_function(y_test[:-1], X_test[1:])
			test_losses.append(loss.item())

			compute_accuracies(X_test, 'test', tokens_train, tokens_test, test_accuracies, alpha, model, letter_to_index, index_to_letter, L, start)

		##########################################
		# train network to produce next letter   #
		##########################################

		# shuffle training data so that in each epoch data is split randomly in batches for training

		_ids = torch.randperm(n_train)
		# np.random.shuffle(train_ids)

		# we are training in batches
		accuracy = 0
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

	# AFTER THE TRAINING

	seq_retrieved = cued_retrieval(X_train, alphabet, tokens_train, tokens_test, model, letter_to_index, index_to_letter, L)

	return train_losses, test_losses, train_accuracies, test_accuracies, seq_retrieved

def compute_accuracies(X, whichset, tokens_train, tokens_test, accuracies, alpha, model, letter_to_index, index_to_letter, L, start):

	in_train=[]
	in_test=[]
	in_none=[]

	if whichset == 'train':
		tokens=tokens_train
	elif whichset == 'test':
		tokens=tokens_test

	# X is L by n_train by alpha

	for i in range(len(X[0,:,0])):
		se = tokens[i]
		seq = [se[j] for j in range(len(se))] # convert string to list of letters
		pred_seq = predict(alpha, model, letter_to_index, index_to_letter, seq[:start], L-start)

		if (tokens_train == pred_seq).all(axis=1).any():
			# print('in train', se, pred_seq)
			in_train += [pred_seq]
			# print(whichset, seq,  10*'-', pred_seq, 'in train')
		elif (tokens_test == pred_seq).all(axis=1).any():
			# print('in test', se, pred_seq)
			in_test += [pred_seq]
			# print(whichset, seq, 10*'-', pred_seq, 'in test')
		else:
			# print('in none', se, pred_seq)
			in_none += [pred_seq]
			# print(whichset, seq, 10*'-', pred_seq, 'in none')
	
	if whichset == 'train':
		accuracies.append( len(in_train)/len(X[0,:,0]) )
		# print( len(in_train)/len(X[0,:]) )
	
	elif whichset == 'test':
		accuracies.append( len(in_test)/len(X[0,:,0]) )
		# print( len(in_test)/len(X[0,:]) )


def cued_retrieval(X, alphabet, tokens_train, tokens_test, model, letter_to_index, index_to_letter, L):

	alpha=len(alphabet)

	pred_seq_list = np.zeros(np.shape(np.vstack((tokens_train, tokens_test)))[0])

	for cue_letter in np.repeat(alphabet,100):
		
		pred_seq = predict(alpha, model, letter_to_index, index_to_letter, [cue_letter], L-1)

		for i, seq in enumerate(np.vstack((tokens_train, tokens_test))):
			seq = [seq[j] for j in range(len(seq))] # convert string to list of letters
			if pred_seq == seq:
				pred_seq_list[i] += 1

	return pred_seq_list

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
