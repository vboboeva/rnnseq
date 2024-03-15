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

def train(X_train, X_test, y_train, y_test, model, optimizer, which_objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, results, which_task ):

	loss_functions = {
		'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		# 'CE': lambda output, target: F.nll_loss(output, torch.argmax(target, dim=-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	}
	# Define loss functions
	loss_function = loss_functions.get(which_objective)

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

	model.train()
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
			# print("=================")
			# print("which_task =", which_task,
			# 		" out_batch.shape=", out_batch.shape,
			# 		" ht.shape=", ht.shape,
			# 		" hT.shape=", hT.shape,
			# 	)
			# print("=================")
			# exit()
			loss = loss_function(out_batch[:-1], X_batch[1:])
		
		elif which_task == 'Class':
			y_batch = y_train[_ids[batch_start:batch_end], :].to(model.device)
			ht, hT, out_batch = model(X_batch)
			# print(ht, hT, out_batch)
			# print("=================")
			# print("which_task =", which_task,
			# 		" out_batch.shape=", out_batch.shape,
			# 		" ht.shape=", ht.shape,
			# 		" hT.shape=", hT.shape,
			# 	)
			# print("=================")
			# exit()
			loss = loss_function(out_batch[-1], y_batch)

		loss.backward()
		optimizer.step()

	return results

##########################################
# 				test network 			 #
##########################################


def test(X_train, X_test, y_train, y_test, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task, results):

	tokens_train=results['Loss']['train'].keys()
	tokens_test=results['Loss']['test'].keys()
	tokens_other=results['Loss']['other'].keys()

	loss_functions = {
		'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		# 'CE': lambda output, target: F.nll_loss(output, torch.argmax(target, dim=-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	}
	# Define loss functions
	loss_function = loss_functions.get(which_objective)

	n_train = X_train.shape[1] 
	n_test = X_test.shape[1] 

	model.eval()
	with torch.no_grad():

		# results['Whh'].append(model.rnn.weight_hh_l0.detach().cpu().numpy())
		
		for tokens, X, y, whichset, n in zip([tokens_train, tokens_test], [X_train, X_test], [y_train, y_test], ['train','test'], [n_train, n_test]):
			X = X.permute((1,0,2))
			X = X.to(model.device)

			for i, (_X, _y, token) in enumerate(zip(X, y, tokens)):
				if which_task == 'Pred':
					ht, hT, out = model(_X)
					# print("=================")
					# print("which_task =", which_task,
					# 		" out.shape=", out.shape,
					# 		" ht.shape=", ht.shape,
					# 		" hT.shape=", hT.shape,
					# 	)
					# print("=================")
					# exit()
					loss = loss_function(out[:-1], _X[1:])

				elif which_task == 'Class':
					_y = _y.to(model.device)
					ht, hT, out = model(_X)
					# print("=================")
					# print("which_task =", which_task,
					# 		" out.shape=", out.shape,
					# 		" ht.shape=", ht.shape,
					# 		" hT.shape=", hT.shape,
					# 	)
					# print("=================")
					# exit()
					loss = loss_function(out[-1], _y)
					labels = torch.argmax(_y, dim=-1)
					preds = torch.argmax(out[-1], dim=-1)
					accuracy = preds.eq(labels).sum().item()					
					results['Accuracy'][whichset][token].append(accuracy)
					# print(loss)

				results['Loss'][whichset][token].append(loss)
				results['yh'][whichset][token].append(ht.detach().cpu().numpy())

		# cued retrieval for testing prediction task
		if which_task == 'Pred':
			cues = np.append([k[0] for k in results['Loss']['train'].keys()], [k[0] for k in results['Loss']['test'].keys()])
			alpha=len(alphabet)
			# cue each letter
			for cue in cues:
				pred_seq = predict(alpha, model, letter_to_index, index_to_letter, [cue], L-1)
				pred_seq = ''.join(pred_seq)
				if pred_seq in results['Retrieval']['train']:
					# print('train')
					results['Retrieval']['train'][pred_seq].append(1)
				elif pred_seq in results['Retrieval']['test']:
					# print('test')
					results['Retrieval']['test'][pred_seq].append(1)
				elif pred_seq in results['Retrieval']['other']:
					# print('other')
					results['Retrieval']['other'][pred_seq].append(1)
				else:
					print('Predicted sequence not in dictionary!')
	return

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
