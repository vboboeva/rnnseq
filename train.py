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


loss_functions = {
	'Pred': {
		# 'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		'CE': lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1).view(-1, output.shape[-1]), torch.argmax(target,dim=-1).view(-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	},
	'Class': {
		# 'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		'CE': lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1), torch.argmax(target, dim=-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	}
}

##########################################
# 			train network 				 #
##########################################

def train(X_train, y_train, model, optimizer, which_objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, which_task, weight_decay=0., delay=0):

	# Define loss functions
	loss_function = loss_functions[which_task][which_objective]

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
		# print(X_batch.shape)
		# exit()

		if which_task == 'Pred':
			ht, hT, out_batch = model.forward(X_batch)
			loss = loss_function(out_batch[:-1], X_batch[1:])
		
		elif which_task == 'Class':
			y_batch = y_train[_ids[batch_start:batch_end], :].to(model.device)
			ht, hT, out_batch = model.forward(X_batch, delay=delay)
			loss = loss_function(out_batch[-1], y_batch)

		# # adding L1 regularization to the loss
		# if weight_decay > 0.:
		# 	loss += weight_decay * torch.mean(torch.abs(model.h2h.weight))
		# 	loss += .3 * weight_decay * torch.linalg.matrix_norm(model.h2h.weight, ord=2) / model.h2h.weight.shape[0]**2

		loss.backward()
		optimizer.step()

	return 

##########################################
# 				test network 			 #
##########################################


def test(X, y, token, label, whichset, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task,
	n_hidden=10,
	idx_ablate=-1, # index of the hidden unit to ablate. -1 = no ablation
	delay=0,
	cue_size=1
	):

	# Define loss functions
	loss_function = loss_functions[which_task][which_objective]

	model.eval()
	with torch.no_grad():

		X = X.to(model.device)

		mask = torch.ones(n_hidden)
		if idx_ablate != 0:
			mask[idx_ablate-1] = 0
		
		if which_task == 'Pred':

			ht, hT, out = model.forward(X, mask=mask)
			# loss is btw activation of output layer at all but last time step (:-1) and target which is sequence starting from second letter (1:)
			loss = loss_function(out[:-1], X[1:])
			# CE between logits for retrieved sequence and token (input) -- NOT RELEVANT

			cue= [str(s) for s in token[:cue_size]] # put token[0] for cueing single letter

			pred_seq = predict(len(alphabet), model, letter_to_index, index_to_letter, cue, L-len(cue))
			pred_seq = ''.join(pred_seq)
			metric = pred_seq

		elif which_task == 'Class':
			y = y.to(model.device)
			ht, hT, out = model.forward(X, mask=mask, delay=delay)
			# loss is btw activation of output layer at last time step (-1) and target which is one-hot vector
			loss = loss_function(out[-1], y)   
			label = torch.argmax(y, dim=-1)
			predicted = torch.argmax(out[-1], dim=-1)
			metric = np.array(predicted)

	return metric, loss.item(), ht.detach().cpu().numpy()

def predict(alpha, model, letter_to_index, index_to_letter, seq_start, len_next_letters):
	with torch.no_grad():

		# goes through each of the seq_start we want to predict
		for i in range(0, len_next_letters):
			
			x = torch.zeros((len(seq_start), alpha), dtype=torch.float32).to(model.device)
			pos = [letter_to_index[w] for w in seq_start[i:]]
			for k, p in enumerate(pos):
				x[k,:] = F.one_hot(torch.tensor(p), alpha)
			# y_pred should have dimensions 1 x L-1 x alpha, ours has dimension L x 1 x alpha, so permute
			# x has to have dimensions (L, sizetrain, alpha)
			_, _, y_pred = model.forward(x) #.permute(1,0,2)

			# last_letter_logits has dimension alpha
			last_letter_logits = y_pred[-1,:]
			# applies a softmax to transform activations into a proba, has dimensions alpha
			proba = temperature_scaled_softmax(last_letter_logits, temperature=1.).detach().cpu().numpy()
			# then samples randomly from that proba distribution 
			letter_index = np.random.choice(len(last_letter_logits), p=proba)

			# appends it into the sequence produced
			seq_start.append(index_to_letter[letter_index])

	return seq_start

def temperature_scaled_softmax(logits, temperature=1.0):
	logits = logits / temperature
	return torch.softmax(logits, dim=0)
