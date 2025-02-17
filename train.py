import numpy as np
import torch
import torch.nn.functional as F
import random

loss_functions = {
	'RNNPred': {
		# 'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		'CE': lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1).view(-1, output.shape[-1]), \
									 torch.argmax(target,dim=-1).view(-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	},
	'RNNClass': {
		# 'CE': lambda output, target: F.cross_entropy(output, target, reduction="mean"),
		'CE': lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1), \
									 torch.argmax(target, dim=-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	},
	'RNNAuto': {
		'CE': lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1).view(-1, output.shape[-1]), \
									 torch.argmax(target, dim=-1).view(-1), reduction="mean"),
		'MSE': lambda output, target: F.mse_loss(output, target, reduction="mean")
	}
}

##########################################
# 			train network 				 #
##########################################

def train_batch(X_batch, y_batch, model, optimizer, loss_function, task, weight_decay=0., delay=0):
	
	optimizer.zero_grad()

	if task == 'RNNPred':
		ht, out_batch = model.forward(X_batch)
		loss = loss_function(out_batch[:-1], X_batch[1:])
	
	elif task == 'RNNClass':
		ht, out_batch = model.forward(X_batch, delay=delay)
		loss = loss_function(out_batch[-1], y_batch)

	elif task == 'RNNAuto':
		ht, latent, out_batch = model.forward(X_batch, delay=delay)
		loss = loss_function(out_batch, X_batch)
	
	if weight_decay > 0.:
		# adding L1 regularization to the loss
		# loss += weight_decay * torch.mean(torch.abs(model.h2h.weight))
		# adding L2 regularization to the loss
		loss += weight_decay * torch.linalg.matrix_norm(model.h2h.weight, ord=2) / model.h2h.weight.shape[0]**2 #.3 *

	loss.backward()
	optimizer.step()
	return

def train(X_train, y_train, model, optimizer, objective, L, n_batches, batch_size, alphabet, letter_to_index, index_to_letter, task, weight_decay=0., delay=0, teacher_forcing_ratio=0.5):

	if task in ['RNNPred', 'RNNClass', 'RNNAuto']:
		task_list = n_batches*[task]
	elif task == 'RNNMulti':
		task_list = np.random.choice(['RNNPred', 'RNNClass', 'RNNAuto'], size=n_batches)
	else:
		raise NotImplementedError(f"Task {task} not implemented")
	
	n_train = X_train.shape[1]

	model.train()
	# shuffle training data
	_ids = torch.randperm(n_train)

	# training in batches
	for batch, _task in enumerate(task_list):

		batch_start = batch * batch_size
		batch_end = (batch + 1) * batch_size

		X_batch = X_train[:, _ids[batch_start:batch_end], :].to(model.device)
		y_batch = y_train[_ids[batch_start:batch_end], :].to(model.device)

		if hasattr(model, 'set_task'):
			model.set_task(_task)

		train_batch(X_batch, y_batch, model, optimizer, loss_functions[_task][objective], _task, weight_decay=weight_decay, delay=delay)

		# # Implement scheduled sampling
		# use_teacher_forcing = random.random() < teacher_forcing_ratio
		# # print(use_teacher_forcing)
		# if use_teacher_forcing:
		# 	train_batch(X_batch, y_batch, model, optimizer, loss_functions[_task][objective], _task, weight_decay=weight_decay, delay=delay)
		# else:
		# 	# Use model's own predictions as inputs
		# 	input_seq = X_batch[0].unsqueeze(0)
			
		# 	for t in range(1, X_batch.size(0)):
		# 		_, output = model.forward(input_seq)
		# 		input_seq = torch.cat((input_seq, output[-1].unsqueeze(0)), dim=0)
		# 	train_batch(input_seq, y_batch, model, optimizer, loss_functions[_task][objective], _task, weight_decay=weight_decay, delay=delay)

	return

##########################################
# 				test network 			 #
##########################################


def tokenwise_test(X, y, token, label, whichset, model, L, alphabet, letter_to_index, index_to_letter, objective, task, n_hidden=10,
	idx_ablate=-1, # index of the hidden unit to ablate. -1 = no ablation
	delay=0,
	cue_size=1
	):

	if hasattr(model, 'set_task'):
		model.set_task(task)
	
	# Define loss functions
	loss_function = loss_functions[task][objective]

	model.eval()
	with torch.no_grad():

		X = X.to(model.device)

		mask = torch.ones(n_hidden)
		if idx_ablate != 0:
			mask[idx_ablate-1] = 0
		
		if task == 'RNNPred':
			ht, out = model.forward(X, mask=mask)
			# loss is btw activation of output layer at all but last time step (:-1) and target which is sequence starting from second letter (1:)
			loss = loss_function(out[:-1], X[1:])
			# CE between logits for retrieved sequence and token (input) -- NOT RELEVANT
			cue = [str(s) for s in token[:cue_size]] # put token[0] for cueing single letter
			predicted = predict(len(alphabet), model, letter_to_index, index_to_letter, cue, L-len(cue))
			predicted = ''.join(predicted)

		elif task == 'RNNClass':
			y = y.to(model.device)
			ht, out = model.forward(X, mask=mask, delay=delay)
			# loss is btw activation of output layer at last time step (-1) and target which is one-hot vector
			loss = loss_function(out[-1], y)   
			predicted = torch.argmax(out[-1], dim=-1)
			predicted = np.array([predicted])[0]

		elif task == 'RNNAuto':
			ht_, latent, out = model.forward(X, mask=mask, delay=delay)
			# loss is btw activation of output layer and input (target is input)
			loss = loss_function(out, X)
			# print(loss)
			predicted = np.array(torch.argmax(out, dim=-1))
			predicted = [index_to_letter[i] for i in predicted]
			predicted = ''.join(predicted)
			ht = (ht_, latent)  # Append along time dimension
	return predicted, loss.item(), ht

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
			_, y_pred = model.forward(x) #.permute(1,0,2)

			# last_letter_logits has dimension alpha
			last_letter_logits = y_pred[-1,:]
			# applies a softmax to transform activations into a proba, has dimensions alpha
			proba = torch.softmax(last_letter_logits, dim=0).detach().cpu().numpy()
			# then samples randomly from that proba distribution 
			letter_index = np.random.choice(len(last_letter_logits), p=proba)

			# appends it into the sequence produced
			seq_start.append(index_to_letter[letter_index])

	return seq_start