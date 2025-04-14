import numpy as np
import torch
import torch.nn.functional as F

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

def train(X_train, y_train, model, optimizer, objective, n_batches, batch_size, task, weight_decay=0., delay=0):

	if task in ['RNNPred', 'RNNClass', 'RNNAuto']:
		task_list = n_batches*[task]
	elif task == 'RNNMulti':
		task_list = np.random.choice(['RNNPred', 'RNNClass', 'RNNAuto'], size=n_batches)
	else:
		raise NotImplementedError(f"Task {task} not implemented")
	
	n_train = X_train.shape[1]
	# print(task_list)
	model.train()
	# shuffle training data
	_ids = np.random.permutation(n_train)

	# training in batches
	for batch, _task in enumerate(task_list):

		batch_start = batch * batch_size
		batch_end = (batch + 1) * batch_size

		X_batch = X_train[:, _ids[batch_start:batch_end], :].to(model.device)
		y_batch = y_train[_ids[batch_start:batch_end], :].to(model.device)

		if hasattr(model, 'set_task'):
			model.set_task(_task)

		train_batch(X_batch, y_batch, model, optimizer, loss_functions[_task][objective], _task, weight_decay=weight_decay, delay=delay)

	return

##########################################
# 				test network 			 #
##########################################

def test_save(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, alphabet, delay, cue_size, epoch=None, idx_ablate = [], class_ablate=None):
	for (X, y, tokens) in zip([X_train, X_test], [y_train, y_test], [tokens_train, tokens_test]):
		X = X.permute((1,0,2))

		for (_X, _y, token) in zip(X, y, tokens):
			token = ''.join(token)

			output, loss, hidden = tokenwise_test(_X, _y, token, model, L, alphabet, letter_to_index, index_to_letter, which_objective, which_task, n_hidden=n_hidden, delay=delay, cue_size=cue_size, idx_ablate = idx_ablate)

			if idx_ablate == []:
				whichkey = epoch
			else:
				whichkey = class_ablate

			results['Loss'][token][whichkey] = loss
			results['Retrieval'][token][whichkey] = output

			if which_task == 'RNNClass' or which_task == 'RNNPred':
				results['HiddenAct'][token][whichkey] = hidden.detach().cpu().numpy()
			
			elif which_task == 'RNNAuto':
				results['HiddenAct'][token][whichkey] = hidden[0].detach().cpu().numpy()
				results['LatentAct'][token][whichkey] = hidden[1].detach().cpu().numpy()


def tokenwise_test(X, y, token, model, L, alphabet, letter_to_index, index_to_letter, objective, task, n_hidden, delay=0, cue_size=1, idx_ablate = []):
	if hasattr(model, 'set_task'):
		model.set_task(task)
	
	# Define loss functions
	loss_function = loss_functions[task][objective]

	model.eval()
	with torch.no_grad():

		X = X.to(model.device)

		mask = torch.ones(n_hidden)
		if idx_ablate == []:
			pass
		else:
			mask[idx_ablate] = 0
		
		if task == 'RNNPred':
			ht, out = model.forward(X, mask=mask)
			# loss is btw activation of output layer at all but last time step (:-1) and target which is sequence starting from second letter (1:)
			loss = loss_function(out[:-1], X[1:])
			# CE between logits for retrieved sequence and token (input) -- NOT RELEVANT
			cue = [str(s) for s in token[:cue_size]] # put token[0] for cueing single letter
			predicted = predict(len(alphabet), model, letter_to_index, index_to_letter, cue, L)
			predicted = ''.join(predicted)

		elif task == 'RNNClass':
			y = y.to(model.device)
			ht, out = model.forward(X, mask=mask, delay=delay)
			# loss is btw activation of output layer at last time step (-1) and target which is one-hot vector
			loss = loss_function(out[-1], y)   
			predicted = torch.argmax(out[-1], dim=-1)
			predicted = np.array([predicted])[0]
			# jac = model.jacobian(ht[-1], X[-1])#.detach().cpu().numpy()
			# singular_values = torch.linalg.svdvals(jac)
			# print(jac)
			# print("Singular values:", singular_values)
			# print(torch.max(singular_values))
			# exit()

		elif task == 'RNNAuto':
			ht_, latent, out = model.forward(X, mask=mask, delay=delay)
			
			# loss is btw activation of output layer and input (target is input)
			loss = loss_function(out, X)
			predicted = np.array(torch.argmax(out, dim=-1))
			predicted = [index_to_letter[i] for i in predicted]
			predicted = ''.join(predicted)
			ht = (ht_, latent)  # Append along time dimension
	return predicted, loss.item(), ht

def predict(alpha, model, letter_to_index, index_to_letter, seq_start, len_next_letters):
	with torch.no_grad():

		# goes through each of the seq_start we want to predict
		for i in range(0, len_next_letters):

			# define x as a sequence of one-hot vectors
			# corresponding to the letters cued
			x = torch.zeros((len(seq_start), alpha), dtype=torch.float32).to(model.device)
			pos = [letter_to_index[w] for w in seq_start]

			for k, p in enumerate(pos):
				x[k,:] = F.one_hot(torch.tensor(p), alpha)

			_, y_pred = model.forward(x)

			# last_letter_logits has dimension alpha
			last_letter_logits = y_pred[-1,:]
			# applies a softmax to transform activations into a proba, has dimensions alpha
			proba = torch.softmax(last_letter_logits, dim=0).detach().cpu().numpy()
			# then samples randomly from that proba distribution 
			# letter_index = np.random.choice(len(last_letter_logits), p=proba)
			letter_index = np.argmax(proba)

			# appends it into the sequence produced
			seq_start.append(index_to_letter[letter_index])

	return seq_start