import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from itertools import product

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


def predict_k_steps_rollout (model, hidden, k_steps):
	'''
	Perform the k-steps roll-out prediction.

	Parameters:
	----------

	model: RNN
		RNN model

	hidden: torch.Tensor
		Initial RNN state for the prediction

	k_steps: int
		Number or steps for the roll-out

	Returns:
	-------
	
	pred_tokens: torch.Tensor
		One-hot encoding of the predicted tokens

	logits: torch.Tensor
		Corresponding logits
	'''

	logits = []
	pred_tokens = []
	for _ in range(k_steps):

		# 1. Compute the next-token logits
		#    These will have to be returned
		output = model.h2o(hidden)
		logits.append(output)

		# 2. Sample the next token
		next_token = F.one_hot(
						torch.multinomial(output.softmax(dim=-1), 1).view(-1),
						num_classes=model.d_input
					 )
		pred_tokens.append(next_token)

		# 3. Update the state according to the prediction
		hidden = model.step(hidden, next_token.type(torch.float32))

	pred_tokens = torch.stack(pred_tokens)

	return pred_tokens, torch.stack(logits)


def compute_k_steps_rollout_loss (model, X_input, X_target, k_steps=3,
								loss_function=F.cross_entropy, **loss_kwargs
								):
	'''
	Compute the k-steps roll-out loss over a batch of sequences.

	Parameters:
	----------

	model: RNN
		RNN model

	X_input: torch.Tensor
		input token sequence (labels)

	X_target: torch.Tensor
		target token sequence (labels)

	k_steps: int (optional; default: 3)

	loss_function: callable (optional; default: torch.nn.functional.cross_entropy)
		Function that takes predicted logits and target token labels as inputs
		and returns the loss (tensor).

	Returns:
	-------

	loss: torch.Tensor
		Mean loss over batches, real time and roll-out time
	'''

	hidden, _ = model.forward(X_input)

	loss = 0
	for t, h_t in enumerate(hidden[:len(X_input)+1-k_steps]):

		# 1. compute the logits for the next k steps (using roll-out samples of tokens)
		rollout_tokens, rollout_logits = predict_k_steps_rollout(model, h_t, k_steps)
		target_tokens = torch.argmax(X_target[t:t+k_steps], dim=-1)
		rollout_tokens = torch.argmax(rollout_tokens, dim=-1)

		# 2. compute the loss against the target tokens (input)
		target_tokens = X_target[t:t+k_steps]
		loss += loss_function(rollout_logits, target_tokens, **loss_kwargs) / (len(X_input) - k_steps + 1)

	return loss


##########################################
# 			train network 				 #
##########################################

def train_batch(X_batch, y_batch, model, optimizer, loss_function, task, weight_decay=0., delay=0, k_steps=1):
	
	optimizer.zero_grad()

	if task == 'RNNPred':
		if k_steps == 1:
			ht, out_batch = model.forward(X_batch)
			loss = loss_function(out_batch[:-1], X_batch[1:])
		elif (k_steps > 0) and (k_steps < len(X_batch)):
			loss = compute_k_steps_rollout_loss(model, X_batch[:-1], X_batch[1:],
								k_steps=k_steps, loss_function=loss_function)
		else:
			raise ValueError(f"Invalid value for `k_steps` ({k_steps}). This must be a positive integer smaller than the target sequence length")
	
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

def train(X_train, y_train, model, optimizer, objective, n_batches, batch_size, task,
		weight_decay=0., delay=0, k_steps=1
		):

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

		train_batch(X_batch, y_batch, model, optimizer, loss_functions[_task][objective], _task,
					weight_decay=weight_decay, delay=delay, k_steps=k_steps
					)

	return

##########################################
# 				test network 			 #
##########################################

def test_save(results, model, X_train, X_test, y_train, y_test, tokens_train, tokens_test, letter_to_index, index_to_letter, which_task, which_objective, n_hidden, L, alphabet, delay, cue_size, epoch=None, idx_ablate = [], class_ablate=None):
	for (X, y, tokens) in zip([X_train, X_test], [y_train, y_test], [tokens_train, tokens_test]):
		try:
			X = X.permute((1,0,2))
		except:
			continue

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



if __name__ == "__main__":

	from model import RNN, LowRankLinear, FullRankLinear
	from torch.optim import Adam
	import matplotlib.pyplot as plt
	import os
	from os.path import join


	def generate_sequences(N: int, L: int):
	    sequences = []
	    pairs = [(a, b) for a, b in product(range(N), repeat=2) if a != b]
	    
	    for a, b in pairs:
	        # Pattern 1: ABABAB...
	        seq1 = [a if i % 2 == 0 else b for i in range(L)]
	        
	        # Pattern 2: AABBAABB...
	        seq2 = [a if (i // 2) % 2 == 0 else b for i in range(L)]

	        # Pattern 3: AAABBBAA...
	        seq3 = [a if (i // 3) % 2 == 0 else b for i in range(L)]
	        
	        sequences.append(seq1)
	        sequences.append(seq2)
	        sequences.append(seq3)

	    sequences = torch.Tensor(sequences).to(torch.int64).T
	    # print(sequences)
	    # exit()

	    return F.one_hot(sequences, num_classes=N).to(torch.float)


	def run_experiment(L, d_input, k_steps, n_epochs, model=None, n_snaps=100, **rnn_kwargs):

		# model and optimizer
		d_hidden = 64
		num_layers = 1
		d_output = d_input
		if model is None:
			model = RNN(d_input, d_hidden, num_layers, d_output, **rnn_kwargs)
		elif not isinstance(model, RNN):
			raise ValueError("`model` can only be None or RNN")
		optimizer = Adam(
				model.parameters(),	# to check if model.parameters() gives all and only the parameters we want
				lr=0.001, weight_decay=0.)

		# training dataset
		X = generate_sequences(d_input, L)

		# set up loss function
		loss_function = lambda output, target: F.nll_loss(F.log_softmax(output, dim=-1).view(-1, output.shape[-1]), \
		                                 torch.argmax(target,dim=-1).view(-1), reduction="mean")
		# loss_function = F.cross_entropy

		X_input, X_target = X[:-1], X[1:]

		losses = []
		for ep in range(n_epochs + 1):
			optimizer.zero_grad()
			if not k_steps: # 0 or None
				if not ep:
					print("Using standard next-token CE")
				_, output = model.forward(X_input)
				loss = loss_function(output, X_target)
			elif isinstance(k_steps, int) and (k_steps > 0):
				if not ep:
					print("Using k-steps rollout CE")
				loss = compute_k_steps_rollout_loss(model, X_input, X_target,
							k_steps=k_steps, loss_function=loss_function)

			losses.append(loss.item())
			if not ep % n_snaps:
				print(f"{str(ep):<6}", losses[-1])

			loss.backward()
			optimizer.step()


		return model, losses



	L=6
	d_input = 4

	out_dir = "tests_k-steps"
	os.makedirs(out_dir, exist_ok=True)

	fig, ax = plt.subplots()

	color='r'
	rnn_kwargs = dict(
			layer_type=nn.Linear
		)
	k_steps = None
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='Linear, standard', ls='-')

	k_steps = 1
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='Linear, k=1', ls='--')
	
	k_steps = 3
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='Linear, k=3', ls=':')

	color='g'
	rnn_kwargs = dict(
			layer_type=LowRankLinear,
			max_rank=None
		)
	k_steps = None
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank None, standard', ls='-')

	k_steps = 1
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank None, k=1', ls='--')

	k_steps = 3
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank None, k=3', ls=':')

	color='b'
	rnn_kwargs = dict(
			layer_type=LowRankLinear,
			max_rank=2
		)
	k_steps = None
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank 2, standard', ls='-')

	k_steps = 1
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank 2, k=1', ls='--')

	k_steps = 3
	model, losses = run_experiment (L, d_input, k_steps, 1000, model=None, n_snaps=20, **rnn_kwargs)
	ax.plot(losses, c=color, label='rank 2, k=3', ls=':')
	
	ax.legend(loc='best')
	fig.savefig(join(out_dir, f"losses_dInput{d_input}"))

	plt.close(fig)
