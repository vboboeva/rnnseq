'''
RNN network
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############################################################################

def freeze(module):
	for name, pars in module.named_parameters():
		pars.requires_grad = False
		

class LinearWeightDropout(nn.Linear):
	'''
	Linear layer with weights dropout (synaptic failures)
	'''
	def __init__(self, in_features, out_features, drop_p=0.0, **kwargs):
		super().__init__(in_features, out_features, **kwargs)
		self.drop_p = drop_p

	def forward(self, input):
		new_weight = (torch.rand((input.shape[0], *self.weight.shape), device=input.device) > self.drop_p) * self.weight[None, :, :]
		output = torch.bmm(new_weight, input[:, :, None])[:, :, 0] / (1. - self.drop_p)
		if self.bias is None:
			return output
		return output + self.bias


class Net(nn.Module):
	'''
	Base class for network models.
	Attribute function `init_weights` is a custom weight initialisation.
	'''

	def init_weights (self, scaling):

		scaling_arr = scaling.split(",")
		assert len(scaling_arr) in [1, len(list(self.named_parameters()))], \
			"The `scaling` parameter must be a string with one of the available options, "+\
			"or multiple available options comma-separated (as many as the number of layers)"
		
		for l, (name, pars) in enumerate(self.named_parameters()):
			if len(scaling_arr) == 1:
				scaling = scaling_arr[0]
			else:
				scaling = scaling_arr[l]

			if "weight" in name:
				f_in = 1.*pars.data.size()[1]
				if scaling == "lin":
					# initialisation of the weights -- N(0, 1/n)
					init_f = lambda f_in: (0., 1./f_in)
				elif scaling == "lin+":
					# initialisation of the weights -- N(0, 1/n)
					init_f = lambda f_in: (1./f_in, 1./f_in)
				elif scaling == "sqrt":
					# initialisation of the weights -- N(0, 1/sqrt(n))
					init_f = lambda f_in: (0., 1./np.sqrt(f_in))
				elif scaling == "const":
					# initialisation of the weights independent of n
					init_f = lambda f_in: (0., 0.001)
				elif scaling == "const+":
					# initialisation of the weights independent of n
					init_f = lambda f_in: (0.001, 0.001)
				elif isinstance(scaling, float) and scaling > 0:
					# initialisation of the weights -- N(0, 1/n**alpha)
					'''
					UNTESTED
					'''
					init_f = lambda f_in: (0., 1./np.power(f_in, scaling))
				else:
					raise ValueError(
						f"Invalid scaling option '{scaling}'\n" + \
						 "Choose either 'sqrt', 'lin' or a float larger than 0")

				mu, sigma = init_f(f_in)
				pars.data.normal_(mu, sigma)

	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location=self.device))

	def __len__ (self):
		return len(self._modules.items())

class RNN (Net):

	def __init__(self, d_input, d_hidden, num_layers, d_output,
			output_activation=None, # choose btw softmax for classification vs linear for regression tasks
			drop_l=None,
			nonlinearity='relu',
			layer_type=nn.Linear,
			init_weights=None,
			model_filename=None, # file with model parameters
			to_freeze = [], # parameters to keep frozen; list with elements in ['i2h', 'h2h', 'h2o']
			from_file = [], # parameters to set from file; list with elements in ['i2h', 'h2h', 'h2o']
			bias=True,
			device="cpu",
			train_i2h = True,
		):

		super(RNN, self).__init__()
		init=init_weights
		self._from_file = []
		self._model_filename=model_filename

		self.device = device
		# Defining the number of layers and the nodes in each layer
		self.d_input = d_input
		self.d_output = d_output
		self.d_hidden = d_hidden

		# if train_i2h:
		#     self.i2h = nn.Linear (d_input, d_hidden, bias=bias)

		# else:
		#     # MANUALLY DEFINE INPUT WEIGHTS HERE
		#     self._input_weights = torch.cat([torch.eye(self.d_input), torch.zeros(self.d_hidden - self.d_input, self.d_input)]) 

		#     # _n_repeats = self.d_hidden // self.d_input
		#     # assert self.d_hidden % self.d_input == 0, \
		#     #     "Hidden layer size should be integer multiple of input size"
		#     # self._input_weights = torch.repeat_interleave(_input_weights, _n_repeats, dim=0)
			
		#     self._input_weights.requires_grad = False

		#     self.i2h = lambda x: torch.matmul( x, self._input_weights.T )

		self.i2h = layer_type(d_input, d_hidden, bias=0)
		if 'i2h' in to_freeze:
			freeze(self.i2h)
		if 'i2h' in from_file:
			self._from_file += ['i2h.'+n for n,_ in self.i2h.named_parameters()]

		self.h2h = layer_type(d_hidden, d_hidden, bias=bias)
		if 'h2h' in to_freeze:
			freeze(self.h2h)
		if 'h2h' in from_file:
			self._from_file += ['h2h.'+n for n,_ in self.h2h.named_parameters()]

		self.h2o = layer_type(d_hidden, d_output, bias=0)
		if 'h2o' in to_freeze:
			freeze(self.h2o)
		if 'h2o' in from_file:
			self._from_file += ['h2o.'+n for n,_ in self.h2o.named_parameters()]

		if nonlinearity in [None, 'linear']:
			self.phi = lambda x: x
		elif nonlinearity == 'relu':
			self.phi = lambda x: F.relu(x)
		elif nonlinearity == 'sigmoid':
			self.phi = lambda x: F.sigmoid(x)
		elif nonlinearity == 'tanh':
			self.phi = lambda x: F.tanh(x)
		else:
			raise NotImplementedError("activation function " + \
							f"\"{nonlinearity}\" not implemented")

		if output_activation in [None, 'linear']:
			self.out_phi = lambda x: x
		elif output_activation == 'softmax':
			self.out_phi = lambda x: F.softmax(x, dim=-1)
		else:
			raise NotImplementedError("output activation function " + \
							f"\"{output_activation}\" not implemented")

		# convert drop_l into a list of strings
		if drop_l == None:
			drop_l = ""
		elif drop_l == "all":
			drop_l = ",".join([str(i+1) for i in range(self.n_layers)])
		drop_l = drop_l.split(",")

		self.init_weights (init)

	def init_weights(self, init, seed=None):
		
		_pars_dict = {}
		if self._model_filename is not None:
			# The parameters to be set from file are removed from the list of
			# parameters to which the initialisation rule is applied
			print(f"Loading parameters from file '{self._model_filename}'")
			_pars_dict = torch.load(self._model_filename)

		if seed is not None:
			torch.manual_seed(seed)

		if init is None:
			pass
		elif init == "Rich":
			# initialisation of the weights -- N(0, 1/n)
			init_f = lambda f_in: 1./f_in
		elif init == "Lazy":
			# initialisation of the weights -- N(0, 1/sqrt(n))
			init_f = lambda f_in: 1./np.sqrt(f_in)
		elif init == "Const":
			# initialisation of the weights independent of n
			init_f = lambda f_in: 0.001
		else:
			raise ValueError(
				f"Invalid init option '{init}'\n" + \
				 "Choose either None, 'Rich', 'Lazy' or 'Const'")
		
		for name, pars in self.named_parameters():
			if name in self._from_file:
				# print(pars)
				# print(_pars_dict[name])
				pars.data = _pars_dict[name]
			elif init is not None:
				if "weight" in name:
					f_in = 1.*pars.data.size()[1]
					std = init_f(f_in)
					pars.data.normal_(0., std)

		# # check
		# for name, pars in self.named_parameters():
		#     print(name, "\t", torch.max(pars - _pars_dict[name]))
		# exit()
		# # end check

	def forward(self, x, mask=None, delay=0):
		'''
		x
		---
		seq_length, batch_size, d_input
		or
		seq_length, d_input
		'''

		# print("x.shape (input) ", x.shape)
		# print("mask.shape ", mask.shape)

		if mask is not None:
			assert isinstance(mask, torch.Tensor) and mask.shape == (self.d_hidden,), \
				f"`mask` must be a 1D torch tensor with the same size as the hidden layer"
			mask = mask.to(self.device)
			_masking = lambda h: h * mask[None,:]
		else:
			_masking = lambda h: h

		# _shape = seq_length, batch_size, n_hidden
		# or 
		# _shape = seq_length, n_hidden
		# This is saved here to remove the batch dimension
		# after the processing, if it is not present in input
		_shape = *x.shape[:-1], self.d_hidden
		# print("_shape ", _shape)

		# If batch dimension missing, add it (in the middle -- as in the
		# default implementation of pytorch RNN module).
		# This allows to treat an input containing a batch of sequences
		# in the same way as a single sequence.
		# print(x.shape)
		if len(x.shape) == 2:
			x = torch.reshape(x, (x.shape[0], 1, x.shape[1]) )
		# print("x.shape (reshaped) ", x.shape)

		# pad input with 0 along the time axis for `delay` time steps

		if delay != 0:
			assert isinstance(delay, int), "delay must be an integer"
			x = torch.cat([x, torch.zeros((delay,*x.shape[1:]))], dim=0)
			# x = torch.cat([x, torch.zeros((x.shape[0], delay, x.shape[-1]))], dim=1)

			_shape = _shape[0]+delay, *_shape[1:]

		# initialization of net        
		# h0 = torch.randn(x.shape[1], self.d_hidden)
		h0 = torch.zeros(x.shape[1], self.d_hidden)
		
		# batch_size, n_hidden
		ht = _masking(h0)
		hidden = []

		# t is the sequence of time-steps
		for t, xt in enumerate(x):

			# xt = batch_size, n_input
			# zt = batch_size, n_hidden

			# process input to feed into recurrent network
			zi = self.i2h (xt)
			zh = self.h2h (ht)
			z = self.phi (zh + zi)
			z = _masking(z)

			hidden.append(z)
			ht = z 

		# print('before', np.shape(hidden))
		hidden = torch.reshape(torch.stack(hidden), _shape)
		# print('after', np.shape(hidden))

		output = self.h2o(hidden)
		# print('outputshape', np.shape(output))

		return hidden, output

	def get_activity(self, x):

		with torch.no_grad():

			_x = x.clone().detach().to(self.device)

			ht, hT, _  = self.forward(_x)

			# print('ht', np.shape(ht)) # activity for whole sequence
			# print('hT', np.shape(hT)) # activity for last element of sequence

			y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

		return y

class RNNEncoder(nn.Module):
	def __init__(self, d_input, d_hidden, d_latent, num_layers):
		super(RNNEncoder, self).__init__()
		# RNN: d_input -> d_latent
		self.rnn = RNN(d_input, d_hidden, num_layers, d_latent)

	def forward(self, x, delay=0):
		# x: (sequence_length, batch_size, d_input) 
		# output: (sequence_length, batch_size, d_hidden)
		# h: (num_layers, batch_size, d_hidden)
		rnn_out, latent = self.rnn(x, delay=delay)
		# return activity of latent layer 
		return rnn_out, latent[-1]

class RNNDecoder(nn.Module):
	def __init__(self, d_latent, d_hidden, d_output, num_layers, sequence_length):
		super(RNNDecoder, self).__init__()
		# RNN: d_latent -> d_output

		self.rnn = RNN(d_latent, d_hidden, num_layers, d_output) 
		self.sequence_length = sequence_length

	def forward(self, latent, delay=0):
		'''
		latent is either
		- (batch_dim, d_latent)
		or
		- (d_latent,)
		'''
		# repeat latent vector as many as sequence length
		latent_expanded = latent.unsqueeze(0).expand(self.sequence_length, *[-1] * latent.dim())

		# Expand the latent vector to match the sequence length, fill with zeros
		# filled_tensor = torch.zeros_like(latent_expanded)
		# filled_tensor[0,:] = latent

		rnn_out, output = self.rnn(latent_expanded, delay=0)
		return output

class RNNAutoencoder(nn.Module):
	def __init__(self, d_input, d_hidden, num_layers, d_latent, sequence_length, device="cpu"):
		super(RNNAutoencoder, self).__init__()

		self.d_input = d_input
		self.d_hidden = d_hidden
		self.d_latent = d_latent
		self.num_layers = num_layers
		self.sequence_length = sequence_length
		self.device = device

		self.encoder = RNNEncoder(d_input, d_hidden, d_latent, num_layers)
		self.decoder = RNNDecoder(d_latent, d_hidden, d_input, num_layers, sequence_length)

	def forward(self, x, mask=None, delay=0):

		self.delay = delay
		if mask is not None:
			assert isinstance(mask, torch.Tensor) and mask.shape == (self.d_hidden,), \
				f"`mask` must be a 1D torch tensor with the same size as the hidden layer"
			mask = mask.to(self.device)
			_masking = lambda h: h * mask[None,:]
		else:
			_masking = lambda h: h

		hidden, latent = self.encoder(x, delay=self.delay)
		reconstructed = self.decoder(latent, delay=0)

		return hidden, latent, reconstructed

class RNNMulti (nn.Module):
	def __init__(self, d_input, d_hidden, num_layers, d_latent, num_classes, sequence_length, device="cpu", model_filename=None, from_file=[], to_freeze=[], init_weights=None, layer_type=nn.Linear):
		super(RNNMulti, self).__init__()
		self.rnn = RNN(d_input, d_hidden, num_layers, d_latent)
		self.out_class = nn.Linear(d_hidden, num_classes)
		self.out_pred = nn.Linear(d_hidden, d_input)
		self.out_auto = RNNDecoder(d_latent, d_hidden, d_input, num_layers, sequence_length)
		self.device = device
		self.task = None

	@property
	def h2h (self):
		return self.rnn.h2h

	def set_task(self, task):
		self.task = task

	def forward(self, x, mask=None, delay=0):

		hidden, output = self.rnn(x, mask=mask, delay=delay)

		if self.task == 'RNNClass':
			output = self.out_class(hidden)
			return hidden, output
		elif self.task == 'RNNPred':
			output = self.out_pred(hidden)
			return hidden, output
		elif self.task == 'RNNAuto':
			latent = output[-1]
			output = self.out_auto(latent, delay=0)
			return hidden, latent, output
		else:
			raise ValueError(f"Invalid task: {self.task}")

	def save(self, filename):
		torch.save(self.state_dict(), filename)

	def load(self, filename):
		self.load_state_dict(torch.load(filename, map_location=self.device))

	def __len__ (self):
		return len(self._modules.items())