'''
RNN network
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from colorstrings import colorstrings as cs
from collections import OrderedDict, namedtuple

# Use PyTorch's internal namedtuple for custom load_state_dict results
LoadStateDictResult = namedtuple('LoadStateDictResult', ['missing_keys', 'unexpected_keys'])

############################################################################

def freeze(module):
    for name, pars in module.named_parameters():
        if pars is not None:
            pars.requires_grad = False

def print_parameters (llr):
    sd = llr.state_dict()
    for i, (n, p) in enumerate(sd.items()):
        print(cs.RED + f"{n}" + cs.END)
        print(80*"-")
        print(p)
        if i < len(sd) - 1:
            print(80*"=")
    return

def print_parameters_comp (source, target):

    src_keys = source.keys()
    tgt_keys = target.keys()
    common_keys = list(set(src_keys).intersection(set(tgt_keys)))
    tgt_missed_keys = list(set(src_keys) - set(tgt_keys))
    src_missed_keys = list(set(tgt_keys) - set(src_keys))
    if len(common_keys):
        print(cs.BOLD + cs.RED, "Common keys", cs.END)
        for i, n in enumerate(common_keys):
            print(cs.BOLD + cs.CYAN + f"{n}" + cs.END)
            print(80*"-")
            print(cs.BOLD + f"source[{n}]" + cs.END)
            print(source[n])
            print(cs.BOLD + f"target[{n}]" + cs.END)
            print(target[n])
            if i < len(common_keys) - 1:
                print(80*"=")
    if len(tgt_missed_keys):
        print(cs.BOLD + cs.RED, "Keys missing in target", cs.END)
        for i, n in enumerate(tgt_missed_keys):
            print(cs.BOLD + cs.CYAN + f"{n}" + cs.END)
            print(80*"-")
            print(cs.BOLD + f"source[{n}]" + cs.END)
            print(source[n])
            if i < len(tgt_missed_keys) - 1:
                print(80*"=")
    if len(src_missed_keys):
        print(cs.BOLD + cs.RED, "Keys missing in source", cs.END)
        for i, n in enumerate(tgt_missed_keys):
            print(cs.BOLD + cs.CYAN + f"{n}" + cs.END)
            print(80*"-")
            print(cs.BOLD + f"target[{n}]" + cs.END)
            print(target[n])
            if i < len(src_missed_keys) - 1:
                print(80*"=")
    return

def state_dicts_equal (source, target):
    '''
    \"Equal\" is not quite an equality.

    We check that for the parameters that are not None in `target`
    the values are actually the same as in the `source`.
    '''
    src_keys = source.keys()
    tgt_keys = target.keys()
    common_keys = list(set(src_keys).intersection(set(tgt_keys)))

    for k in common_keys:
        v1 = source[k]
        v2 = target[k]
        if isinstance(v2, torch.Tensor):
            if (isinstance(v1, torch.Tensor) and not torch.equal(v1, v2)) or (v1 is None) :
                return False
    return True



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

class FullRankLinear(nn.Linear):
    max_rank = None

class LowRankLinear(FullRankLinear):
    def __init__(self, in_features, out_features, bias=True, max_rank=None, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        self.max_rank = max_rank

        if max_rank is not None and max_rank < min(in_features, out_features):
            self.low_rank = True
            del self.weight
            if self.bias is not None:
                del self.bias
            self.U = nn.Parameter(torch.empty(max_rank, in_features, device=device, dtype=dtype))
            self.V = nn.Parameter(torch.empty(out_features, max_rank, device=device, dtype=dtype))
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            self.reset_parameters()
        else:
            self.low_rank = False
            self.U = None
            self.V = None

        for p in self.parameters():
            if p is not None:
                p.requires_grad_(True)

    def reset_parameters(self):
        if self.max_rank is None or self.max_rank >= min(self.in_features, self.out_features):
            super().reset_parameters()
        else:
            k = 1 / np.sqrt(self.in_features)
            nn.init.uniform_(self.U, -k, k)
            nn.init.uniform_(self.V, -k, k)
            if self.bias is not None:
                nn.init.uniform_(self.bias, -k, k)

    @property
    def weight(self):
        if self.max_rank is None or self.max_rank >= min(self.in_features, self.out_features):
            return super().weight
        else:
            return self.V @ self.U

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)

    def _extra_state_keys(self):
        return ['max_rank']

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if self.low_rank:
            sd['weight'] = self.weight.detach()
            sd['U'] = self.U.detach()
            sd['V'] = self.V.detach()
        return sd

    def load_state_dict(self, state_dict, strict=True):
        src_has_lowrank = 'U' in state_dict and 'V' in state_dict
        if src_has_lowrank:
            src_rank = state_dict['U'].shape[0]
            if self.max_rank is None:
                # low-rank -> full-rank
                weight = state_dict.get('weight', state_dict['V'] @ state_dict['U'])
                self.weight.data.copy_(weight)
                if 'bias' in state_dict and self.bias is not None:
                    self.bias.data.copy_(state_dict['bias'])
            else:
                # low-rank -> low-rank
                if self.max_rank != src_rank:
                    raise ValueError("Cannot load low-rank with different max_rank.")
                self.U.data.copy_(state_dict['U'])
                self.V.data.copy_(state_dict['V'])
                if 'bias' in state_dict and self.bias is not None:
                    self.bias.data.copy_(state_dict['bias'])
        else:
            # full-rank -> full-rank only
            if self.max_rank is not None:
                raise ValueError("Cannot load full-rank into low-rank module.")
            super().load_state_dict(state_dict, strict)


class RNN (nn.Module):

    def __init__(self, d_input, d_hidden, num_layers, d_output,
            output_activation=None, # choose btw softmax for classification vs linear for regression tasks
            drop_l=None,
            nonlinearity='relu',
            layer_type=LowRankLinear,
            max_rank=None,
            init_weights=None,
            model_filename=None, # file with model parameters
            to_freeze = [], # parameters to keep frozen; list with elements in ['i2h', 'h2h', 'h2o']
            from_file = [], # parameters to set from file; list with elements in ['i2h', 'h2h', 'h2o']
            bias=True,
            device="cpu",
            train_i2h = True,
            sim_id = None,
        ):

        super(RNN, self).__init__()
        init=init_weights
        print('sim_id', sim_id)
        self._model_filename=model_filename

        self.device = device
        # Defining the number of layers and the nodes in each layer
        self.d_input = d_input
        self.d_output = d_output
        self.d_hidden = d_hidden

        self.i2h = nn.Linear(d_input, d_hidden, bias=0)

        if layer_type == LowRankLinear:
            self.h2h = LowRankLinear(d_hidden, d_hidden, max_rank=max_rank, bias=bias)
        else:
            self.h2h = layer_type(d_hidden, d_hidden, bias=bias)

        self.h2o = nn.Linear(d_hidden, d_output, bias=bias)

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

        self.initialize_weights(init, seed=sim_id)

        if len(from_file):
            self.load_state_dict(torch.load(self._model_filename), modules_to_load=from_file)

        if 'i2h' in to_freeze:
            freeze(self.i2h)
        if 'h2h' in to_freeze:
            freeze(self.h2h)
        if 'h2o' in to_freeze:
            freeze(self.h2o)

        # print("print_parameters(self.i2h)")
        # print_parameters(self.i2h)
        # print("print_parameters(self.h2h)")
        # print_parameters(self.h2h)
        # print("print_parameters(self.h2o)")
        # print_parameters(self.h2o)

    def load_state_dict(self, state_dict, strict=True, modules_to_load=None):
        """
        Custom load_state_dict that optionally filters the keys to load only a subset of sub-modules.
        
        :param state_dict: The state dictionary to load.
        :param strict: Whether to check if all keys in state_dict match all keys in the module.
        :param modules_to_load: A list of module names (strings, e.g., ['layer1']) to load. 
                                If None, all modules are loaded (default behavior).
        """
        if modules_to_load is not None:
            # Build a set of prefixes (e.g., 'layer1.', 'layer2.')
            prefixes = tuple(f'{name}.' for name in modules_to_load)
            
            # Filter the incoming state_dict
            filtered_state_dict = {}
            missing_keys_due_to_filter = []
            
            for k, v in state_dict.items():
                if k.startswith(prefixes):
                    filtered_state_dict[k] = v
                else:
                    # Keep track of keys we ignored because they weren't in the list
                    missing_keys_due_to_filter.append(k)

            print(f"\n[Parent Load] Loading only modules: {modules_to_load}. Filtered keys: {list(filtered_state_dict.keys())}")
            
            # Load the filtered state dict using the standard recursive process
            # We set strict=False here because we have intentionally removed keys
            result = super().load_state_dict(filtered_state_dict, strict=False)

            # The keys we removed are now 'missing' from the load operation.
            # We must re-add them to the 'unexpected' list of the final result, 
            # as they were present in the input but intentionally ignored.
            final_unexpected_keys = result.unexpected_keys + missing_keys_due_to_filter
            return LoadStateDictResult(result.missing_keys, final_unexpected_keys)

        else:
            # Default behavior: load everything
            return super().load_state_dict(state_dict, strict=strict)


    def initialize_weights(self, init, seed=None):
        print('init', init)

        if seed is not None:
            torch.manual_seed(1990+seed)

        if init == None:
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
        
        if init is not None:
            if "weight" in name:
                f_in = 1.*pars.data.size()[1]
                std = init_f(f_in)
                pars.data.normal_(0., std)

    def __hidden_update (self, h, x):
        '''
        A single iteration of the RNN function
        h: hidden activity
        x: input
        '''
        zi = self.i2h (x)
        zh = self.h2h (h)
        return self.phi (zh + zi)

    def step (self, h, x):
        return self.__hidden_update(h,x)

    def forward(self, x, mask=None, delay=0):
        '''
        x
        ---
        seq_length, batch_size, d_input
        or
        seq_length, d_input

        mask: torch.Tensor of bools
            True for active neurons, False for ablated neurons
        
        delay: int
            Number of time steps to allow after the input
        '''

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
            # zi = self.i2h (xt)
            # zh = self.h2h (ht)
            # z = self.phi (zh + zi)
            z = self.__hidden_update(ht, xt)
            z = _masking(z)

            hidden.append(z)
            ht = z

        # print('before', np.shape(hidden))
        hidden = torch.reshape(torch.stack(hidden), _shape)
        # print('after', np.shape(hidden))

        output = self.h2o(hidden)
        # print('outputshape', np.shape(output))

        return hidden, output

    def jacobian (self, h, x, mask=None): # remember to change when ablating
        '''
        Returns the Jacobian of the RNN, evaluated at the hidden activity `h`
        and for instantaneous input `x`
        '''
        _h = h.clone().detach().requires_grad_(True)
        _x = x.clone().detach().requires_grad_(False)
        _f = lambda h: self.__hidden_update(h, _x)
        _grad = torch.autograd.functional.jacobian(_f, _h)
        return _grad

    # def get_activity(self, x):

    #   with torch.no_grad():

    #       _x = x.clone().detach().to(self.device)

    #       ht, hT, _  = self.forward(_x)

    #       # print('ht', np.shape(ht)) # activity for whole sequence
    #       # print('hT', np.shape(hT)) # activity for last element of sequence

    #       y = ht.permute(1,0,2) # y is of size num_tokens (train/test) x L x N 

    #   return y


class RNNEncoder(nn.Module):
    def __init__(self, d_input, d_hidden, num_layers, d_latent, nonlinearity, device,
            model_filename, from_file, to_freeze, init_weights, layer_type):
        super(RNNEncoder, self).__init__()
        
        # RNN: d_input -> d_latent
        self.rnn = RNN(d_input, d_hidden, num_layers, d_latent, nonlinearity=nonlinearity, device=device,
            model_filename=model_filename, from_file=from_file,
            to_freeze=to_freeze, init_weights=init_weights, layer_type=layer_type)

    def forward(self, x, delay=0):
        # x: (sequence_length, batch_size, d_input) 
        # output: (sequence_length, batch_size, d_hidden)
        # h: (num_layers, batch_size, d_hidden)
        rnn_out, latent = self.rnn(x, delay=delay)
        # return activity of latent layer at end of sequence
        # latent = F.relu(latent)
        return rnn_out, latent[-1]

class RNNDecoder(nn.Module):
    def __init__(self, d_latent, d_hidden, num_layers, d_input, nonlinearity, device,
            init_weights, layer_type, sequence_length):
        super(RNNDecoder, self).__init__()
        # RNN: d_latent -> d_input

        self.rnn = RNN(d_latent, d_hidden, num_layers, d_input, nonlinearity=nonlinearity, device=device,
            init_weights=init_weights, layer_type=layer_type)
        
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
    def __init__(self, d_input, d_hidden, num_layers, d_latent, sequence_length,
                        nonlinearity='relu',
                        device="cpu",
                        model_filename=None, # file with model parameters
                        to_freeze = [], # parameters to keep frozen; list with elements in ['i2h', 'h2h', 'h2o']
                        from_file = [], # parameters to set from file; list with elements in ['i2h', 'h2h', 'h2o']
                        init_weights=None,
                        layer_type=nn.Linear,
                ):
        super(RNNAutoencoder, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.d_latent = d_latent
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.device = device

        self.encoder = RNNEncoder(d_input, d_hidden, num_layers, d_latent, nonlinearity, device,
            model_filename, from_file, to_freeze, init_weights, layer_type)

        self.decoder = RNNDecoder(d_latent, d_hidden, num_layers, d_input, nonlinearity, device,
            init_weights=init_weights, layer_type=layer_type, sequence_length=sequence_length)

    @property
    def h2h (self):
        return self.encoder.rnn.h2h

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



if __name__ == "__main__":

    in_features = 6
    out_features = 4


    ### Tests
    
    ## Full-rank to full-rank (compatible)

    # 1. Low(None) -> Low(None)
    test_txt = "TESTING: Low(None) -> Low(None)"
    print(cs.BOLD + f"\n\n1. {test_txt}" + cs.END)
    old = LowRankLinear(in_features, out_features, max_rank=None, bias=True)
    old_state_dict = old.state_dict()
    new = LowRankLinear(in_features, out_features, max_rank=None, bias=True)
    new.load_state_dict(old_state_dict)
    print_parameters_comp(old.state_dict(), new.state_dict())
    if not state_dicts_equal(old_state_dict, new.state_dict()):
        raise ValueError(cs.RED + f"[FAILED] {test_txt}" + cs.END)
    else:
        print(cs.GREEN + f"[PASSED] {test_txt}" + cs.END)
    
    ## Low-rank to full-rank (compatible) -- the weight matrix can be loaded

    # 2. Low(int) -> Low(None)
    test_txt = "TESTING: Low(int) -> Low(None)"
    print(cs.BOLD + f"\n\n2. {test_txt}" + cs.END)
    old = LowRankLinear(in_features, out_features, max_rank=2, bias=True)
    old_state_dict = old.state_dict()
    new = LowRankLinear(in_features, out_features, max_rank=None, bias=True)
    new.load_state_dict(old_state_dict)
    print_parameters_comp(old.state_dict(), new.state_dict())
    if not state_dicts_equal(old_state_dict, new.state_dict()):
        raise ValueError(cs.RED + f"[FAILED] {test_txt}" + cs.END)
    else:
        print(cs.GREEN + f"[PASSED] {test_txt}" + cs.END)

    ## Full-rank to low-rank (incompatible) -- error should be raised

    # 3. Low(None) -> Low(int)
    test_txt = "TESTING: Low(None) -> Low(int)"
    print(cs.BOLD + f"\n\n3. {test_txt}" + cs.END)
    old = LowRankLinear(in_features, out_features, max_rank=None, bias=True)
    old_state_dict = old.state_dict()
    new = LowRankLinear(in_features, out_features, max_rank=2, bias=True)
    passed = False
    try:
        new.load_state_dict(old_state_dict)
    except Exception as e:
        print(f"\"{type(e).__name__}\" exception raised: {e}")
        passed = True
    if not passed:
        raise ValueError(cs.RED + f"[FAILED] {test_txt}" + cs.END)
    else:
        print(cs.GREEN + f"[PASSED] {test_txt}" + cs.END)

    ## Low-rank to low-rank (compatible) -- loading U/V should work

    # 4. Low(int) -> Low(int)
    test_txt = "TESTING: Low(int) -> Low(int)"
    print(cs.BOLD + f"\n\n4. {test_txt}" + cs.END)
    old = LowRankLinear(in_features, out_features, max_rank=2, bias=True)
    old_state_dict = old.state_dict()
    new = LowRankLinear(in_features, out_features, max_rank=2, bias=True)
    new.load_state_dict(old_state_dict)
    print_parameters_comp(old.state_dict(), new.state_dict())
    if not state_dicts_equal(old_state_dict, new.state_dict()):
        raise ValueError(cs.RED + f"[FAILED] {test_txt}" + cs.END)
    else:
        print(cs.GREEN + f"[PASSED] {test_txt}" + cs.END)
    
    ## Low-rank to low-rank (incompatible) -- if different max_rank, error should be raised

    # 5. Low(int) -> Low(int2)
    test_txt = "TESTING: Low(int) -> Low(int2)"
    print(cs.BOLD + f"\n\n5. {test_txt}" + cs.END)
    old = LowRankLinear(in_features, out_features, max_rank=2, bias=True)
    old_state_dict = old.state_dict()
    new = LowRankLinear(in_features, out_features, max_rank=3, bias=True)
    passed = False
    try:
        new.load_state_dict(old_state_dict)
    except Exception as e:
        print(f"\"{type(e).__name__}\" exception raised: {e}")
        passed = True
    if not passed:
        raise ValueError(cs.RED + f"[FAILED] {test_txt}" + cs.END)
    else:
        print(cs.GREEN + f"[PASSED] {test_txt}" + cs.END)
