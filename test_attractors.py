import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import os
from os.path import join
import pickle
import par
from make_data import *

if __name__ == "__main__":

	datasplit = 0
	c=0

	if par.epoch == -1:
		epoch_= np.arange(0, int(par.n_epochs)+1, par.snap_freq)[-1]
	else:
		epoch_= par.epoch

	folder_name = f'Task{par.task}_N{par.n_hidden}_nlatent{par.n_latent}_L{par.L}_m{par.m}_alpha{par.alpha}_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}_init{par.init}_transfer{par.transfer}_cuesize{par.cue_size}_delay{par.delay}_datasplit{datasplit}_CT'
	print(folder_name)

	if par.task != 'RNNMulti':
		task = par.task
	else:
		task = 'RNNClass' # choose btw 'RNNPred', 'RNNClass', RNNAuto'
	
	if os.path.isfile(join(par.folder, folder_name, f'results_task{task}_classcomb{c}.pkl')):
		with open(join(par.folder, folder_name, f'results_task{task}_classcomb{c}.pkl'), 'rb') as handle:
			results=pickle.load(handle)

	if os.path.isfile(join(par.folder, folder_name, f'token_to_set_classcomb{c}.pkl')):
		with open(join(par.folder, folder_name, f'token_to_set_classcomb{c}.pkl'), 'rb') as handle:
			token_to_set=pickle.load(handle)

	if os.path.isfile(join(par.folder, folder_name, f'token_to_type_classcomb{c}.pkl')):
		with open(join(par.folder, folder_name, f'token_to_type_classcomb{c}.pkl'), 'rb') as handle:
			token_to_type=pickle.load(handle)
	# print(token_to_type)
	epoch_snapshots = np.arange(0, int(par.n_epochs)+1, par.snap_freq)

	all_tokens = token_to_set.keys()
	hidden = np.zeros((len(all_tokens),
					  len(epoch_snapshots),
					  par.L + par.cue_size, 
					  par.n_hidden))

	for token_id, token in enumerate(all_tokens):
		hidden[token_id, :, :, :] = np.array([list(results['HiddenAct'][token][epoch][0]) for epoch in epoch_snapshots ])
	

	fig, ax = plt.subplots(figsize=[18, 3])

	# for which_timestep in range(par.L+par.cue_size):
	for neuron in range(par.n_hidden):
		ax.plot(np.mean(hidden[1, -100:, :, neuron], axis=0))
	# ax.set_xticks(epoch_snapshots[::20])
	# ax.set_xticklabels(epoch_snapshots[::20])
	ax.set_xlabel('Time')
	ax.set_ylabel('Activity')
	fig.tight_layout()
	fig.savefig(join(par.folder, f'figs/test%s.svg'%folder_name), bbox_inches="tight")
	exit()

	attractor_results = {}
	tolerance=1e-5
	for token_id, (token, tokenclass) in enumerate(token_to_type.items()):
		hidden_laststep = hidden[token_id, :, which_timestep,:]
		for step in range(len(epoch_snapshots)): 
			# print('step', step)

			if step >= 2:  # Only start checking from step 2 onwards
				diff = np.linalg.norm(hidden_laststep[step-1, :] - hidden_laststep[step-2, :])
				if diff < tolerance:  # If change is small, assume fixed point attractor
					attractor_results[tokenclass] = "Fixed Point Attractor"
					break
		# Check for limit cycle if no fixed point found
		for past_idx in range(len(epoch_snapshots) - 2):
			cycle_length = len(epoch_snapshots) - past_idx - 1
			cycle_diff = np.linalg.norm(hidden_laststep[-1,:] - hidden_laststep[past_idx,:])
			if cycle_diff < tolerance:
				attractor_results[tokenclass] = f"Limit Cycle (Period {cycle_length})"
				break
		else:
			attractor_results[tokenclass] = "No Convergence (Possible Chaos)"

	print(attractor_results)