import numpy as np
import scipy
from numpy import savetxt
import torch
import os
import sys
from os.path import join
from glob import glob
# import par_transfer as par
import par
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# from make_data import make_data
from sklearn.decomposition import PCA
from collections import defaultdict
from colorstrings import CYAN_STR, RED_STR, END_STR
from lowrank_utils import * #plot_mean_loss, align_singular_vectors, plot_SVD_results, plot_SVDs_dimensionality_all, plot_recurrent_weights, plot_all_weights, dimension, compute_umcontent, compute_cosine_similarity, compute_distance, to_dict


folder_name = join(f'Task{par.task}_N{par.n_hidden}_nlatent{par.n_latent}_L{par.L}_m{par.m}_alpha{par.alpha}'+\
                   f'_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}'+\
                   # f'_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}_'+\
                   f'_init{par.init}_transfer{par.transfer}_cuesize{par.cue_size}_delay{par.delay}_datasplit{par.datasplit}'
                   
                  )

DATA_DIR = join(par.folder, folder_name)
print("Data dir: ", DATA_DIR)


classcomb_path = join(par.folder, folder_name, 'classes.pkl')
with open(classcomb_path, 'rb') as handle:
    classcomb = pickle.load(handle)

types_set = list(classcomb[par.classcomb])
token_to_type_path = join(par.folder, folder_name, f'token_to_type_classcomb{par.classcomb}.pkl')
with open(token_to_type_path, 'rb') as handle:
    token_to_type = pickle.load(handle)

token_to_set_path = join(par.folder, folder_name, f'token_to_set_classcomb{par.classcomb}.pkl')
with open(token_to_set_path, 'rb') as handle:
    token_to_set = pickle.load(handle)


# _epoch_time_ids, _epoch = (4,-1), 40    # 4th saved epoch, last timestep in sequence
_epoch_time_ids, _epoch = (-1,-1), 25   # last saved epoch, last timestep in sequence

def get_all_sims_paths (folder_name):
    """
    Get all simulation paths for a given folder name.
    """
    model_paths = sorted(glob(join(DATA_DIR, f'model_state_classcomb{par.classcomb}_sim*.pth')))
    # model_paths = sorted(glob(join(DATA_DIR, f'model_state_classcomb{par.classcomb}_sim*.pth')))
    results_paths = sorted(glob(join(DATA_DIR, f'results_task{par.task}_classcomb{par.classcomb}_sim*.pkl')))
    return model_paths, results_paths


model_paths, results_paths = get_all_sims_paths(folder_name)

# for i, path in enumerate(model_paths):
#     print(f"{str(i):<5}{path}")
# for i, path in enumerate(results_paths):
#     print(f"{str(i):<5}{path}")
mean_model_path = join(DATA_DIR, f'model_state_classcomb{par.classcomb}_mean.pth')


print('Load all results')
all_models = [torch.load(path, map_location="cpu") for path in model_paths]
all_results = [pickle.load(open(path, 'rb')) for path in results_paths]

assert len(all_results) and len(all_models), "Empty results..."
assert len(all_results) == len(all_models), f"Length mismatch betwen results and models, {len(all_results)} vs {len(all_models)}"

FIGS_DIR = join(par.folder, 'figs', folder_name)
# FIGS_DIR = join(par.folder, 'figs', folder_name, f"classcomb_{par.classcomb}", f"{par.whichset}", f"svg_{par.common_rotation}")
print("Figs dir: ", FIGS_DIR)
os.makedirs(FIGS_DIR, exist_ok=True)

plot_mean_loss(all_results, FIGS_DIR)


# 0. Extract some info about results -- all lists indexed by simulation
print(f'Extract activity -- only sequences in {par.whichset} set')
all_sequences = [np.array([token_to_type[k] for k in res['HiddenAct'].keys() if token_to_set[k] == par.whichset]) #np.array(list(res['HiddenAct'].keys()))
                for res in all_results] # classes (n_seq,)
assert np.all([all_sequences[0] == s for s in all_sequences]), \
       "All sequences must be in the same order across simulations"
all_classes = [np.array([token_to_type[k] for k in res['HiddenAct'].keys() if token_to_set[k] == par.whichset]) for res in all_results] # classes (n_seq,)
all_hs = [np.array([list(d.values()) for k, d in res['HiddenAct'].items() if token_to_set[k] == par.whichset]).transpose((1,2,0,3)) for res in all_results]    # hidden act (n_epochs_saved, L, n_seq, n_hidden,) -- convenient for SVD

print(f'Extract weights')
# extract all weights
all_input_weights = [m['i2h.weight'].numpy() for m in all_models]
all_output_weights = [m['h2o.weight'].numpy() for m in all_models]
all_rec_weights = [m['h2h.weight'].numpy() for m in all_models]

# extract all biases (only those in the recurrent layer are supposed to be non-vanishing)
all_rec_biases = [m['h2h.bias'].numpy() for m in all_models]
try:
    all_out_biases = [m['h2o.bias'].numpy() for m in all_models]
except KeyError as e:
    out_bias = False
    print(f"{type(e).__name__}: {e}.\nNo output bias in these simulations")

print('Perform NEW SVDs')
# SVD of recurrent weights
all_rec_SVDs = [np.linalg.svd(W) for W in all_rec_weights]
all_rec_Us = [U for U, _, _ in all_rec_SVDs]
all_rec_Ss = [S for _, S, _ in all_rec_SVDs]
all_rec_Vhs = [Vh for _, _, Vh in all_rec_SVDs]

# SVD at final time step
all_final_hs = [h[_epoch_time_ids] for h in all_hs]   # hidden act after training at last time step, at the 4th saved snapshot, (n_sim, n_seq, n_hidden,)
# all_final_hs = [h[-1,-1] for h in all_hs]   # hidden act after training at last time step, at the end of training, (n_sim, n_seq, n_hidden,)
# all_final_hs = [h[-1,-1] - np.mean(h[-1,-1], axis=0)[None,:] for h in all_hs]   # hidden act after training at last time step, (n_seq, n_hidden,) minus mean
_SVDs = [np.linalg.svd(h) for h in all_final_hs]
all_final_VEs = [np.cumsum(S**2)/np.sum(S**2) for _, S, _ in _SVDs]   # fraction of variance explained by first principal components
all_final_Vhs = [Vh for _, _, Vh in _SVDs]    # principal vectors, component (1st index) by feature (2nd index)

# define rotation matrices
if par.common_rotation == 'weights':
    _all_Rs = all_rec_Vhs
elif par.common_rotation == 'hidden':
    _all_Rs = all_final_Vhs
else:
    raise NotImplementedError(f'Unknown option "{par.common_rotation}" for `common_rotation`.')

# array containing reference for all simulations,
# the first sequence of first simulation is the reference
all_refs = [hs[0] for hs in all_final_hs] # (n_sim, n_hidden)

# find the orthogonal transformation that rotates all the final hs to be aligned with each other
print(f"Rotations using {par.common_rotation} SVs")

all_rec_Vhs_aligned = align_singular_vectors(all_rec_Vhs, all_refs, n_components=10)
all_Rs = align_singular_vectors(_all_Rs, all_refs, n_components=10)

# Transforming the weights in each simulation using Vh brings them all close
all_output_weights_proj = [W @ R.T for R, W in zip(all_Rs, all_output_weights)] # (C, N) x (N, N)
all_input_weights_proj = [R @ W for R, W in zip(all_Rs, all_input_weights)]     # (N, N) x (N, a)
all_rec_weights_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_rec_weights)]   # (N, N) x (N, N) x (N, N)
all_rec_biases_proj = [R @ b for R, b in zip(all_Rs, all_rec_biases)]

make_save_mean_model(all_input_weights_proj, all_rec_weights_proj, all_output_weights_proj, all_rec_biases_proj, all_Rs, all_rec_Us, all_rec_Vhs, all_input_weights, all_output_weights, all_rec_biases, all_out_biases, DATA_DIR)

############### ANALYSIS OF HIDDEN-LAYER ACTIVITY ###############################################

# projections along principal components, i.e. rotated data (n_seq, n_epochs_saved, L, n_hidden,)
print('Plot SVD results')
all_hs_proj = [np.dot(h, R.T) for h, R in zip(all_hs, all_Rs)]
all_final_hs_proj = [h[_epoch_time_ids] for h in all_hs_proj]

plot_SVD_results(all_final_VEs, all_final_hs_proj, all_Rs, FIGS_DIR, n_components=6, cmap='bwr')

# Dimensionality reduces over time for any given class
print('Compute and plot dimensionality over time')
# separate the hidden-layer activity by class, at the end of learning and for all times in the sequence
# (store in a dictionary with class as key)
unique_classes = np.unique(all_classes[0])

all_hs_class = [{c: np.take(h, np.where(classes==c)[0], axis=2)[-1] for c in unique_classes} for h, classes in zip(all_hs, all_classes)]

plot_SVDs_dimensionality_all(all_hs_class, all_Rs, all_hs, types_set, FIGS_DIR)

#############################################################   ANALYSIS OF WEIGHTS #############################################################

print('Plot weights -- their SVDs')

# # Transforming the weights in each simulation using Vh brings them all close
# all_output_weights_proj = [W @ R.T for R, W in zip(all_Rs, all_output_weights)] # (C, N) x (N, N)
# all_input_weights_proj = [R @ W for R, W in zip(all_Rs, all_input_weights)]     # (N, N) x (N, a)
# all_rec_weights_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_rec_weights)]   # (N, N) x (N, N) x (N, N)
# all_rec_biases_proj = [R @ b for R, b in zip(all_Rs, all_rec_biases)]

plot_SVDs_weights(all_rec_weights, all_rec_weights_proj, FIGS_DIR, label='Recurrent')

plot_SVDs_weights(all_input_weights, all_input_weights_proj, FIGS_DIR, label='Input')

plot_SVDs_weights(all_output_weights, all_output_weights_proj, FIGS_DIR, label='Output')

plot_recurrent_weights(all_rec_weights, all_rec_weights_proj, FIGS_DIR)

print('Plot recurrent weights')

print('Plot all weights')

plot_all_weights(all_input_weights_proj, all_rec_weights_proj, all_output_weights_proj, FIGS_DIR)


#
#   ALIGNMENT ANALYSIS
#
# We want to check, at each time, and for each class, the **alignment** of the activity
# with the modes of the recurrent weights (cosine similarity between principal vectors)
# This is done after rotating activities and weights using the common transformation
# identified at the last time step.
print('Compute components of activity along weights SVs - mean over sequences by class')

n_components = 4
all_rec_Vhs_aligned = align_singular_vectors(all_rec_Vhs, all_refs, n_components=n_components)

print('... plotting')

hs_proj_class = defaultdict(list)
for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
    for c, hs in hs_class.items():
        hs_proj_class[c].append(np.mean(np.dot(hs, rec_Vh.T), axis=1))
hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sim, time, p, N)

# hs_proj_class = defaultdict(list)
# for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
#     for c, hs in hs_class.items():
#         hs_proj_class[c].append(np.dot(hs, rec_Vh.T))
# hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sims, time, seq, N)
# hs_proj_class = np.mean(hs_proj_class, axis=1) # (class, time, seq, N)    
# one plot for each time step

plot_alignment_weights_activity_class(hs_proj_class, types_set, FIGS_DIR)


plot_alignment_weights_activity_timestep(hs_proj_class, types_set, FIGS_DIR)

print('Compute components of activity along weights SVs - all simulations separately')


plot_alignment_activity_trans(hs_proj_class, types_set, FIGS_DIR)

hs_proj_class = defaultdict(list)
for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
    for c, hs in hs_class.items():
        hs_proj_class[c].append(np.dot(hs, rec_Vh.T))
hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sims, time, seq, N)

plot_alignment_weights_activity_times_allseq(hs_proj_class, types_set, FIGS_DIR)

print('Cosine similarity')

plot_cosine_similarity(hs_proj_class, all_classes, FIGS_DIR)

# plot_UC()


