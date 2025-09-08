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
from analysis_utils import dimension, compute_umcontent, compute_cosine_similarity, compute_distance, to_dict



datasplit=0

data_opts = []
for transfer_id in range(7):
    for c in range(5):
        data_opts.append((transfer_id, c))

# data_id = 0
# # data_id = int(sys.argv[1])-1
# transfer_id, c = data_opts[data_id]

# transfer_id = int(sys.argv[1])
c = 2

folder_name = join(f'Task{par.task}_N{par.n_hidden}_nlatent{par.n_latent}_L{par.L}_m{par.m}_alpha{par.alpha}'+\
                   f'_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}'+\
                   # f'_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}_'+\
                   f'_init{par.init}_transfer{par.transfer}_cuesize{par.cue_size}_delay{par.delay}_datasplit{datasplit}_0'
                   
                   # '_transfer_mean'
                   #f'_noise{par.train_noise:.3f}'
                   # f'_{transfer_id}'
                   
                  )

data_dir = join(par.folder, folder_name)
print("Data dir: ", data_dir)

# if input("Continue? (Y/n) ").lower() in ['no', 'n']:
#     exit()


classcomb_path = join(par.folder, folder_name, 'classes.pkl')
with open(classcomb_path, 'rb') as handle:
    classcomb = pickle.load(handle)

types_set = list(classcomb[c])
token_to_type_path = join(par.folder, folder_name, f'token_to_type_classcomb{c}.pkl')
with open(token_to_type_path, 'rb') as handle:
    token_to_type = pickle.load(handle)

token_to_set_path = join(par.folder, folder_name, f'token_to_set_classcomb{c}.pkl')
with open(token_to_set_path, 'rb') as handle:
    token_to_set = pickle.load(handle)


# _epoch_time_ids, _epoch = (4,-1), 40    # 4th saved epoch, last timestep in sequence
_epoch_time_ids, _epoch = (-1,-1), 25   # last saved epoch, last timestep in sequence

def get_all_sims_paths (folder_name):
    """
    Get all simulation paths for a given folder name.
    """
    model_paths = sorted(glob(join(data_dir, f'model_state_classcomb{c}_sim*.pth')))
    # model_paths = sorted(glob(join(data_dir, f'model_state_classcomb{c}_sim*.pth')))
    results_paths = sorted(glob(join(data_dir, f'results_task{par.task}_classcomb{c}_sim*.pkl')))
    return model_paths, results_paths


model_paths, results_paths = get_all_sims_paths(folder_name)

for i, path in enumerate(model_paths):
    print(f"{str(i):<5}{path}")
for i, path in enumerate(results_paths):
    print(f"{str(i):<5}{path}")
mean_model_path = join(data_dir, f'model_state_classcomb{c}_mean.pth')

# if input("Continue? (Y/n) ").lower() in ['no', 'n']:
#     exit()

print('Load all results')
all_models = [torch.load(path, map_location="cpu") for path in model_paths]
all_results = [pickle.load(open(path, 'rb')) for path in results_paths]

assert len(all_results) and len(all_models), "Empty results..."
assert len(all_results) == len(all_models), f"Length mismatch betwen results and models, {len(all_results)} vs {len(all_models)}"

figs_ext = 'svg'
figs_dir = join(par.folder, 'figs', folder_name)
# figs_dir = join(par.folder, 'figs', folder_name, f"classcomb_{c}", f"{par.whichset}", f"{figs_ext}_{par.common_rotation}")
print("Figs dir: ", figs_dir)
os.makedirs(figs_dir, exist_ok=True)


# all_losses = [np.array([[l for _, l in loss.items()] for _, loss in res['Loss'].items()]) for res in all_results]
# print('Plot loss')
# fig, ax = plt.subplots()
# for loss in all_losses:
#     ax.plot(np.mean(loss, axis=0), lw=.2, c='k', alpha=.2)
# ax.plot(np.mean(np.mean(all_losses,axis=0),axis=0), lw=2, c='r', ls='--')
# fig.savefig(join(figs_dir, f'loss_training.{figs_ext}'), dpi=300)
# plt.close(fig)


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
out_bias = True
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

def align_singular_vectors(all_Rs, all_refs, n_components=2):
    all_Rs_aligned = []
    for R, r in zip(all_Rs, all_refs):
        _R = R.copy()
        for n, v in enumerate(R[:n_components]):
            if np.dot(v, r) < 0:
                _R[n] = - v
            else:
                _R[n] = v
        all_Rs_aligned.append(_R)
    return all_Rs_aligned

# find the orthogonal transformation that rotates all the final hs to be aligned with each other
print(f"Rotations using {par.common_rotation} SVs")

all_rec_Vhs_aligned = align_singular_vectors(all_rec_Vhs, all_refs, n_components=10)
all_Rs = align_singular_vectors(_all_Rs, all_refs, n_components=10)

# Transforming the weights in each simulation using Vh brings them all close
all_output_weights_proj = [W @ R.T for R, W in zip(all_Rs, all_output_weights)] # (C, N) x (N, N)
all_input_weights_proj = [R @ W for R, W in zip(all_Rs, all_input_weights)]     # (N, N) x (N, a)
all_rec_weights_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_rec_weights)]   # (N, N) x (N, N) x (N, N)
all_rec_biases_proj = [R @ b for R, b in zip(all_Rs, all_rec_biases)]

# Compute average transformed weights across simulations
mean_all_output_weights_proj = np.mean(all_output_weights_proj, axis=0)
mean_all_input_weights_proj = np.mean(all_input_weights_proj, axis=0)
mean_all_rec_weights_proj = np.mean(all_rec_weights_proj, axis=0)
mean_all_rec_biases_proj = np.mean(all_rec_biases_proj, axis=0)

# # Recreating a "filtered" version of the weights by rotating "back" the average low-rank
# all_output_weights_filtered = [mean_all_output_weights_proj @ R for R in all_Rs]
# all_input_weights_filtered = [R.T @ mean_all_input_weights_proj for R in all_Rs]
# all_rec_weights_filtered = [R.T @ mean_all_rec_weights_proj @ R for R in all_Rs]
# all_rec_biases_filtered = [R.T @ mean_all_rec_biases_proj for R in all_Rs]

# Recreate all recurrent weight matrices with the average spectrum (show that mean can support task)
# 1. Define the diagonal matrix with the spectrum of the average (rotated) weights
S_bar = np.linalg.svd(mean_all_rec_weights_proj, compute_uv=False) * np.eye(len(mean_all_rec_weights_proj))
# print(S_bar[:10,:10])
# 2. Re-compose the (filtered) weight matrices by multiplying by the left and right singular vectors
all_rec_weights_filtered = [U @ S_bar @ Vh for U, Vh in zip(all_rec_Us, all_rec_Vhs)]

print("Saving transformed weights")

for sim_id, (W_out, W_in, W_rec, b_rec) in enumerate(zip(
        # all_output_weights_filtered, all_input_weights_filtered, all_rec_weights_filtered, all_rec_biases_filtered
        all_output_weights, all_input_weights, all_rec_weights_filtered, all_rec_biases
    )):
    model_dict = {}
    model_dict['i2h.weight'] = torch.tensor(W_in, dtype=torch.float)
    model_dict['h2o.weight'] = torch.tensor(W_out, dtype=torch.float)
    model_dict['h2h.weight'] = torch.tensor(W_rec, dtype=torch.float)
    model_dict['h2h.bias'] = torch.tensor(b_rec, dtype=torch.float)
    if out_bias:
        model_dict['h2o.bias'] = torch.tensor(all_out_biases[sim_id], dtype=torch.float)
    print(f"{str(sim_id):<5}: {', '.join(list(model_dict.keys()))}")
    torch.save(model_dict, join(data_dir, f'model_state_filtered_classcomb{c}_sim{sim_id}.pth'))
    # torch.save(model_dict, join(data_dir, f'model_state_filtered_classcomb{c}_sim{sim_id}_epoch{_epoch}.pth'))

exit()
# fig, axs = plt.subplots(2,2, figsize=(8,6))
# plt.tight_layout
# for i, (name, raw, mean, ax) in enumerate(zip(
#         ['W_out', 'W_in', 'W_rec', 'b_rec'],
#         [all_output_weights, all_input_weights, all_rec_weights, all_rec_biases],
#         [mean_all_output_weights_proj, mean_all_input_weights_proj, mean_all_rec_weights_proj, mean_all_rec_biases_proj],
#         axs.ravel()
#     )):
#     ax.set_title(name)
#     ax.hist(np.stack(raw).reshape(-1), color=f"C{i}", alpha=.3, density=True, bins=30)
#     ax.hist(mean.reshape(-1), color=f"C{i}", histtype='step', density=True, bins=30)

# fig.savefig("hist_weights.svg", dpi=300)

def plot_SVD_results (all_VEs, all_hs_proj, all_Vhs, n_components=2, **kwargs):

    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    # plot all singular values across simulations
    ax = axs[0]
    ax.set_title("Spectrum")
    for i, VE in enumerate(all_VEs):
        ax.plot(1.- VE, lw=.3, c='k', alpha=.2)
    ax.plot(1. - np.mean(all_VEs, axis=0), c='r', ls='--')
    ax.set_xlabel("Index")
    ax.set_ylabel("Fraction of (un)explained variance")
    ax.set_yscale('log')

    # plot all (left) singular vectors across simulations
    ax = axs[1]
    ax.set_title("Singular vectors")
    _X = np.vstack([Vh[:n_components].reshape(1,-1) for Vh in all_Vhs])
    _vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
    _vmin = - _vmax
    im = ax.imshow(_X,
                    aspect='auto', interpolation='nearest',
                    vmin=_vmin, vmax=_vmax,
                    **kwargs)
    
    for i in range(1, n_components):
        ax.axvline(i*all_Vhs[0].shape[1]-.5, c='k', ls='--')
    ax.set_xlabel(f"First {n_components} SV, all units")
    ax.set_ylabel("Simulation ID")
    fig.colorbar(im, ax=ax)
    fig.savefig(join(figs_dir, f'hidden_last_step_SVD.{figs_ext}'), dpi=300)
    plt.close(fig)

    # plot all rotated data across simulations
    fig, ax = plt.subplots()
    # plt.subplots_adjust(bottom=.1, top=.95, right=.95)
    ax.set_title("Coefficients of activity along singular vectors across simulations")
    ax.set_xlabel("Coefficient (all data points)")
    ax.set_ylabel("Simulation ID")
    for i in range(1, n_components):
        ax.axvline(i*len(all_hs_proj[0])-.5, c='k', ls='--')
    _X = np.stack([_h[:,:n_components].T.ravel() for _h in all_hs_proj])
    _vmax = 10 # np.max(np.fabs([np.max(_X), np.min(_X)]))
    _vmin = -10 # - _vmax
    im = ax.imshow(_X,
                    aspect='auto', interpolation='nearest',
                    vmin=_vmin, vmax=_vmax,
                    **kwargs)
    fig.colorbar(im, ax=ax)
    fig.savefig(join(figs_dir, f'hidden_last_step_coeffs.{figs_ext}'), dpi=300)
    plt.close(fig)


#####################################   ANALYSIS OF HIDDEN-LAYER ACTIVITY ###############################################

# projections along principal components, i.e. rotated data (n_seq, n_epochs_saved, L, n_hidden,)
print('Plot SVD results')
all_hs_proj = [np.dot(h, R.T) for h, R in zip(all_hs, all_Rs)]
all_final_hs_proj = [h[_epoch_time_ids] for h in all_hs_proj]
plot_SVD_results(all_final_VEs, all_final_hs_proj, all_Rs, n_components=6, cmap='bwr')


# Dimensionality reduces over time for any given class
print('Compute and plot dimensionality over time')
# separate the hidden-layer activity by class, at the end of learning and for all times in the sequence
# (store in a dictionary with class as key)
unique_classes = np.unique(all_classes[0])
all_hs_class = [{c: np.take(h, np.where(classes==c)[0], axis=2)[-1] for c in unique_classes} for h, classes in zip(all_hs, all_classes)]
all_SVDs_class = [{c: np.linalg.svd(h)[1:] for c, h in h_class.items()} for h_class in all_hs_class]
all_VEs_class = [{c: np.cumsum(SVDs[0]**2, axis=-1)/np.sum(SVDs[0]**2, axis=-1)[:,None] for c, SVDs in SVDs_class.items()} for SVDs_class in all_SVDs_class]
all_Vhs_class = [{c: SVDs[1] for c, SVDs in SVDs_class.items()} for SVDs_class in all_SVDs_class]
all_Vhs_class_proj= [{c: Vh_class @ R.T for c, Vh_class in Vhs_class.items()} for Vhs_class, R in zip(all_Vhs_class, all_Rs)]

all_SVDs = [np.linalg.svd(h[-1])[1:] for h in all_hs] # h[-1] for the end of training
all_VEs = np.array([np.cumsum(S**2, axis=1)/np.sum(S**2, axis=1)[:, None] for S, _ in all_SVDs])
# all_Vhs = [Vh for _, Vh in all_SVDs]

# for each class find the dimensionality of activity vs time (for each simulation)
# and plot box / mean+-std of dimensionality across simulations vs time
# (one curve for each class)
frac_threshold = .9

# plot all fraction of variance explained by class
VE_class = defaultdict(list)
for VEs_class in all_VEs_class:
    for c, VEs in VEs_class.items():
        VE_class[c].append(VEs)
Ds_class = {c: dimension(np.stack(VE), frac_threshold) for c,VE in VE_class.items()}

all_Ds = dimension(all_VEs, frac_threshold)

fig, ax = plt.subplots(figsize=(2,1.5))
# plt.subplots_adjust(left=.2, right=.99, bottom=.2, top=.99)
# d_max = np.max(all_Ds)
d_max = 30
_high = np.percentile(all_Ds, 95, axis=0)
_low = np.percentile(all_Ds, 5, axis=0)
# plot dimensionality all classes together
# ax.set_ylabel("Dimensionality")
# ax.set_xlabel("Timestep")
_xticks=list(np.arange(len(_high))+1)
ax.set_xticks(_xticks)
# ax.set_ylim([0, d_max])
ax.set_xlim(min(_xticks),max(_xticks))
ax.fill_between(np.arange(len(_high))+1, _low, _high, color='k', alpha=.3, lw=0)
ax.plot(np.arange(len(_high)) + 1, np.median(all_Ds, axis=0), c='k', ls='--', lw=2, label='all')
# plot dimensionality all classes separately
for i, (c, Ds) in enumerate(Ds_class.items()):
    _high = np.percentile(Ds, 95, axis=0)
    _low = np.percentile(Ds, 5, axis=0)
    ax.fill_between(np.arange(len(_high))+1, _low, _high, color=f'C{i}', alpha=.2, lw=0)
    ax.plot(np.arange(len(_high))+1, np.median(Ds, axis=0), c=f'C{i}', lw=2, label=types_set[c])
# ax.legend(loc='best', title='Class')
fig.tight_layout()
# fig.savefig(join(figs_dir, f'SVDs_and_dimensionality_by_class.{figs_ext}'), dpi=300)
fig.savefig(join(figs_dir, f'SVDs_and_dimensionality_all.{figs_ext}'), dpi=300)
plt.close(fig)


#############################################################   ANALYSIS OF WEIGHTS #######################################################################


print('Plot weights -- their SVDs')

# # Transforming the weights in each simulation using Vh brings them all close
# all_output_weights_proj = [W @ R.T for R, W in zip(all_Rs, all_output_weights)] # (C, N) x (N, N)
# all_input_weights_proj = [R @ W for R, W in zip(all_Rs, all_input_weights)]     # (N, N) x (N, a)
# all_rec_weights_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_rec_weights)]   # (N, N) x (N, N) x (N, N)
# all_rec_biases_proj = [R @ b for R, b in zip(all_Rs, all_rec_biases)]


#
# Labels with main text notation
#

# all SVDs are the same across simulations -- whether we rotate or not
# but the SVDs of the average matrices are going to be very different
all_rec_S_proj = [np.linalg.svd(W, compute_uv=False) for W in all_rec_weights_proj]
all_rec_S = [np.linalg.svd(W, compute_uv=False) for W in all_rec_weights]

fig, ax = plt.subplots(figsize=(3,1.6))
n_components = 15
_xticks = [0,5,10,15]
_yticks = [0,4,8]
ax.set_xlim(min(_xticks),max(_xticks))
# ax.set_ylim(min(_yticks),max(_yticks))
ax.set_xticks(_xticks)
# ax.set_yticks(_yticks)
ax.set_xticklabels([str(t) for t in _xticks])
# ax.set_yticklabels([str(t) for t in _yticks])
ax.axvspan(0, 3.5, color='wheat', alpha=0.5, lw=0)
# for S_proj, S in zip(all_rec_S_proj,all_rec_S):
#     ax.plot(np.arange(n_components)+1, S[:n_components], c='b', lw=.1, alpha=.2)
ax.plot(np.arange(n_components)+1, np.mean(all_rec_S,axis=0)[:n_components], ls='--', c='b', lw=2, label='av S orig')
ax.fill_between(np.arange(n_components)+1,
                np.percentile(all_rec_S,5,axis=0)[:n_components],
                np.percentile(all_rec_S,95,axis=0)[:n_components],
                ls='--', color='b', lw=0, alpha=.2, label='av S orig')
# ax.plot(np.arange(n_components)+1, np.mean(all_rec_S_proj,axis=0)[:n_components], ls=':', c='r', lw=2, label='av S rot')
S_mean = np.linalg.svd(np.mean(all_rec_weights, axis=0), compute_uv=False)
S_mean_proj = np.linalg.svd(np.mean(all_rec_weights_proj, axis=0), compute_uv=False)
ax.plot(np.arange(n_components)+1,S_mean[:n_components], c='b', lw=1, marker='o', markersize=3, label='S av orig')
ax.plot(np.arange(n_components)+1,S_mean_proj[:n_components], c='r', lw=1, marker='o', markersize=3, label='S av rot')
# ax.set_ylim(.1,10)
# ax.set_yscale('log')
# ax.set_xlabel("Index")
ax.set_ylabel("Singular value")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()
fig.savefig(join(figs_dir, f'SVDs_recurrent_weights.{figs_ext}'), dpi=300)
plt.close(fig)


fig, axs = plt.subplots(1,2, figsize=(6,3))
# plt.subplots_adjust(bottom=.1, top=.95, left=.05, right=1.)
# _vmax = 4 # np.max(np.fabs([np.max(_X), np.min(_X)]))
# _vmin = - _vmax
ax = axs[0]
_X = np.mean(all_rec_weights_proj,axis=0)
im = ax.imshow(_X[:20,:20], aspect='equal', interpolation='nearest', cmap='PiYG') #vmin=_vmin, vmax=_vmax,
ax = axs[1]
_X = np.mean(all_rec_weights,axis=0)
im = ax.imshow(_X[:20,:20], aspect='equal', interpolation='nearest', cmap='PiYG') #vmin=_vmin, vmax=_vmax,
fig.colorbar(im, ax=axs)
fig.tight_layout()
fig.savefig(join(figs_dir, f"recurrent_weights.{figs_ext}"), dpi=300)
plt.close(fig)


# Plot the actual (rotated) weights
fig, axs = plt.subplots(1,3, figsize=(6,2))
_vmax = 4 # np.max(np.fabs([np.max(_X), np.min(_X)]))
_vmin = - _vmax
# plt.subplots_adjust(bottom=.1, top=.95, left=.05, right=1.)
axs = axs.ravel()
n_simulations = 10
# 1. Input weights are all over the place
_weights = all_input_weights_proj[:n_simulations]
ax = axs[0]
# ax.set_title("Rotated W_i2h")
n_components = 3
# ax.set_xlabel(f"First {n_components} rows")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w[:n_components].ravel() for _w in _weights])
# _vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
# _vmin = - _vmax
im = ax.imshow(_X, aspect='auto', interpolation='nearest', cmap='PiYG') #vmin=_vmin, vmax=_vmax, 
for i in range(1, n_components):
    ax.axvline(i * _weights[0].shape[1] - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)


# 2. Recurrent weights can be aligned across simulations...
_weights = all_rec_weights_proj[:n_simulations]
ax = axs[1]
# ax.set_title("Rotated W_h2h")
n_components = 5
# ax.set_xlabel(f"Dominant {n_components}x{n_components} block")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w[:n_components,:n_components].ravel() for _w in _weights])
# _vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
# _vmin = - _vmax
im = ax.imshow(_X, aspect='auto', interpolation='nearest', cmap='PiYG') #vmin=_vmin, vmax=_vmax, 
for i in range(1, n_components):
    ax.axvline(i * n_components - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)


# and 3. consistently, the output weights also aligned across simulations.
_weights = all_output_weights_proj[:n_simulations]
ax = axs[2]
# ax.set_title("Rotated W_h2o")
n_components = 5
# ax.set_xlabel(f"First {n_components} cols")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w.T[:n_components].ravel() for _w in _weights])
# _vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
# _vmin = - _vmax
im = ax.imshow(_X, aspect='auto', interpolation='nearest', cmap='PiYG') #vmin=_vmin, vmax=_vmax, 
for i in range(1, n_components):
    ax.axvline(i * _weights[0].shape[0] - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(join(figs_dir, f'rotated_weights.{figs_ext}'), dpi=300)
plt.close(fig)


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

print('... plotting')
# one plot for each class
fig, axs = plt.subplots(1, len(hs_proj_class), figsize=(4*len(hs_proj_class), 5))
# plt.subplots_adjust(left=.05, right=.95)
for c,(hs_proj, ax) in enumerate(zip(hs_proj_class, axs.ravel())):
    ax.set_title(types_set[c])
    ax.set_ylabel('Simulation')
    ax.set_xlabel('Time')
    _hs = hs_proj[:, :, :n_components].reshape(len(hs_proj), -1)
    im = ax.imshow(_hs, vmin=-1, vmax=+1, aspect='auto', cmap='bwr')
    for i in range(1, hs_proj.shape[1]):
        ax.axvline(i*n_components-.5, ls='--', c='k')
    fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(join(figs_dir, f'alignment_weights-activity_classes.{figs_ext}'), dpi=300)
plt.close(fig)

# one plot for each time step
fig, axs = plt.subplots(1, hs_proj_class.shape[2], figsize=(4*hs_proj_class.shape[2], 5))
# plt.subplots_adjust(left=.05, right=.95)
_hs_proj_class = hs_proj_class.transpose(2,1,3,0) # (time, sim, N, class,)
for t, (hs_proj, ax) in enumerate(zip(_hs_proj_class, axs.ravel())):
    ax.set_title(f'Time {t+1}')
    ax.set_ylabel('Simulation')
    ax.set_xlabel('Class')
    ax.set_xticks([l for l in range(len(types_set))])
    ax.set_xticklabels([s[:t+1] for s in types_set], rotation=45, ha='right', fontsize=8)
    _hs = hs_proj[:,:n_components].reshape(len(hs_proj), -1)
    _vmax = np.max(np.fabs(_hs)) # 1
    _vmin = - _vmax # -1
    im = ax.imshow(_hs, vmin=_vmin, vmax=_vmax, aspect='auto', cmap='bwr')
    for i in range(1, n_components):
        ax.axvline(i*len(types_set)-.5, ls='--', c='k')
    fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(join(figs_dir, f'alignment_weights-activity_times.{figs_ext}'), dpi=300)
plt.close(fig)


# for each time-step, plot cosine similarity beween hidden activity before and after transition
fig, axs = plt.subplots(1, hs_proj_class.shape[2]-1, figsize=(4*(hs_proj_class.shape[2]-1), 5))
# plt.subplots_adjust(left=.05, right=.95)
_hs_proj_class = hs_proj_class.transpose(2,1,3,0) # (time, sim, N, class,)
for t, (hs_proj_old, hs_proj_new, ax) in enumerate(zip(_hs_proj_class[:-1], _hs_proj_class[1:], axs.ravel())):
    ax.set_title(f'Time {t+1} --> {t+2}')
    ax.set_ylabel('Simulation')
    ax.set_xlabel('Class')
    ax.set_xticks([l for l in range(len(types_set))])
    ax.set_xticklabels([s[:t+2] for s in types_set], rotation=45, ha='right')
    _sim = np.sum(hs_proj_old * hs_proj_new, axis=-2) / np.sqrt(np.sum(hs_proj_old**2, axis=-2) * np.sum(hs_proj_new**2, axis=-2))
    im = ax.imshow(_sim, vmin=0, vmax=1, aspect='auto', cmap='viridis')
    # for i in range(1, n_components):
    #     ax.axvline(i*len(types_set)-.5, ls='--', c='k')
    fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(join(figs_dir, f'alignment_activity_trans.{figs_ext}'), dpi=300)
plt.close(fig)

print('Compute components of activity along weights SVs - all simulations separately')

hs_proj_class = defaultdict(list)
for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
    for c, hs in hs_class.items():
        hs_proj_class[c].append(np.dot(hs, rec_Vh.T))
hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sims, time, seq, N)

# one plot for each time step -- one single simulation, all sequences
sims = [0]
# sims = [0,1,2]
print('... plotting simulations '+', '.join([str(sim) for sim in sims]))
fig, axss = plt.subplots(len(sims), hs_proj_class.shape[2], figsize=(2*hs_proj_class.shape[2], 2*len(sims)))
axss = axss.reshape(-1,hs_proj_class.shape[2])
plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05)
fig.tight_layout()
_hs_proj_class = hs_proj_class.transpose(1,2,3,4,0) # (sim, time, seq, N, class,) 
for axs, sim in zip(axss, sims):
    for t, (hs_proj, ax) in enumerate(zip(_hs_proj_class[sim], axs.ravel())):
        # ax.set_title(f'Time {t+1}')
        ax.set_ylabel('Sequence ID')
        ax.set_xlabel('Class')
        ax.set_xticks([l for l in range(len(types_set))])
        ax.set_xticklabels([s[:t+1] for s in types_set], rotation=45, ha='right', fontsize=8)
        _hs = hs_proj[:,:n_components].reshape(len(hs_proj), -1)
        _vmax = np.max(np.fabs(_hs)) # 1
        _vmin = - _vmax # -1
        im = ax.imshow(_hs, vmin=_vmin, vmax=_vmax, aspect='auto', cmap='bwr')
        for i in range(1, n_components):
            ax.axvline(i*len(types_set)-.5, ls='--', c='k')
        fig.colorbar(im, ax=ax, orientation='horizontal')
    fig.savefig(join(figs_dir, f'alignment_weights-activity_times_allseq.{figs_ext}'), dpi=300)
    plt.close(fig)




print('Cosine similarity')
# Ultrametricity over time
_hs_proj = hs_proj_class.transpose(1,2,0,3,4) # (class, sim, time, p, N) -> (sim, time, class, p, N,)
print(np.shape(_hs_proj))

# subsample -- for faster devel
ids = np.random.choice(_hs_proj.shape[-2], size=20, replace=False)
_ids_classes = (_hs_proj.shape[-2]*np.arange(_hs_proj.shape[-3])[:,None]  + ids[None,:]).reshape(-1)
_all_classes = [np.take(_classes, _ids_classes) for _classes in all_classes]
_hs_proj = np.take(_hs_proj, ids, axis=-2)

_shape = _hs_proj.shape
_hs_proj = np.reshape(_hs_proj, (*_shape[:2], _shape[2]*_shape[3], _shape[4]))

# plot cosine similarity of few simulations
hs = _hs_proj[0]
fig, axs = plt.subplots(1, hs.shape[0], figsize=(2*hs.shape[0], 2))
fig.subplots_adjust(top=.95, right=.8, left=.05, bottom=.05)
_cs = compute_cosine_similarity(hs)
for ax, _s in zip(axs.ravel(), _cs):
    im = ax.imshow(_s, vmin=0, vmax=+1, interpolation='nearest')
    # cb = fig.colorbar(im, ax=ax)
cbar_ax = fig.add_axes([.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)
fig.savefig(join(figs_dir, f'cosine_similarity.{figs_ext}'), dpi=300)
plt.close(fig)


if par.compute_uc:

    distance_dict = {'euclidean': lambda x: compute_distance(x),
                     'angle': lambda x: np.arccos(compute_cosine_similarity(x))}

    dist_type = 'euclidean'
    distance_f = distance_dict[dist_type]

    # ultrametric content over time for all sequences
    print('Ultra-metric content over time for all sequences')
    all_UCs = defaultdict(dict)
    for _n in [2,3,4,6,8,10,160]:
        print(RED_STR + f"{_n}" + END_STR)
        for _t in range(_hs_proj[0].shape[0]):
            print(CYAN_STR + f"{_t}" + END_STR)
            all_UCs[_n][_t] = [compute_umcontent(distance_f(hs[None,_t,:,:_n]), return_triplets=False)[0] for hs in _hs_proj[:10]]
    with open(join(data_dir, 'ultrametric_content_all.pkl'), 'wb') as f:
        pickle.dump(to_dict(all_UCs), f)


    print('Ultra-metric content over time by class')
    # calculate the ultrametric content within sequences that share the structure up to the current time
    # (this should be a lower bound to the ultrametric content for all the sequences together)
    all_UCs_class = defaultdict(lambda: defaultdict(dict))
    for _n in [2,3,4,6,8,10,160]:
        print(RED_STR + f"{_n}" + END_STR)
        # (for t=0 the sequence 1 element, and all sequences would be considered together;
        # we start from t=1)
        for _t in range(1,_hs_proj[0].shape[0]):
            print(CYAN_STR + f"{_t}" + END_STR)
            types_seq = np.take([c[:_t+1] for c in types_set], _all_classes[0])
            for _c in np.unique(types_seq):
                print(_c)
                _mask = types_seq == _c
                all_UCs_class[_n][_t][_c] = [compute_umcontent(distance_f(hs[None,_t,_mask,:_n]), return_triplets=False)[0] for hs in _hs_proj[:10]]
    with open(join(data_dir, 'ultrametric_content_class.pkl'), 'wb') as f:
        pickle.dump(to_dict(all_UCs_class), f)


    print('Ultra-metric content over time by shared partial sequence')
    # at time t we look for the groups of sequences that share the structure up to time t-1
    all_UCs_group = defaultdict(lambda: defaultdict(dict))
    for _n in [2,3,4,6,8,10,160]:
        print(RED_STR + f"{_n}" + END_STR)
        # (for t=0,1 the preceding part of the sequence is either empty or has 1 element,
        # so all sequences would be considered together; we start from t=2)
        for _t in range(2,_hs_proj[0].shape[0]):
            print(CYAN_STR + f"{_t}" + END_STR)
            types_seq = np.take([c[:_t] for c in types_set], _all_classes[0])
            for _c in np.unique(types_seq):
                print(_c)
                _mask = types_seq == _c
                all_UCs_group[_n][_t][_c] = [compute_umcontent(distance_f(hs[None,_t,_mask,:_n]), return_triplets=False)[0] for hs in _hs_proj[:10]]
    with open(join(data_dir, 'ultrametric_content_group.pkl'), 'wb') as f:
        pickle.dump(to_dict(all_UCs_group), f)

try:
    print('Plot ultra-metric content results')

    with open(join(data_dir, 'ultrametric_content_all.pkl'), 'rb') as f:
        all_UCs = pickle.load(f)
    with open(join(data_dir, 'ultrametric_content_group.pkl'), 'rb') as f:
        all_UCs_group = pickle.load(f)
    with open(join(data_dir, 'ultrametric_content_class.pkl'), 'rb') as f:
        all_UCs_class = pickle.load(f)
except Exception as e:
    print(f"{type(e).__name__}: Error loading ultra-metric content results\n\t{e}")


_n_to_plot = [2,3,4,6,10]
# _n_to_plot = list(all_UCs.keys())

fig, ax = plt.subplots(figsize=(2,1.6))
ax.set_xlabel('Time')
ax.set_ylabel('UC')
_xticks = np.arange(1,par.cue_size+par.L+1)
# _yticks = np.linspace(.4,.7,4)
ax.set_xticks(_xticks)
# ax.set_yticks(_yticks)
ax.set_xlim(min(_xticks),max(_xticks))
# ax.set_ylim(.35,.7)
for i, (_n, _UCs) in enumerate(all_UCs.items()):
    if _n not in _n_to_plot:
        continue
    _ts = np.array(list(_UCs.keys()))+1
    _ucs = np.array(list(_UCs.values())).T
    # for _uc in _ucs:
    #     ax.plot(_ts, _uc, lw=.1, c=f'C{i}')
    ax.plot(_ts, np.nanmean(_ucs, axis=0), lw=2, c=f'C{i}', label=f'{_n}')
    # ax.fill_between(_ts, np.percentile(_ucs, 10, axis=0), np.percentile(_ucs, 90, axis=0), lw=0, alpha=.2, color=f'C{i}')
# for i, (_n, _UCs_group) in enumerate(all_UCs_group.items()):
#     _ts = np.array(list(_UCs_group.keys()))+1
#     _uc = np.array([np.median(np.concatenate(list(_ucs_group.values()))) for _ucs_group in _UCs_group.values()])
#     ax.plot(_ts, _uc, lw=2, c=f'C{i}', ls=':')#, label=f'{_n}')
for i, (_n, _UCs_class) in enumerate(all_UCs_class.items()):
    if _n not in _n_to_plot:
        continue
    _ts = np.array(list(_UCs_class.keys()))+1
    _uc = np.array([np.median(np.concatenate(list(_ucs_class.values()))) for _ucs_class in _UCs_class.values()])
    ax.plot([1,len(_UCs_class)+1], 2*[_uc[-1]], lw=1, c=f'C{i}', ls='--', label=f'{_n}')
fig.tight_layout()
ax.legend(loc='best', title='num SV')
fig.savefig(join(figs_dir, f'ultrametric_content.{figs_ext}'), dpi=300)
plt.close(fig)

