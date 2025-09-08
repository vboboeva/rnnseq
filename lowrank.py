import numpy as np
import scipy
from numpy import savetxt
import torch
import os
from os.path import join
from glob import glob
import par
import matplotlib.pyplot as plt
import pickle
# from make_data import make_data
from sklearn.decomposition import PCA
from collections import defaultdict

from analysis_utils import compute_umcontent, compute_cosine_similarity, compute_distance

t=0
datasplit=0
c=0

folder_name = f'Task{par.task}_N{par.n_hidden}_nlatent{par.n_latent}_L{par.L}_m{par.m}_alpha{par.alpha}_nepochs{par.n_epochs}_ntypes{par.n_types}_fractrain{par.frac_train:.1f}_obj{par.loss}_init{par.init}_transfer{par.transfer}_cuesize{par.cue_size}_delay{par.delay}_datasplit{datasplit}'

print(folder_name)

classcomb_path = join(folder_name, 'classes.pkl')

if not os.path.exists(classcomb_path):
    raise ValueError("Files not found!")

with open(classcomb_path, 'rb') as handle:
    classcomb = pickle.load(handle)

types_set = list(classcomb[c])
token_to_type_path = join(folder_name, f'token_to_type_classcomb{c}.pkl')

if not os.path.exists(token_to_type_path):
    raise ValueError("Files not found!")
with open(token_to_type_path, 'rb') as handle:
    token_to_type = pickle.load(handle)


figs_dir = join(par.folder, 'figs', folder_name, f'alignment_by_{par.common_rotation}')
os.makedirs(figs_dir, exist_ok=True)

def all_sims_paths (folder_name):
    """
    Get all simulation paths for a given folder name.
    """
    model_paths = sorted(glob(join(par.folder, folder_name, f'model_state_classcomb{t}_sim*.pth')))
    results_paths = sorted(glob(join(par.folder, folder_name, f'results_task{par.task}_classcomb{c}_sim*.pkl')))
    return model_paths, results_paths

model_paths, results_paths = all_sims_paths(folder_name)

all_models = [torch.load(path, map_location="cpu") for path in model_paths]
all_results = [pickle.load(open(path, 'rb')) for path in results_paths]

# 0. Extract some info about results -- all lists indexed by simulation
all_sequences = [np.array(list(res['HiddenAct'].keys())) for res in all_results] # classes (n_seq,)
assert np.all([all_sequences[0] == s for s in all_sequences]), \
       "All sequences must be in the same order across simulations"
all_classes = [np.array([token_to_type[k] for k in res['HiddenAct'].keys()]) for res in all_results] # classes (n_seq,)
all_hs = [np.array([list(d.values()) for d in res['HiddenAct'].values()]).transpose((1,2,0,3)) for res in all_results]    # hidden act (n_epochs_saved, L, n_seq, n_hidden,) -- convenient for SVD

# extract all weights
all_input_weights = [m['i2h.weight'].numpy() for m in all_models]
all_output_weights = [m['h2o.weight'].numpy() for m in all_models]
all_rec_weights = [m['h2h.weight'].numpy() for m in all_models]
# SVD of recurrent weights
all_rec_SVDs = [np.linalg.svd(W)[1:] for W in all_rec_weights]
all_rec_Ss = [S for S, _ in all_rec_SVDs]
all_rec_Vhs = [Vh for _, Vh in all_rec_SVDs]

# SVD at final time step
all_final_hs = [h[-1,-1] for h in all_hs]   # hidden act after training at last time step, (n_seq, n_hidden,)
# all_final_hs = [h[-1,-1] - np.mean(h[-1,-1], axis=0)[None,:] for h in all_hs]   # hidden act after training at last time step, (n_seq, n_hidden,)
_SVDs = [np.linalg.svd(h) for h in all_final_hs]
all_final_VEs = [np.cumsum(S**2)/np.sum(S**2) for _, S, _ in _SVDs]   # fraction of variance explained by first principal components
all_final_Vhs = [Vh for _, _, Vh in _SVDs]    # principal vectors, component (1st index) by feature (2nd index)

# define rotation matrices
if par.common_rotation == 'weights':
    all_Rs = all_rec_Vhs
elif par.common_rotation == 'hidden':
    all_Rs = all_final_Vhs
else:
    raise NotImplementedError(f'Unknown option "{par.common_rotation}" for `common_rotation`.')
all_refs = [hs[0] for hs in all_final_hs]

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
all_rec_Vhs_aligned = align_singular_vectors(all_rec_Vhs, all_refs, n_components=10)
all_Rs = align_singular_vectors(all_Rs, all_refs, n_components=10)

def plot_SVD_results (all_VEs, all_hs_proj, all_Vhs, n_components=2, **kwargs):

    fig, axs = plt.subplots(1, 2, figsize=(10,4))

    # plot all singular values across simulations
    ax = axs[0]
    ax.set_title("Similar dimensionality/structure of final activity across simulations")
    for i, VE in enumerate(all_VEs):
        ax.plot(1-VE, lw=.3, c='k', alpha=.2)
    ax.plot(1-np.mean(all_VEs, axis=0), c='r', ls='--')
    ax.set_xlabel("Index")
    ax.set_ylabel("Fraction of (un)explained variance")
    ax.set_yscale('log')

    # plot all (left) singular vectors across simulations
    ax = axs[1]
    ax.set_title("Similar dimensionality/structure of final activity across simulations")
    _X = np.vstack([Vh[:n_components].reshape(1,-1) for Vh in all_Vhs])
    _vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
    _vmin = - _vmax
    im = ax.imshow(_X,
                    aspect='auto', interpolation='nearest',
                    vmin=_vmin, vmax=_vmax,
                    **kwargs)
    for i in range(1, n_components):
        ax.axvline(i*all_Vhs[0].shape[1]-.5, c='k', ls='--')
    ax.set_xlabel("PC, input unit")
    ax.set_ylabel("Simulation ID")
    fig.colorbar(im, ax=ax)
    fig.savefig(join(figs_dir, f'hidden_last_step_SVD.png'), dpi=300)
    plt.close(fig)

    # plot all rotated data across simulations
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=.1, top=.95, right=.95)
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
    fig.savefig(join(figs_dir, f'hidden_last_step_coeffs.png'), dpi=300)
    plt.close(fig)


#
#   ANALYSIS OF HIDDEN-LAYER ACTIVITY
#
# projections along principal components, i.e. rotated data (n_seq, n_epochs_saved, L, n_hidden,)
all_hs_proj = [np.dot(h, R.T) for h, R in zip(all_hs, all_Rs)]
all_final_hs_proj = [h[-1,-1] for h in all_hs_proj]
plot_SVD_results(all_final_VEs, all_final_hs_proj, all_Rs, n_components=6, cmap='bwr')


# Dimensionality reduces over time for any given class

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
def dimension (VE, threshold=.8):
    '''
    Input
    -----
    VE, np.array (N,) or (M,N)
        fraction of explained variance. If (N,), for a single N-dimensional dataset
        if (M,N,) for M different datasets.
    Output
    ------
    dimensions: np.array (1,) or (M,)
    '''
    _shape = VE.shape
    _VE = VE
    if len(_shape) == 1:
        _VE = np.reshape(VE, (1,*VE.shape))
    above_threshold = _VE > threshold
    dimensions = np.argmax(above_threshold, axis=-1) + 1
    return dimensions

frac_threshold = .9

# plot all fraction of variance explained by class
VE_class = defaultdict(list)
for VEs_class in all_VEs_class:
    for c, VEs in VEs_class.items():
        VE_class[c].append(VEs)
Ds_class = {c: dimension(np.stack(VE), frac_threshold) for c,VE in VE_class.items()}


all_Ds = dimension(all_VEs, frac_threshold)

fig, axs = plt.subplots(2,len(VE_class) + 1, figsize=(4*(len(VE_class) + 1),8))
plt.subplots_adjust(left=.05, right=.95, wspace=.4)
for (c, VEs), ax in zip(VE_class.items(), axs[0,1:]):
    ax.set_title(types_set[c])
    ax.set_yscale('log')
    ax.set_xlabel('Index')
    ax.set_ylabel('Fraction of (un)explained variance')
    _VEs = np.mean(VEs,axis=0)
    for t, VE in enumerate(_VEs):
        ax.plot(1-VE, c=f'C{t}', label=f't={t}')
    ax.legend(loc='upper right')
ax = axs[0,0]
_VEs = np.mean(all_VEs, axis=0)
ax.set_title('All sequences')
ax.set_yscale('log')
ax.set_xlabel('Index')
ax.set_ylabel('Fraction of (un)explained variance')
for t, VE in enumerate(_VEs):
    ax.plot(1-VE, c=f'C{t}', label=f't={t}')
ax.legend(loc='upper right')

d_max = max([np.max(Ds) for Ds in Ds_class.values()])
for (c, Ds), ax in zip(Ds_class.items(), axs[1,1:]):
    ax.set_ylabel("Dimensionality")
    ax.set_xlabel("Timestep")
    ax.set_ylim([0, d_max])
    # for D in Ds:
    #     ax.plot(D, c='k', lw=.1, alpha=.2)
    _high = np.percentile(Ds, 95, axis=0)
    _low = np.percentile(Ds, 5, axis=0)
    ax.fill_between(np.arange(len(_high)), _low, _high, color='k', alpha=.4, lw=0)
    ax.plot(np.median(Ds, axis=0), c='r', ls='--', lw=2)
# plot dimensionality all classes together
ax = axs[1,0]
ax.set_ylabel("Dimensionality")
ax.set_xlabel("Timestep")
ax.set_ylim([0, d_max])
# for D in Ds:
#     ax.plot(D, c='k', lw=.1, alpha=.2)
_high = np.percentile(all_Ds, 95, axis=0)
_low = np.percentile(all_Ds, 5, axis=0)
ax.fill_between(np.arange(len(_high)), _low, _high, color='k', alpha=.4, lw=0)
ax.plot(np.median(all_Ds, axis=0), c='r', ls='--', lw=2)

fig.savefig(join(figs_dir, 'SVDs_and_dimensionality_by_class.png'), dpi=300)
plt.close(fig)




#
#   ANALYSIS OF WEIGHTS
#

# Transforming the weights in each simulation using Vh brings them all close

all_output_weights_proj = [W @ R.T for R, W in zip(all_Rs, all_output_weights)]
all_input_weights_proj = [R @ W for R, W in zip(all_Rs, all_input_weights)]
all_rec_weights_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_rec_weights)]

# all SVDs are the same across simulations -- whether we rotate or not
# but the SVDs of the average matrices are going to be very different
all_rec_S_proj = [np.linalg.svd(W, compute_uv=False) for W in all_rec_weights_proj]
all_rec_S = [np.linalg.svd(W, compute_uv=False) for W in all_rec_weights]
fig, ax = plt.subplots()
n_components = 10
for S_proj, S in zip(all_rec_S_proj,all_rec_S):
    ax.plot(S[:n_components], c='b', lw=.1, alpha=.2)
    ax.plot(S_proj[:n_components], c='r', lw=.1, alpha=.2)
ax.plot(np.mean(all_rec_S,axis=0)[:n_components], ls='--', c='b', lw=2, label='average of S of original mat')
ax.plot(np.mean(all_rec_S_proj,axis=0)[:n_components], ls=':', c='r', lw=2, label='average of S of rotated mat')
S_mean = np.linalg.svd(np.mean(all_rec_weights, axis=0), compute_uv=False)
S_mean_proj = np.linalg.svd(np.mean(all_rec_weights_proj, axis=0), compute_uv=False)
ax.plot(S_mean[:n_components], c='b', lw=2, label='S of average original mat')
ax.plot(S_mean_proj[:n_components], c='r', lw=2, label='S of average rotated mat')
# ax.set_ylim(.1,10)
# ax.set_yscale('log')
ax.set_xlabel("index")
ax.set_ylabel("Singular value")
ax.legend(loc='upper right')
fig.savefig(join(figs_dir, 'SVDs_recurrent_weights.png'), dpi=300)
plt.close(fig)


# Plot the actual (rotated) weights
fig, axs = plt.subplots(1,3, figsize=(16,5))
plt.subplots_adjust(bottom=.1, top=.95, left=.05, right=1.)
axs = axs.ravel()

# 1. Input weights are all over the place
_weights = all_input_weights_proj
ax = axs[0]
ax.set_title("Rotated input weights across simulations")
n_components = 3
ax.set_xlabel(f"All weights, rows concat (first {n_components} rows)")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w[:n_components].ravel() for _w in _weights])
_vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
_vmin = - _vmax
im = ax.imshow(_X, vmin=_vmin, vmax=_vmax, aspect='auto', interpolation='nearest', cmap='bwr')
for i in range(1, n_components):
    ax.axvline(i * _weights[0].shape[1] - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)
fig.savefig(join(figs_dir, f'rotated_weights.png'), dpi=300)
plt.close(fig)

# 2. Recurrent weights can be aligned across simulations...
_weights = all_rec_weights_proj
ax = axs[1]
ax.set_title("Rotated recurrent weights across simulations")
n_components = 5
ax.set_xlabel(f"All weights, rows concat (dominant {n_components}x{n_components} block)")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w[:n_components,:n_components].ravel() for _w in _weights])
_vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
_vmin = - _vmax
im = ax.imshow(_X, vmin=_vmin, vmax=_vmax, aspect='auto', interpolation='nearest', cmap='bwr')
for i in range(1, n_components):
    ax.axvline(i * n_components - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)

# and 3. consistently, the output weights also aligned across simulations.
_weights = all_output_weights_proj
ax = axs[2]
ax.set_title("Rotated output weights across simulations")
n_components = 5
ax.set_xlabel(f"All weights, cols concat (first {n_components} cols)")
ax.set_ylabel("Simulation ID")
_X = np.stack([_w.T[:n_components].ravel() for _w in _weights])
_vmax = np.max(np.fabs([np.max(_X), np.min(_X)]))
_vmin = - _vmax
im = ax.imshow(_X, vmin=_vmin, vmax=_vmax, aspect='auto', interpolation='nearest', cmap='bwr')
for i in range(1, n_components):
    ax.axvline(i * _weights[0].shape[0] - .5, ls='--', c='k')
fig.colorbar(im, ax=ax)
fig.savefig(join(figs_dir, f'rotated_weights.png'), dpi=300)
plt.close(fig)


#
#   ALIGNMENT ANALYSIS
#
# We want to check, at each time, and for each class, the **alignment** of the activity
# with the modes of the recurrent weights (cosine similarity between principal vectors)
# This is done after rotating activities and weights using the common transformation
# identified at the last time step.

all_rec_Vhs_aligned = align_singular_vectors(all_rec_Vhs, all_refs, n_components=3)

hs_proj_class = defaultdict(list)
for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
    for c, hs in hs_class.items():
        hs_proj_class[c].append(np.mean(np.dot(hs, rec_Vh.T), axis=1))
hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sim, time, p, N)

n_components = 4

# one plot for each class
fig, axs = plt.subplots(1, len(hs_proj_class), figsize=(4*len(hs_proj_class), 5))
plt.subplots_adjust(left=.05, right=.95)
for c,(hs_proj, ax) in enumerate(zip(hs_proj_class, axs.ravel())):
    ax.set_title(types_set[c])
    ax.set_ylabel('Simulation')
    ax.set_xlabel('Time')
    _hs = hs_proj[:,:,:n_components].reshape(len(hs_proj), -1)
    im = ax.imshow(_hs, vmin=-1, vmax=+1, aspect='auto', cmap='bwr')
    for i in range(1, hs_proj.shape[1]):
        ax.axvline(i*n_components-.5, ls='--', c='k')
    fig.colorbar(im, ax=ax)
fig.savefig(join(figs_dir, f'alignment_weights-activity_classes.png'), dpi=300)
plt.close(fig)

# one plot for each time step
fig, axs = plt.subplots(1, hs_proj_class.shape[2], figsize=(4*hs_proj_class.shape[2], 5))
plt.subplots_adjust(left=.05, right=.95)
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
fig.savefig(join(figs_dir, f'alignment_weights-activity_times.png'), dpi=300)
plt.close(fig)


# for each time-step, plot cosine similarity beween hidden activity before and after transition
fig, axs = plt.subplots(1, hs_proj_class.shape[2]-1, figsize=(4*(hs_proj_class.shape[2]-1), 5))
plt.subplots_adjust(left=.05, right=.95)
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
fig.savefig(join(figs_dir, f'alignment_activity_trans.png'), dpi=300)
plt.close(fig)




hs_proj_class = defaultdict(list)
for rec_Vh, hs_class in zip(all_rec_Vhs_aligned, all_hs_class):
    for c, hs in hs_class.items():
        hs_proj_class[c].append(np.dot(hs, rec_Vh.T))
hs_proj_class = np.stack([np.array(hs_proj) for c, hs_proj in hs_proj_class.items()]) # (class, sims, time, seq, N)

# one plot for each time step -- one single simulation, all sequences
sims = [0,1,2]
fig, axss = plt.subplots(len(sims), hs_proj_class.shape[2], figsize=(4*hs_proj_class.shape[2], 4*len(sims)))
plt.subplots_adjust(left=.05, right=.95, top=.95, bottom=.05)

_hs_proj_class = hs_proj_class.transpose(1,2,3,4,0) # (sim, time, seq, N, class,) 
for axs, sim in zip(axss, sims):
    for t, (hs_proj, ax) in enumerate(zip(_hs_proj_class[sim], axs.ravel())):
        ax.set_title(f'Time {t+1}')
        ax.set_ylabel('Sequence')
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
    fig.savefig(join(figs_dir, f'alignment_weights-activity_times_allseq.png'), dpi=300)
    plt.close(fig)



# Ultrametricity over time
_hs_proj = hs_proj_class.transpose(1,2,0,3,4) # (class, sim, time, p, N) -> (sim, time, class, p, N,)
# subsample -- for faster devel
# _hs_proj = np.take(_hs_proj, np.random.choice(_hs_proj.shape[-2], size=10, replace=False), axis=-2)
_shape = _hs_proj.shape
_hs_proj = np.reshape(_hs_proj, (*_shape[:2], _shape[2]*_shape[3], _shape[4]))

# plot cosine similarity of few simulations
fig, axss = plt.subplots(3, _hs_proj[0].shape[0], figsize=(4*_hs_proj[0].shape[0], 3*4))
plt.subplots_adjust(top=.95, right=.95, left=.05, bottom=.05)
for hs, axs in zip(_hs_proj, axss):
    # hs (L, p, N) --> _cs (L, p, p)
    _cs = compute_cosine_similarity(hs)
    for ax, _s in zip(axs.ravel(), _cs):
        im = ax.imshow(_s, vmin=0, vmax=+1, interpolation='nearest')
        cb = fig.colorbar(im, ax=ax)
fig.savefig(join(figs_dir, 'cosine_similarity.png'), dpi=100)
plt.close(fig)

# ultrametric content over time
# dictionary {number of singular vectors: ultrametric content over time}
# all_UCs = {_n: np.stack([compute_umcontent(np.arccos(compute_cosine_similarity(hs[:,:,:_n])), return_triplets=False) for hs in _hs_proj[:5]]) for _n in [2,3,4,6,8,10]}
# fig_title = 'ultrametric_content_by_time_angle.png'
all_UCs = {_n: np.stack([compute_umcontent(compute_distance(hs[:,:,:_n]), return_triplets=False) for hs in _hs_proj[:10]]) for _n in [2,3,4,6,8,10]}
with open(join(par.folder, folder_name, 'ultrametric_content.pkl'), 'wb') as f:
    pickle.dump(all_UCs, f)
fig_title = 'ultrametric_content_by_time_euclidean.png'


with open(join(par.folder, folder_name, 'ultrametric_content.pkl'), 'rb') as f:
    all_UCs = pickle.load(f)

fig, ax = plt.subplots()
ax.set_xlabel('Time')
ax.set_ylabel('UC')
for i, (_n, _UCs) in enumerate(all_UCs.items()):
    for _uc in _UCs:
        ax.plot(np.arange(len(_uc))+1, _uc, lw=.1, c=f'C{i}')
    ax.plot(np.arange(len(_UCs[0]))+1, np.nanmean(_UCs, axis=0), lw=2, c=f'C{i}', ls='--', label=f'{_n}')
ax.legend(loc='best', title='num SV')
fig.savefig(join(figs_dir, fig_title), dpi=300)
plt.close(fig)
