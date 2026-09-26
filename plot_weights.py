"""
Utilities for loading and plotting Win, Wh, Wout across multiple simulations.
"""

import os
import pickle
from glob import glob
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch

import par
from lowrank_utils import align_singular_vectors


def load_weights_multi_sim(data_dir=None, classcomb_idx=None, L=None):
    """Load Win, Wh, Wout from all sim*.pth models in a run folder.

    Returns
    -------
    all_Win, all_Wh, all_Wout : list of np.ndarray
    """
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if data_dir is None:
        data_dir = join(par.folder, par.run_folder_name(L=L))

    model_paths = sorted(glob(join(data_dir, f'model_state_classcomb{classcomb_idx}_sim*.pth')))
    assert model_paths, f"No model files found in {data_dir}"
    all_models = [torch.load(path, map_location='cpu') for path in model_paths]
    all_Win = [m['i2h.weight'].numpy() for m in all_models]
    all_Wh = [m['h2h.weight'].numpy() for m in all_models]
    all_Wout = [m['h2o.weight'].numpy() for m in all_models]
    return all_Win, all_Wh, all_Wout


def load_types_set(data_dir=None, classcomb_idx=None, L=None):
    """Load the class labels for the chosen class combination."""
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if data_dir is None:
        data_dir = join(par.folder, par.run_folder_name(L=L))
    with open(join(data_dir, 'classes.pkl'), 'rb') as handle:
        classcomb = pickle.load(handle)
    return list(classcomb[classcomb_idx])


def _load_results_and_maps(data_dir, classcomb_idx):
    with open(join(data_dir, f'token_to_type_classcomb{classcomb_idx}.pkl'), 'rb') as handle:
        token_to_type = pickle.load(handle)
    with open(join(data_dir, f'token_to_set_classcomb{classcomb_idx}.pkl'), 'rb') as handle:
        token_to_set = pickle.load(handle)
    results_paths = sorted(glob(join(data_dir, f'results_task{par.task}_classcomb{classcomb_idx}_sim*.pkl')))
    assert results_paths, f"No results files found in {data_dir}"
    all_results = [pickle.load(open(path, 'rb')) for path in results_paths]
    return all_results, token_to_type, token_to_set


def align_weights_multi_sim(all_Win, all_Wh, all_Wout, data_dir=None,
                            classcomb_idx=None, L=None, epoch_time_ids=(-1, -1),
                            n_components=10):
    """Align Win/Wh/Wout across simulations (same rotation as lowrank.py).

    Returns
    -------
    all_Win_proj, all_Wh_proj, all_Wout_proj : list of np.ndarray
    """
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if data_dir is None:
        data_dir = join(par.folder, par.run_folder_name(L=L))

    all_results, token_to_type, token_to_set = _load_results_and_maps(data_dir, classcomb_idx)
    assert len(all_results) == len(all_Wh), "Mismatch between models and results"

    all_hs = [
        np.array([
            list(d.values()) for k, d in res['HiddenAct'].items() if token_to_set[k] == par.whichset
        ]).transpose((1, 2, 0, 3))
        for res in all_results
    ]

    all_rec_Vhs = [np.linalg.svd(W)[2] for W in all_Wh]
    all_final_hs = [h[epoch_time_ids] for h in all_hs]
    all_final_Vhs = [np.linalg.svd(h)[2] for h in all_final_hs]

    if par.common_rotation == 'weights':
        _all_Rs = all_rec_Vhs
    elif par.common_rotation == 'hidden':
        _all_Rs = all_final_Vhs
    else:
        raise NotImplementedError(f'Unknown option "{par.common_rotation}" for `common_rotation`.')

    all_refs = [hs[0] for hs in all_final_hs]
    all_Rs = align_singular_vectors(_all_Rs, all_refs, n_components=n_components)

    all_Win_proj = [R @ W for R, W in zip(all_Rs, all_Win)]
    all_Wh_proj = [R @ W @ R.T for R, W in zip(all_Rs, all_Wh)]
    all_Wout_proj = [W @ R.T for R, W in zip(all_Rs, all_Wout)]
    return all_Win_proj, all_Wh_proj, all_Wout_proj


def average_aligned_weights(all_Win_proj, all_Wh_proj, all_Wout_proj):
    """Mean of aligned Win, Wh, Wout across simulations."""
    return (
        np.mean(np.stack(all_Win_proj), axis=0),
        np.mean(np.stack(all_Wh_proj), axis=0),
        np.mean(np.stack(all_Wout_proj), axis=0),
    )


def _filename_with_L(filename, L):
    """Insert L into a default filename, e.g. weights_foo.svg -> weights_foo_L4.svg."""
    if L is None or f'_L{L}' in filename:
        return filename
    root, ext = os.path.splitext(filename)
    return f'{root}_L{L}{ext}'


def plot_weights_multi_sim(all_Win, all_Wh, all_Wout, figs_dir,
                           n_simulations=None, wh_slice=20, cmap='PiYG',
                           filename=None,
                           L=None, types_set=None, classcomb_idx=None):
    """Plot Win, Wh, Wout heatmaps for multiple simulations.

    Layout: three rows (Win | Wh | Wout), one column per simulation.
    Wh is shown as the leading ``wh_slice x wh_slice`` block for readability.
    """
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if filename is None:
        filename = _filename_with_L('weights_Win_Wh_Wout.svg', L)
    n = len(all_Win) if n_simulations is None else min(n_simulations, len(all_Win))
    all_Win, all_Wh, all_Wout = all_Win[:n], all_Wh[:n], all_Wout[:n]

    vmax = max(
        np.max(np.abs(np.stack(all_Win))),
        np.max(np.abs([W[:wh_slice, :wh_slice] for W in all_Wh])),
        np.max(np.abs(np.stack(all_Wout))),
    )

    fig, axs = plt.subplots(3, n, figsize=(1.8 * n, 5.5), squeeze=False,
                            constrained_layout=True)
    row_labels = [r'$W_{\mathrm{in}}$', r'$W_{\mathrm{h}}$', r'$W_{\mathrm{out}}$']
    for i, (Win, Wh, Wout) in enumerate(zip(all_Win, all_Wh, all_Wout)):
        mats = [Win, Wh[:wh_slice, :wh_slice], Wout]
        for j, (ax, mat) in enumerate(zip(axs[:, i], mats)):
            im = ax.imshow(mat, aspect='auto', interpolation='nearest',
                           cmap=cmap, vmin=-vmax, vmax=vmax)
            if j == 0:
                ax.set_title(f'sim {i}')
            if i == 0:
                ax.set_ylabel(row_labels[j])
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)

    title_parts = []
    if L is not None:
        title_parts.append(f'L={L}')
    title_parts.append(f'classcomb={classcomb_idx} ({len(types_set) if types_set else par.n_types} classes)')
    if types_set:
        title_parts.append(', '.join(types_set))
    fig.suptitle('\n'.join(title_parts), fontsize=9)

    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(join(figs_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_mean_aligned_weights(mean_Win, mean_Wh, mean_Wout, figs_dir,
                              wh_slice=20, cmap='PiYG',
                              filename=None,
                              L=None, types_set=None, classcomb_idx=None,
                              n_simulations=None):
    """Plot averaged aligned Win, Wh, Wout (one panel each)."""
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if filename is None:
        filename = _filename_with_L('weights_mean_aligned.svg', L)

    mats = [mean_Win, mean_Wh[:wh_slice, :wh_slice], mean_Wout]
    titles = [r'$\langle W_{\mathrm{in}}\rangle$',
              r'$\langle W_{\mathrm{h}}\rangle$',
              r'$\langle W_{\mathrm{out}}\rangle$']
    vmax = max(np.max(np.abs(m)) for m in mats)

    fig, axs = plt.subplots(1, 3, figsize=(8, 2.6), constrained_layout=True)
    for ax, mat, title in zip(axs, mats, titles):
        im = ax.imshow(mat, aspect='auto', interpolation='nearest',
                       cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axs, fraction=0.03, pad=0.02)

    title_parts = ['mean aligned weights']
    if L is not None:
        title_parts.append(f'L={L}')
    if n_simulations is not None:
        title_parts.append(f'n_sim={n_simulations}')
    title_parts.append(f'classcomb={classcomb_idx} ({len(types_set) if types_set else par.n_types} classes)')
    if types_set:
        title_parts.append(', '.join(types_set))
    fig.suptitle('\n'.join(title_parts), fontsize=9)

    os.makedirs(figs_dir, exist_ok=True)
    fig.savefig(join(figs_dir, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return mean_Win, mean_Wh, mean_Wout


def plot_weights_multi_sim_from_run(L=None, figs_dir=None, data_dir=None,
                                    classcomb_idx=None, **plot_kwargs):
    """Load Win/Wh/Wout for one run and plot them across simulations."""
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if data_dir is None:
        data_dir = join(par.folder, par.run_folder_name(L=L))
    if figs_dir is None:
        figs_dir = join(par.folder, 'figs', os.path.basename(data_dir))

    all_Win, all_Wh, all_Wout = load_weights_multi_sim(
        data_dir=data_dir, classcomb_idx=classcomb_idx, L=L
    )
    types_set = load_types_set(data_dir=data_dir, classcomb_idx=classcomb_idx, L=L)
    plot_weights_multi_sim(
        all_Win, all_Wh, all_Wout, figs_dir,
        L=L if L is not None else par.L,
        types_set=types_set,
        classcomb_idx=classcomb_idx,
        **plot_kwargs,
    )
    return all_Win, all_Wh, all_Wout


def plot_mean_aligned_weights_from_run(L=None, figs_dir=None, data_dir=None,
                                       classcomb_idx=None, epoch_time_ids=(-1, -1),
                                       **plot_kwargs):
    """Align Win/Wh/Wout across sims, average, and plot the means."""
    if classcomb_idx is None:
        classcomb_idx = par.classcomb
    if data_dir is None:
        data_dir = join(par.folder, par.run_folder_name(L=L))
    if figs_dir is None:
        figs_dir = join(par.folder, 'figs', os.path.basename(data_dir))

    all_Win, all_Wh, all_Wout = load_weights_multi_sim(
        data_dir=data_dir, classcomb_idx=classcomb_idx, L=L
    )
    all_Win_proj, all_Wh_proj, all_Wout_proj = align_weights_multi_sim(
        all_Win, all_Wh, all_Wout,
        data_dir=data_dir, classcomb_idx=classcomb_idx, L=L,
        epoch_time_ids=epoch_time_ids,
    )
    mean_Win, mean_Wh, mean_Wout = average_aligned_weights(
        all_Win_proj, all_Wh_proj, all_Wout_proj
    )
    types_set = load_types_set(data_dir=data_dir, classcomb_idx=classcomb_idx, L=L)
    plot_mean_aligned_weights(
        mean_Win, mean_Wh, mean_Wout, figs_dir,
        L=L if L is not None else par.L,
        types_set=types_set,
        classcomb_idx=classcomb_idx,
        n_simulations=len(all_Win),
        **plot_kwargs,
    )
    return mean_Win, mean_Wh, mean_Wout


if __name__ == '__main__':
    plot_weights_multi_sim_from_run(L=par.L)
    plot_mean_aligned_weights_from_run(L=par.L)
