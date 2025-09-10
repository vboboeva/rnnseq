import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import string
import torch
from matplotlib import rc
from pylab import rcParams
import sys
import os
from os.path import join
import pickle
import par
import scipy.cluster.hierarchy as sch
from itertools import combinations
from collections import defaultdict
from mpl_toolkits.axes_grid1 import make_axes_locatable

######################################################
# Dot product of the representations in hidden layer #
######################################################


"""Return the Hamming distance between equal-length sequences."""
def compute_hamming_distance(tokens_set):
    # print(tokens_set)
    D=np.ndarray((par.L, len(tokens_set), len(tokens_set)))

    for k, l in enumerate(range(1,par.L+1)):
        _set = tokens_set[:,:l]
        for i, s1 in enumerate(_set):
            for j, s2 in enumerate(_set):
                if len(s1) != len(s2):
                    raise ValueError("Undefined for sequences of unequal length.")
                else:
                    D[k,i,j] = sum(char1 != char2 for char1, char2 in zip(s1, s2))
    return D

def compute_correlation_matrix(y_hidden, task):
    y_hidden=np.array(y_hidden)
    n_set=len(y_hidden)
    overlap_tokens=np.ndarray((par.cue_size+par.L+par.delay, n_set, n_set))
    
    for idx_token1 in range(n_set):
        # print(idx_token1)
        for idx_token2 in range(n_set):
            # print(idx_token2)
            for idx_t in range(par.cue_size+par.L+par.delay):
                v1 = y_hidden[idx_token1, idx_t, :]
                norm1 = np.sqrt(np.sum(v1*v1))
                v2 = y_hidden[idx_token2, idx_t, :]
                norm2 = np.sqrt(np.sum(v2*v2))
                overlap_tokens[idx_t, idx_token1, idx_token2] = np.dot(v1, v2)/(norm1 * norm2)
    return overlap_tokens

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

def compute_cosine_similarity (X):
    '''
    X: ndarray (..., p, D)
        p = number of data points
        D = dimension

    output: ndarray (..., p, p)
        Cosine similarity between data points
    '''
    *_shape, p, D = X.shape
    _X = X.reshape(-1, p, D)
    _dot_prod = np.sum(_X[:,None,:,:] * _X[:,:,None,:], axis=-1)
    _norm = np.sqrt(np.sum(_X**2, axis=-1))
    _cs = _dot_prod / (_norm[:,:,None]*_norm[:,None,:])

    _cs = _cs.reshape(*_shape, p, p)

    return _cs

def compute_distance (X):
    '''
    X: ndarray (..., p, D)
        p = number of data points
        D = dimension

    output: ndarray (..., p, p)
        Euclidean distance between data points
    '''
    *_shape, p, D = X.shape
    _X = X.reshape(-1, p, D)
    
    _dot_prod = np.sum(_X[:,None,:,:] * _X[:,:,None,:], axis=-1)
    _norm_sq = np.sum(_X**2, axis=-1)
    _dist = np.sqrt(_norm_sq[:,:,None] + _norm_sq[:,None,:] - 2.*_dot_prod)
    _dist = _dist.reshape(*_shape, p, p)

    return _dist

def compute_umcontent(M, return_triplets=True):
    '''
    M: ndarray (time, seq, seq)
        Matrix of cosine similarities between sequences
        (i.e. between their corresponding hidden activities)
        at every time in the sequence.

    returns tuple of ndarray (time,)
    - d_min / d_max
    - d_med / d_max
    - uc
    '''

    D_combos = np.stack([[sorted([D[i,j], D[j,k], D[i,k]]) for i, j, k in combinations(range(D.shape[0]), 3)] for D in M])
    # D_combos = np.stack([np.sort(np.array([[D[i,j], D[j,k], D[i,k]] for i, j, k in combinations(range(D.shape[0]), 3)]), axis=1) for D in M])

    if return_triplets:
        return D_combos

    # D_combos (time, triplets, 3)
    _log_min_max = np.log( D_combos[:,:,0] / D_combos[:,:,2] )
    _log_med_max = np.log( D_combos[:,:,1] / D_combos[:,:,2] )
    lamda_combos = (_log_min_max - _log_med_max) / (_log_min_max + _log_med_max)
    return np.nanmean(lamda_combos, axis=1)


def plot_triplets_sorted(delta1_t, delta2_t, lamda_t, filename):
    fig, ax = plt.subplots(1, (par.cue_size+par.L+par.delay), figsize=[14, 2])
    ax = ax.ravel()

    for idx_t in range(par.cue_size+par.L+par.delay):
        notmask = np.where(delta1_t[idx_t] != -99)

        x=np.arange(0.0,1.1,0.1)
        ax[idx_t].plot(x, x, alpha=1, color='white', lw=1.5, zorder=2)
        ax[idx_t].plot(x, 1.-x, alpha=1, color='white', lw=1.5, zorder=2)
        im = ax[idx_t].hexbin(np.append(x, delta2_t[idx_t, notmask]), np.append(x, delta1_t[idx_t, notmask]), gridsize=50)
        ax[idx_t].set_xlabel('dmed/dmax') 
        ax[idx_t].set_ylabel('dmin/dmax')
        ax[idx_t].set_title('uc = %.2f'%np.nanmean(lamda_t[idx_t, notmask]))
        ax[idx_t].set_xlim(0.5,1)
        ax[idx_t].set_ylim(0.,1)

    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(f"{par.folder}/figs/umcontent_{filename}.svg", bbox_inches="tight", dpi=600)

def compute_overlap(M, tokens, types):

    n_set=len(tokens)
    tokens = np.array([list(token) for token in tokens])

    M_t = np.ndarray((par.cue_size+par.L+par.delay, n_set, n_set))
    sorted_idx_t =  np.ndarray((par.cue_size+par.L+par.delay, n_set))
    sorted_ticklabels_t =  np.empty((par.cue_size+par.L+par.delay, n_set), dtype=object)
    
    for idx_t in range(par.L):

        if idx_t < par.L:
            T=idx_t
        else:
            T=par.L-1
        
        if par.clustering == 'class':
            idx = np.arange(len(tokens))
            ticklabels = np.array(types).astype(int)

        elif par.clustering == 'timestep':
            idx = np.argsort(tokens[:,T])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted[:,:T+1].astype(str)]  

        elif par.clustering == 'hierarchical':
            idx = np.lexsort(tokens.T[::-1])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted[:,:T+1].astype(str)]  

        elif par.clustering == 'transitions':
            if idx_t == 0:
                tokens_ = tokens[:,idx_t].reshape(-1,1)
            else:
                tokens_ = tokens[:,idx_t-1:idx_t+1]
            idx = np.lexsort(tokens_.T[::-1])
            tokens_sorted=tokens_[idx]
            ticklabels=["".join(token) for token in tokens_sorted]

        elif par.clustering == 'distance':
            pairwise_distances = sch.distance.pdist(M)
            linkage = sch.linkage(pairwise_distances, method='complete')
            cluster_distance_threshold = pairwise_distances.max()/2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
            idx = np.argsort(idx_to_cluster_array)
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted[:,:T+1].astype(str)]  

        M_sorted = M[idx_t].copy()
        M_sorted = np.take(M_sorted, idx, axis=0) 
        M_sorted = np.take(M_sorted, idx, axis=1)

        M_t[idx_t] = M_sorted
        sorted_idx_t[idx_t] = idx
        sorted_ticklabels_t[idx_t] = ticklabels
    
    for idx_t in range(par.L, par.cue_size+par.L+par.delay):
        M_sorted = M[idx_t].copy()
        M_sorted = np.take(M_sorted, idx, axis=0) 
        M_sorted = np.take(M_sorted, idx, axis=1)

        M_t[idx_t] = M_sorted
        sorted_idx_t[idx_t] = idx
        sorted_ticklabels_t[idx_t] = ticklabels

    return M_t, sorted_ticklabels_t


def plot_overlap_sorted(M_sorted, ticklabels, filename, task):
    fig, ax = plt.subplots(2, (par.cue_size+par.L+par.delay)//2, figsize=[10, 4])
    ax = ax.ravel()
    for idx_t in range(par.cue_size+par.L+par.delay):
        im=ax[idx_t].imshow(M_sorted[idx_t])        
        ax[idx_t].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False) 
        ax[idx_t].set_xticks([])#np.arange(len(ticklabels[idx_t])))
        ax[idx_t].set_xticklabels([])#ticklabels[idx_t], rotation=60, size=4)            
        ax[idx_t].set_yticks([])#np.arange(len(ticklabels[idx_t])))
        ax[idx_t].set_yticklabels([])#ticklabels[idx_t], size=4)
        ax[idx_t].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(f"{par.folder}/figs/{whichmatrix}_{filename}_whichclustering{par.clustering}_{par.whichset}_task{task}.svg", bbox_inches="tight", dpi=600)

def compute_overlap_AE(M, tokens, types):
    n_set=len(tokens)
    tokens = np.array([list(token) for token in tokens])

    M_t = np.ndarray((par.cue_size+par.L+par.delay, n_set, n_set))
    sorted_idx_t =  np.ndarray((par.cue_size+par.L+par.delay, n_set))
    sorted_ticklabels_t =  np.empty((par.cue_size+par.L+par.delay, n_set), dtype=object)

    for idx_t in range(par.L):
        
        if par.clustering == 'class':
            idx = np.arange(len(tokens))
            ticklabels = np.array(types).astype(int)

        elif par.clustering == 'hierarchical':
            idx = np.lexsort(tokens.T[::-1])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted.astype(str)]  

        elif par.clustering == 'distance':
            pairwise_distances = sch.distance.pdist(M)
            linkage = sch.linkage(pairwise_distances, method='complete')
            cluster_distance_threshold = pairwise_distances.max()/2
            idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
            idx = np.argsort(idx_to_cluster_array)
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted.astype(str)]  

        elif par.clustering == 'timestep':
            idx = np.argsort(tokens[:,idx_t])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted]

        elif par.clustering == 'transitions':
            if idx_t == 0:
                tokens = tokens[:,idx_t].reshape(-1,1)
            else:
                tokens = tokens[:,idx_t-1:idx_t+1]
            idx = np.lexsort(tokens.T[::-1])
            tokens_sorted=tokens[idx]
            ticklabels=["".join(token) for token in tokens_sorted[:,:T+1].astype(str)]  

        M_sorted = np.take(M, idx, axis=0)
        M_sorted = np.take(M_sorted, idx, axis=1)
        M_t[idx_t] = M_sorted
        sorted_idx_t[idx_t] = idx
        sorted_ticklabels_t[idx_t] = ticklabels

    return M_t, sorted_ticklabels_t

def plot_overlap_AE(M_t, ticklabels_t, filename, task):
    fig, ax = plt.subplots(1, par.L, figsize=[10, 2.5])
    ax = ax.ravel()

    for idx_t in range(par.L):
        im=ax[idx_t].imshow(M_t[idx_t])
        
        # this only works with whichclustering=class
        ax[idx_t].set_xticks(np.arange(len(ticklabels_t[idx_t])))
        ax[idx_t].set_xticklabels(ticklabels_t[idx_t], rotation=45, size=4)            
        ax[idx_t].set_yticks(np.arange(len(ticklabels_t[idx_t])))
        ax[idx_t].set_yticklabels(ticklabels_t[idx_t], size=4)            

        ax[idx_t].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    
    fig.tight_layout()
    fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.046, pad=0.04)
    fig.savefig(f"{par.folder}/figs/{whichmatrix}_{filename}_whichclustering{par.clustering}_{par.whichset}_task{task}.svg", bbox_inches="tight", dpi=600)


def to_dict(d):
    '''
    Converts a defaultdict with any depth into a standard dictionary
    '''
    if isinstance(d, defaultdict):
        return {k: to_dict(v) for k, v in d.items()}
    return d

if __name__ == "__main__":

    exit()