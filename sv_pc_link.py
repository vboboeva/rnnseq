"""
How do the singular vectors (SVs) of the recurrent weight matrix relate to the principal
components (PCs) of the last-timestep activity, and to the transition structure of the
sequences?

Setting (outBiasTrue, N=160, 7 classes, cue_size=4, L=0): a sequence is 4 letters over {a,b}
(renamed to arbitrary letters). Its three transitions are
    t_0 = (l_0 != l_1)   "first"
    t_1 = (l_1 != l_2)   "previous"
    t_2 = (l_2 != l_3)   "latest"      (same = 0, different = 1)
and the 7 classes are exactly the 7 non-constant combinations of these three bits.

Everything reported is gauge-free: the sign of every SV and PC is arbitrary, so only
squares, absolute values and subspaces are used. SVs and PCs live in the same unit space
of the same network, so their overlaps are meaningful per simulation (but NOT across sims).

Statistics
  eta2(axis, label)   squared correlation between the coordinate along an axis and a
                      binary label (= between-group variance share)
  best split          with 7 classes there are only C(7,3)=35 distinct 3-vs-4 partitions,
                      so a single-label permutation p cannot go below 1/35. Instead we
                      enumerate all 35 splits per axis: which split separates the classes
                      best along that axis, and what rank does each labelled split have?
  eta_class           share of an axis' variance that is between classes (7-way)
  overlap             |cos(SV_i, PC_j)|, the position of each PC in the SV spectrum,
                      principal angles between span(SV_1..r) and span(PC_1..r), and the
                      variance of the activity along each SV vs each PC.

Usage
  python sv_pc_link.py --selftest
  python sv_pc_link.py --data-dir outBiasTrue/<folder> [--out DIR] [--K 10]

Plan: docs/plans/2026-11-04_sv-pc-link.md
"""
import argparse
import csv
import glob
import hashlib
import itertools
import os
import pickle
import re
from os.path import join

import numpy as np

LABELS = ('first', 'previous', 'latest')
C_OBS, C_NULL, C_ACC = '#2a6fbb', '#8a8a8a', '#c4452d'
C_SAME, C_DIFF = '#1b9e77', '#d95f02'
C_LABEL = {'first': '#7570b3', 'previous': '#e7298a', 'latest': '#66a61e', 'other': '#b0b0b0'}
C_CLASS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#17becf']


# ----------------------------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------------------------
def canon(token):
    """Rename letters by order of first appearance: 'cddc' -> 'abba'."""
    m = {}
    for ch in token:
        m.setdefault(ch, 'abcdefghijklmnop'[len(m)])
    return ''.join(m[ch] for ch in token)


def transition_bits(tokens):
    """(n_seq, len-1) array, 1 where consecutive letters differ. Column k = letters k,k+1."""
    return np.array([[int(t[k] != t[k + 1]) for k in range(len(t) - 1)] for t in tokens])


def class_setup(tokens):
    """Class ids, label table and the enumeration of class splits (see module docstring)."""
    bits = transition_bits(tokens)
    names = [canon(t) for t in tokens]
    uniq = sorted(set(names))
    cid = np.array([uniq.index(nm) for nm in names])
    C = len(uniq)
    tbl = np.stack([bits[cid == c][0] for c in range(C)])
    assert all(np.all(bits[cid == c] == tbl[c]) for c in range(C)), 'bits not constant within class'
    assert len({tuple(r) for r in tbl}) == C, 'transition bits do not identify the class'
    w = np.bincount(cid) / len(cid)
    half = C // 2
    splits = np.array([[1.0 if c in sub else 0.0 for c in range(C)]
                       for sub in itertools.combinations(range(C), half)])

    def split_index(ind):
        ind = np.asarray(ind, float)
        if ind.sum() > half:
            ind = 1 - ind
        assert ind.sum() == half, 'label is not a half-split of the classes'
        return int(np.where((splits == ind).all(1))[0][0])

    return dict(bits=bits, names=uniq, cid=cid, tbl=tbl, w=w, splits=splits,
                label_idx=[split_index(tbl[:, k]) for k in range(tbl.shape[1])])


# ----------------------------------------------------------------------------------------
# data access
# ----------------------------------------------------------------------------------------
def _sim_ids(paths):
    return [int(re.search(r'sim(\d+)', os.path.basename(p)).group(1)) for p in paths]


def _load_one(args):
    path, token_set = args
    H = pickle.load(open(path, 'rb'))['HiddenAct']
    toks = sorted(t for t in H if token_set[t] == 'train')
    ep = max(H[toks[0]].keys())                       # last saved epoch
    X = np.stack([np.asarray(H[t][ep]) for t in toks])
    if X.ndim == 4:                                   # (n, T, 1, N) -> (n, T, N)
        X = X[:, :, 0]
    return toks, ep, X.astype(np.float32)


def extract_activity(data_dir, cache_path, workers=8):
    """X (n_sim, n_seq, T, N) float32 at the last saved epoch, train tokens, all timesteps.
    The data directory is only read; the derived array is cached at `cache_path`."""
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        return dict(X=z['X'], tokens=[str(t) for t in z['tokens']], sims=z['sims'].tolist(),
                    epoch=int(z['epoch']))
    from concurrent.futures import ProcessPoolExecutor
    token_set = pickle.load(open(join(data_dir, 'token_to_set_classcomb0.pkl'), 'rb'))
    paths = sorted(glob.glob(join(data_dir, 'results_task*_classcomb0_sim*.pkl')))
    assert paths, f'no results in {data_dir}'
    print(f'extracting activity from {len(paths)} results files (slow over NFS) ...')
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_load_one, [(p, token_set) for p in paths]))
    tokens = out[0][0]
    assert all(o[0] == tokens for o in out), 'token order differs across sims'
    assert len({o[1] for o in out}) == 1, 'saved-epoch keys differ across sims'
    X = np.stack([o[2] for o in out])
    sims = _sim_ids(paths)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    np.savez(cache_path, X=X, tokens=np.array(tokens), sims=np.array(sims), epoch=out[0][1])
    return dict(X=X, tokens=tokens, sims=sims, epoch=out[0][1])


def load_weights(data_dir):
    """Recurrent weights (n_sim, N, N), in the order of the sorted model_state files."""
    paths = sorted(glob.glob(join(data_dir, 'model_state_classcomb0_sim*.pth')))
    cache = join(data_dir, 'rec_weights_classcomb0.npy')
    if os.path.exists(cache):
        return np.load(cache), _sim_ids(paths)
    import torch
    Ws = []
    for p in paths:
        m = torch.load(p, map_location='cpu')
        Ws.append((m['h2h.weight'] if 'h2h.weight' in m else m['weight']).numpy())
    return np.stack(Ws), _sim_ids(paths)


# ----------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------
def corr2(Z, L):
    """Squared Pearson correlation between every column of Z and every column of L."""
    Zc, Lc = Z - Z.mean(0), L - L.mean(0)
    num = (Zc.T @ Lc) ** 2
    den = (Zc ** 2).sum(0)[:, None] * (Lc ** 2).sum(0)[None, :]
    return num / np.maximum(den, 1e-30)


def axis_stats(Z, ctx):
    """For centred coordinates Z (n, A): r2 of every class split (n_split, A) and the
    between-class variance share per axis (A,). Class-level, exact for labels that are
    constant within a class."""
    cid, w, S = ctx['cid'], ctx['w'], ctx['splits']
    C = len(w)
    Mc = np.stack([Z[cid == c].mean(0) for c in range(C)])
    varz = np.maximum(Z.var(0), 1e-30)
    Sc = S - (S @ w)[:, None]
    vs = (Sc ** 2) @ w
    cov = (Sc * w[None, :]) @ Mc
    r2 = cov ** 2 / (vs[:, None] * varz[None, :])
    ecls = (w[:, None] * Mc ** 2).sum(0) / varz
    return r2, ecls


def label_stats(r2, ctx):
    """eta2 (A,3), rank of each labelled split among all splits (A,3), best split (A,)."""
    idx = ctx['label_idx']
    obs = r2[idx]                                                    # (3, A)
    rank = 1 + (r2[None, :, :] > obs[:, None, :] + 1e-12).sum(1)     # (3, A)
    return obs.T, rank.T, r2.argmax(0)


def analyze_sim(svd, X_last, ctx, K=10):
    """All per-simulation statistics. `svd` = (U, S, Vh) of the recurrent weights."""
    U, S, Vh = svd
    n, N = X_last.shape
    mu = X_last.mean(0)
    Xc = X_last - mu
    _, sp, P = np.linalg.svd(Xc, full_matrices=False)               # centred PCA, P (A, N)
    A = P.shape[0]
    totvar = (Xc ** 2).sum() / n
    Zs, Zp = Xc @ Vh.T, Xc @ P.T
    out = {}

    # --- how the two bases relate -----------------------------------------------------
    out['cos'] = np.abs(Vh[:K] @ P[:K].T)                            # (K,K) SV rows, PC cols
    out['cos_U'] = np.abs(U[:, :K].T @ P[:K].T)                      # left-vector control
    Q = 4
    out['mass'] = np.cumsum((Vh @ P[:Q].T) ** 2, axis=0)             # (N, Q): PC_j mass in top-r SVs
    pa = np.full((K, K), np.nan)
    sub = np.zeros(K)
    for r in range(1, K + 1):
        sv = np.linalg.svd(Vh[:r] @ P[:r].T, compute_uv=False)
        pa[r - 1, :r] = sv
        sub[r - 1] = np.mean(sv ** 2)
    out['pa'], out['subspace'] = pa, sub
    out['var_sv'] = Zs.var(0) / totvar                               # (N,)
    out['var_pc'] = sp ** 2 / n / totvar                             # (A,)
    # controls for the centring choice
    out['mean_share'] = float(mu @ mu / np.mean((X_last ** 2).sum(1)))
    mh = mu / max(np.linalg.norm(mu), 1e-30)
    out['cos_mean'] = np.abs(Vh[:3] @ mh)

    # --- what each axis encodes ---------------------------------------------------------
    for tag, Z in (('sv', Zs), ('pc', Zp)):
        r2, ecls = axis_stats(Z, ctx)
        eta, rank, best = label_stats(r2, ctx)
        out[f'eta_{tag}'], out[f'rank_{tag}'], out[f'best_{tag}'], out[f'ecls_{tag}'] = eta, rank, best, ecls
    return out


def time_resolved(svd, X, ctx, Kt=5):
    """eta2(SV_k coordinate at time t, lag-l transition) for t=1..T-1, lag l=0..t-1
    (lag l at time t is the transition between letters t-l-1 and t-l). Returns a list
    indexed by t-1 of (Kt, t) arrays."""
    Vh = svd[2]
    T = X.shape[1]
    out = []
    for t in range(1, T):
        Z = (X[:, t] - X[:, t].mean(0)) @ Vh[:Kt].T
        L = ctx['bits'][:, [t - 1 - l for l in range(t)]].astype(float)
        out.append(corr2(Z, L))
    return out


def diagonal_test(E):
    """E (Kt, n_lag) mean eta2. Statistic: sum_l E[l, l] (lag l on SV l+1). Exact null: all
    injective assignments of lags to SV indices. Returns (D, rank_of_identity, n_maps,
    best_map, D_best)."""
    Kt, nl = E.shape
    D = sum(E[l, l] for l in range(nl))
    maps = list(itertools.permutations(range(Kt), nl))
    vals = np.array([sum(E[m[l], l] for l in range(nl)) for m in maps])
    best = maps[int(vals.argmax())]
    return D, 1 + int((vals > D + 1e-12).sum()), len(maps), best, float(vals.max())


# ----------------------------------------------------------------------------------------
# aggregation and reporting
# ----------------------------------------------------------------------------------------
def _frac_table(winners, n_bins):
    """Fraction of sims whose winner is axis 0..n_bins-1; the last column is 'beyond'."""
    out = np.zeros(n_bins + 1)
    for k in range(n_bins):
        out[k] = np.mean(winners == k)
    out[n_bins] = np.mean(winners >= n_bins)
    return out


def summarize(res, ctx, K=10, show=6):
    n, N = len(res), res[0]['eta_sv'].shape[0]
    g = lambda key: np.stack([r[key] for r in res])
    s = dict(n=n, N=N, K=K, show=show)
    s['eta_sv'], s['eta_pc'] = g('eta_sv'), g('eta_pc')                     # (n, N, 3)
    s['rank_sv'], s['rank_pc'] = g('rank_sv'), g('rank_pc')
    s['best_sv'], s['best_pc'] = g('best_sv'), g('best_pc')                 # (n, N)
    s['ecls_sv'], s['ecls_pc'] = g('ecls_sv'), g('ecls_pc')
    s['cos'], s['cos_U'] = g('cos'), g('cos_U')
    s['mass'], s['pa'], s['subspace'] = g('mass'), g('pa'), g('subspace')
    s['var_sv'], s['var_pc'] = g('var_sv'), g('var_pc')
    s['mean_share'], s['cos_mean'] = g('mean_share'), g('cos_mean')
    s['win_sv'] = s['eta_sv'].argmax(1)                                      # (n, 3): winning SV per label
    s['win_pc'] = s['eta_pc'][:, :N].argmax(1)
    lab = ctx['label_idx']
    for tag in ('sv', 'pc'):                                                 # which labelled split is each axis' best?
        b = np.full(s[f'best_{tag}'].shape, -1)                              # -1 = some other 3-vs-4 split
        for l, idx in enumerate(lab):
            b[s[f'best_{tag}'] == idx] = l
        s[f'best_label_{tag}'] = b
    return s


def print_summary(s):
    n, N, K, m = s['n'], s['N'], s['K'], s['show']
    f = lambda a: np.array2string(a, precision=2, suppress_small=True)
    print(f'\n===== {n} sims, N={N}; centred PCs of last-timestep activity; SV = right singular vectors =====')
    for tag in ('sv', 'pc'):
        nm = tag.upper()
        print(f'\nmean eta2 of {nm}_1..{m} (rows) with first / previous / latest (cols):')
        print(f(s[f'eta_{tag}'][:, :m].mean(0)))
        print(f'median:\n{f(np.median(s[f"eta_{tag}"][:, :m], 0))}')
        print(f'between-class variance share of {nm}_1..{m}: {f(s[f"ecls_{tag}"][:, :m].mean(0))}')
        print(f'winning {nm} index per label (fraction of sims at {nm}_1..{m}, then beyond {m}):')
        for l, name in enumerate(LABELS):
            w = s[f'win_{tag}'][:, l]
            print(f'  {name:9s}', f(_frac_table(w, m)), f'  (median winner index {np.median(w) + 1:.0f})')
        print(f'best class split along {nm}_1..{m} -> fraction of sims where it IS first/previous/latest/other:')
        for k in range(m):
            b = s[f'best_label_{tag}'][:, k]
            fr = [np.mean(b == l) for l in range(3)] + [np.mean(b == -1)]
            print(f'  {nm}_{k + 1}:', f(np.array(fr)),
                  f'  mean ranks of labelled splits (1 = best of 35): {f(s[f"rank_{tag}"][:, k].mean(0))}')

    print(f'\nmean |cos(SV_i, PC_j)|, i (rows) j (cols) <= {m}; chance 1/sqrt(N) = {1 / np.sqrt(N):.3f}:')
    print(f(s['cos'][:, :m, :m].mean(0)))
    print(f'  with LEFT singular vectors (control):\n{f(s["cos_U"][:, :m, :m].mean(0))}')
    print('mass of PC_j (cols PC1..4) in the top-r SVs, cumulative (rows r = 1,3,5,10,20,40), chance r/N:')
    rr = [1, 3, 5, 10, 20, 40]
    print(f(np.stack([s['mass'][:, r - 1, :].mean(0) for r in rr])), ' chance', f(np.array(rr) / N))
    print(f'principal-angle overlap mean(cos^2) of span(SV_1..r), span(PC_1..r), r=1..{K}: {f(s["subspace"].mean(0))}')
    print(f'  chance r/N: {f(np.arange(1, K + 1) / N)}')
    print(f'variance fraction along SV_1..{m}: {f(s["var_sv"][:, :m].mean(0))}')
    print(f'variance fraction along PC_1..{m}: {f(s["var_pc"][:, :m].mean(0))}')
    cs, cp = np.cumsum(s['var_sv'][:, :K], 1).mean(0), np.cumsum(s['var_pc'][:, :K], 1).mean(0)
    print(f'cumulative variance, top-r SVs: {f(cs)}\n                     top-r PCs: {f(cp)}  (upper bound)')
    print(f'centring control: mean activity carries {s["mean_share"].mean():.3f} of mean |h|^2; '
          f'|cos(SV_1..3, mean direction)| = {f(s["cos_mean"].mean(0))}')


def lag_ranks(E):
    """For each lag l: rank (1 = highest) of the predicted SV_{l+1} among the Kt SVs, by eta2.
    Needed because the summed diagonal statistic can be carried by the first lags alone."""
    return [1 + int((E[:, l] > E[l, l] + 1e-12).sum()) for l in range(E.shape[1])]


def print_time(E_by_t, per_sim_win=None):
    print('\ntime-resolved: mean eta2 of SV_1..Kt (rows) with lag0 (latest so far) .. lag(t-1) (cols)')
    rows = []
    for t, E in enumerate(E_by_t, start=1):
        D, rank, nm, best, Db = diagonal_test(E)
        lr = lag_ranks(E)
        print(f'  t={t}:\n{np.array2string(E, precision=2, suppress_small=True)}')
        print(f'     diagonal (lag l on SV_l+1) sum = {D:.2f}; rank of identity among {nm} assignments = {rank}'
              f' (1 = best); best assignment lag->SV {[b + 1 for b in best]} sum = {Db:.2f}')
        print(f'     per-lag rank of the predicted SV among the {E.shape[0]} SVs (1 = highest): {lr}')
        rows.append((t, D, rank, nm, Db, lr))
    return rows


def write_csv(s, ctx, path):
    m = s['show']
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'axis', 'var_frac', 'eta_class'] + [f'eta_{l}' for l in LABELS] +
                   [f'frac_win_{l}' for l in LABELS] + [f'frac_bestsplit_{l}' for l in LABELS] + ['frac_bestsplit_other'])
        for tag in ('sv', 'pc'):
            for k in range(m):
                b = s[f'best_label_{tag}'][:, k]
                w.writerow([tag.upper(), k + 1, f'{s[f"var_{tag}"][:, k].mean():.4g}',
                            f'{s[f"ecls_{tag}"][:, k].mean():.4g}'] +
                           [f'{s[f"eta_{tag}"][:, k, l].mean():.4g}' for l in range(3)] +
                           [f'{np.mean(s[f"win_{tag}"][:, l] == k):.3f}' for l in range(3)] +
                           [f'{np.mean(b == l):.3f}' for l in range(3)] + [f'{np.mean(b == -1):.3f}'])


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _plt():
    import matplotlib
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def _clean(axs):
    for a in np.ravel(axs):
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)


def plot_scatter(sim_data, ctx, out_dir, fname='svpc_scatter.svg'):
    """sim_data: list of (sim_id, Zs (n,3), Zp (n,3)). SV planes coloured by transition,
    PC planes coloured by the 7 classes."""
    plt = _plt()
    from matplotlib.lines import Line2D
    bits, cid = ctx['bits'], ctx['cid']
    nr = len(sim_data)
    fig, axs = plt.subplots(nr, 6, figsize=(17, 2.9 * nr), squeeze=False)
    for r, (sid, Zs, Zp) in enumerate(sim_data):
        specs = [(Zs, 0, 1, 'SV', 'latest'), (Zs, 0, 1, 'SV', 'previous'), (Zs, 1, 2, 'SV', 'previous'),
                 (Zp, 0, 1, 'PC', 'class'), (Zp, 0, 2, 'PC', 'class'), (Zp, 1, 2, 'PC', 'class')]
        for c, (Z, i, j, nm, col) in enumerate(specs):
            ax = axs[r, c]
            if col == 'class':
                colors = [C_CLASS[k] for k in cid]
            else:
                colors = [C_DIFF if b else C_SAME for b in bits[:, LABELS.index(col)]]
            ax.scatter(Z[:, i], Z[:, j], c=colors, s=5, alpha=.55, lw=0)
            ax.set_xlabel(f'{nm}{i + 1}'); ax.set_ylabel(f'{nm}{j + 1}')
            ax.set_title(f'sim {sid}: {nm}{i + 1}-{nm}{j + 1}, colour = {col}', fontsize=8)
    h_t = [Line2D([], [], marker='o', ls='', c=C_SAME, label='same (AA)'),
           Line2D([], [], marker='o', ls='', c=C_DIFF, label='different (AB)')]
    h_c = [Line2D([], [], marker='o', ls='', c=C_CLASS[k], label=ctx['names'][k]) for k in range(len(ctx['names']))]
    axs[0, 2].legend(handles=h_t, fontsize=6.5, frameon=False, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)
    axs[0, 5].legend(handles=h_c, fontsize=6.5, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    _clean(axs); fig.tight_layout()
    fig.savefig(join(out_dir, fname), dpi=200, bbox_inches='tight'); plt.close(fig)


def plot_overlap(s, out_dir, fname='svpc_overlap.svg'):
    plt = _plt()
    N, K = s['N'], s['K']
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    ax = axs[0]
    im = ax.imshow(s['cos'][:, :8, :8].mean(0), cmap='Blues', vmin=0, interpolation='nearest')
    ax.set_xlabel('PC j'); ax.set_ylabel('SV i'); fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    ax.set_title(r'(a) mean $|\cos(SV_i,PC_j)|$, chance %.2f' % (1 / np.sqrt(N)), fontsize=9)
    ax.set_xticks(range(8)); ax.set_xticklabels(range(1, 9)); ax.set_yticks(range(8)); ax.set_yticklabels(range(1, 9))

    ax = axs[1]
    r = np.arange(1, N + 1)
    for j in range(s['mass'].shape[2]):
        ax.plot(r, s['mass'][:, :, j].mean(0), lw=1.6, label=f'PC{j + 1}', color=plt.get_cmap('Blues')(.35 + .2 * (3 - j) / 3 + .25))
    ax.plot(r, r / N, c=C_NULL, ls='--', label='chance r/N')
    ax.set_xscale('log'); ax.set_xlabel('r  (top-r SVs)'); ax.set_ylabel(r'cumulative share of $PC_j$')
    ax.set_title('(b) where each PC sits in the SV spectrum', fontsize=9); ax.legend(fontsize=6.5, frameon=False)

    ax = axs[2]
    rk = np.arange(1, K + 1)
    ax.fill_between(rk, np.percentile(s['subspace'], 5, 0), np.percentile(s['subspace'], 95, 0), color=C_OBS, alpha=.2, lw=0)
    ax.plot(rk, s['subspace'].mean(0), c=C_OBS, lw=2, marker='o', ms=3, label='observed')
    ax.plot(rk, rk / N, c=C_NULL, ls='--', label='chance r/N')
    ax.set_xlabel('r'); ax.set_ylabel(r'mean $\cos^2$ of principal angles')
    ax.set_title('(c) span(SV$_{1..r}$) vs span(PC$_{1..r}$)', fontsize=9); ax.legend(fontsize=6.5, frameon=False)

    ax = axs[3]
    k = np.arange(1, 11)
    for key, c, lab in (('var_sv', C_ACC, 'along SV_k'), ('var_pc', C_OBS, 'along PC_k')):
        v = s[key][:, :10]
        ax.fill_between(k, np.percentile(v, 5, 0), np.percentile(v, 95, 0), color=c, alpha=.15, lw=0)
        ax.plot(k, v.mean(0), c=c, lw=2, marker='o', ms=3, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('k'); ax.set_ylabel('fraction of activity variance')
    ax.set_title('(d) variance along each axis', fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    _clean(axs); fig.tight_layout(w_pad=2.5)
    fig.savefig(join(out_dir, fname), dpi=200); plt.close(fig)


def plot_decoding(s, out_dir, fname='svpc_decoding.svg'):
    plt = _plt()
    m = s['show']
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for c, tag in enumerate(('sv', 'pc')):
        ax = axs[0, c]
        im = ax.imshow(s[f'eta_{tag}'][:, :m].mean(0), cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(3)); ax.set_xticklabels(LABELS); ax.set_yticks(range(m)); ax.set_yticklabels(range(1, m + 1))
        ax.set_ylabel(tag.upper() + ' index'); ax.set_title(f'({"ab"[c]}) mean $\\eta^2$ of {tag.upper()} coordinate vs label', fontsize=9)
        for i in range(m):
            for j in range(3):
                v = s[f'eta_{tag}'][:, i, j].mean()
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=7, color='w' if v > .5 else 'k')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    ax = axs[0, 2]
    k = np.arange(1, m + 1)
    for tag, c, lab in (('sv', C_ACC, 'SV_k'), ('pc', C_OBS, 'PC_k')):
        ax.plot(k, s[f'ecls_{tag}'][:, :m].mean(0), c=c, lw=2, marker='o', ms=4, label=f'{lab}: between-class share')
    ax.set_ylim(0, 1); ax.set_xlabel('axis index k'); ax.set_ylabel('share of variance between the 7 classes')
    ax.set_title('(c) how class-structured is each axis', fontsize=9); ax.legend(fontsize=6.5, frameon=False)

    for c, tag in enumerate(('sv', 'pc')):
        ax = axs[1, c]
        x = np.arange(m + 1)
        for l, name in enumerate(LABELS):
            ax.bar(x + (l - 1) * .27, _frac_table(s[f'win_{tag}'][:, l], m), .27, color=C_LABEL[name], label=name)
        ax.set_xticks(x); ax.set_xticklabels(list(range(1, m + 1)) + [f'>{m}'])
        ax.set_xlabel(f'{tag.upper()} index with the largest $\\eta^2$'); ax.set_ylabel('fraction of sims')
        ax.set_title(f'({"de"[c]}) winning {tag.upper()} per label', fontsize=9); ax.legend(fontsize=6.5, frameon=False)
    ax = axs[1, 2]
    cols = [('sv', k) for k in range(4)] + [('pc', k) for k in range(4)]
    bottoms = np.zeros(len(cols))
    for l, name in enumerate(list(LABELS) + ['other']):
        vals = np.array([np.mean(s[f'best_label_{t}'][:, k] == (l if l < 3 else -1)) for t, k in cols])
        ax.bar(range(len(cols)), vals, bottom=bottoms, color=C_LABEL[name], label=name); bottoms += vals
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'{t.upper()}{k + 1}' for t, k in cols], fontsize=7)
    ax.set_ylabel('fraction of sims'); ax.set_title('(f) best 3-vs-4 class split along the axis', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    _clean(axs); fig.tight_layout(w_pad=2.5)
    fig.savefig(join(out_dir, fname), dpi=200); plt.close(fig)


def plot_time(E_by_t, rows, out_dir, fname='svpc_time.svg'):
    plt = _plt()
    T1 = len(E_by_t)
    fig, axs = plt.subplots(1, T1, figsize=(4.6 * T1, 3.6), squeeze=False)
    for i, E in enumerate(E_by_t):
        ax = axs[0, i]
        im = ax.imshow(E.T, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        t, D, rank, nm, Db, lr = rows[i]
        for l in range(E.shape[1]):
            ax.add_patch(plt.Rectangle((l - .5, l - .5), 1, 1, fill=False, ec=C_ACC, lw=1.5))
        for a in range(E.shape[1]):
            for b in range(E.shape[0]):
                ax.text(b, a, f'{E[b, a]:.2f}', ha='center', va='center', fontsize=7, color='w' if E[b, a] > .5 else 'k')
        ax.set_xticks(range(E.shape[0])); ax.set_xticklabels(range(1, E.shape[0] + 1)); ax.set_xlabel('SV index')
        ax.set_yticks(range(E.shape[1])); ax.set_yticklabels([f'lag{l}' for l in range(E.shape[1])])
        ax.set_title(f'time t={t}: identity rank {rank}/{nm}; per-lag rank of predicted SV {lr}', fontsize=8)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.suptitle(r'mean $\eta^2$(SV coordinate at time t, transition lag l); red = predicted copying diagonal', fontsize=9)
    fig.tight_layout(); fig.savefig(join(out_dir, fname), dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------------
# self test
# ----------------------------------------------------------------------------------------
def _synthetic(seed=0, N=40, per_class=30):
    rng = np.random.default_rng(seed)
    templates = ['abbb', 'abba', 'abab', 'abaa', 'aabb', 'aaba', 'aaab']
    letters = 'cdefghij'
    tokens = []
    for tp in templates:
        for _ in range(per_class):
            x, y = rng.choice(list(letters), 2, replace=False)
            tokens.append(''.join(x if ch == 'a' else y for ch in tp))
    ctx = class_setup(tokens)
    bits = ctx['bits'].astype(float)
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))               # columns = orthonormal v's
    V3 = Q[:, :3]
    X = (3 * (bits[:, 2] - bits[:, 2].mean())[:, None] * V3[:, 0] +
         2 * (bits[:, 1] - bits[:, 1].mean())[:, None] * V3[:, 1] +
         1 * (bits[:, 0] - bits[:, 0].mean())[:, None] * V3[:, 2] + 0.1 * rng.standard_normal((len(tokens), N)))
    Uo, _ = np.linalg.qr(rng.standard_normal((N, N)))
    S = np.concatenate([[10, 8, 6], np.linspace(2, 0.5, N - 3)])
    W = Uo @ np.diag(S) @ Q.T                                       # right singular vectors = Q columns
    return tokens, ctx, X, W


def selftest():
    # 1. labels
    tpls = ['abbb', 'abba', 'abab', 'abaa', 'aabb', 'aaba', 'aaab']
    want = {'abbb': (1, 0, 0), 'abba': (1, 0, 1), 'abab': (1, 1, 1), 'abaa': (1, 1, 0),
            'aabb': (0, 1, 0), 'aaba': (0, 1, 1), 'aaab': (0, 0, 1)}          # (first, previous, latest), 1 = different
    ctx = class_setup([t for t in tpls for _ in range(3)])
    for c, nm in enumerate(ctx['names']):
        assert tuple(ctx['tbl'][c]) == want[nm], (nm, ctx['tbl'][c])
    assert canon('cddc') == 'abba' and len({tuple(r) for r in ctx['tbl']}) == 7
    C = np.corrcoef(ctx['tbl'].T)
    assert np.allclose(C[np.triu_indices(3, 1)], -1 / 6), C
    assert ctx['splits'].shape == (35, 7)

    # 2. class-level eta2 equals the sequence-level squared correlation
    tokens, ctx, X, W = _synthetic()
    svd = np.linalg.svd(W)
    Xc = X - X.mean(0)
    Zs = Xc @ svd[2].T
    r2, ecls = axis_stats(Zs, ctx)
    eta, rank, best = label_stats(r2, ctx)
    ref = corr2(Zs, ctx['bits'].astype(float))
    assert np.allclose(eta, ref), np.abs(eta - ref).max()

    # 3. synthetic recovery: SV1 <-> latest, SV2 <-> previous, SV3 <-> first
    assert eta[0, 2] > .95 and eta[1, 1] > .9 and eta[2, 0] > .6, eta[:3]
    assert eta[0, :2].max() < .05 and eta[1, [0, 2]].max() < .05
    assert rank[0, 2] == 1 and rank[1, 1] == 1 and best[0] == ctx['label_idx'][2]
    res = analyze_sim(svd, X, ctx, K=6)
    assert np.allclose(np.diag(res['cos'])[:3], 1, atol=.05), np.diag(res['cos'])[:3]   # PCs ~ SVs here
    assert res['var_pc'][0] >= res['var_pc'][1] >= res['var_pc'][2]

    # 4. gauge invariance: random sign flips of SVs, and a rotation of the unit coordinates
    flip = np.random.default_rng(1).choice([-1., 1.], size=W.shape[0])
    svd_f = (svd[0] * flip[None, :], svd[1], svd[2] * flip[:, None])
    res_f = analyze_sim(svd_f, X, ctx, K=6)
    Qr, _ = np.linalg.qr(np.random.default_rng(2).standard_normal(W.shape))
    res_r = analyze_sim(np.linalg.svd(Qr @ W @ Qr.T), X @ Qr.T, ctx, K=6)
    for other in (res_f, res_r):
        for key in ('cos', 'cos_U', 'mass', 'pa', 'subspace', 'var_sv', 'var_pc', 'eta_sv', 'eta_pc',
                    'rank_sv', 'rank_pc', 'ecls_sv', 'ecls_pc'):
            assert np.allclose(res[key], other[key], atol=1e-6, equal_nan=True), key
        assert np.array_equal(res['best_sv'], other['best_sv']) and np.array_equal(res['best_pc'], other['best_pc'])

    # 5. time-resolved: lag bookkeeping and the exact diagonal null
    Xt = np.zeros((len(tokens), 4, W.shape[0]))
    bits = ctx['bits'].astype(float)
    Q2 = svd[2]
    for t in range(1, 4):                      # lag l at time t sits on SV l+1 by construction
        for l in range(t):
            Xt[:, t] += 2.0 * (bits[:, t - 1 - l] - .5)[:, None] * Q2[l]
    Xt += 0.05 * np.random.default_rng(3).standard_normal(Xt.shape)
    E = time_resolved(svd, Xt, ctx, Kt=5)
    for t, Et in enumerate(E, start=1):
        assert np.all(np.diag(Et) > .9), (t, Et)
        D, rk, nm, best, _ = diagonal_test(Et)
        assert rk == 1 and list(best) == list(range(t)) and nm == int(np.prod(range(5, 5 - t, -1))), (t, rk, nm, best)
    print('selftest OK: labels, class-level eta2 == sequence-level corr, synthetic recovery, '
          'sign/rotation invariance, lag bookkeeping + exact diagonal null')


# ----------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------
def run(data_dir, out_dir, K=10, sims_plot=(0, 1), time=True, workers=8):
    os.makedirs(out_dir, exist_ok=True)
    tag = hashlib.md5(os.path.basename(os.path.normpath(data_dir)).encode()).hexdigest()[:8]
    here = os.path.dirname(os.path.abspath(__file__))
    act = extract_activity(data_dir, join(here, 'cache', f'activity_{tag}.npz'), workers)
    Ws, w_sims = load_weights(data_dir)
    assert act['sims'] == w_sims, 'simulation ids of activity and weights differ'
    X = act['X']
    n_sim, n_seq, T, N = X.shape
    print(f'{n_sim} sims, {n_seq} sequences, T={T}, N={N}, epoch key {act["epoch"]}; data: {data_dir}')
    ctx = class_setup(act['tokens'])
    print('classes:', ctx['names'], '\nlabel table (first, previous, latest; 1 = different):\n', ctx['tbl'])

    svds = [np.linalg.svd(W) for W in Ws]
    res = [analyze_sim(svds[i], X[i, :, -1, :].astype(np.float64), ctx, K) for i in range(n_sim)]
    s = summarize(res, ctx, K)
    print_summary(s)
    print('\nreproducibility check (first 4 sims): eta2 of SV_1..4 / PC_1..4 vs [first previous latest]')
    for i in range(min(4, n_sim)):
        print(f' sim index {i} (id {act["sims"][i]}):\n  SV', np.array2string(res[i]['eta_sv'][:4], precision=2, suppress_small=True).replace('\n', '\n    '),
              '\n  PC', np.array2string(res[i]['eta_pc'][:4], precision=2, suppress_small=True).replace('\n', '\n    '))

    rows, E_by_t = [], None
    if time:
        per = [time_resolved(svds[i], X[i].astype(np.float64), ctx, Kt=5) for i in range(n_sim)]
        E_by_t = [np.mean([p[t] for p in per], axis=0) for t in range(T - 1)]
        rows = print_time(E_by_t)
        print('  caveat: the network is trained on the final step only; intermediate-step structure need not match.')

    sim_data = []
    for i in sims_plot:
        if i < n_sim:
            Xc = X[i, :, -1, :].astype(np.float64)
            Xc = Xc - Xc.mean(0)
            P = np.linalg.svd(Xc, full_matrices=False)[2]
            sim_data.append((act['sims'][i], Xc @ svds[i][2][:3].T, Xc @ P[:3].T))
    plot_scatter(sim_data, ctx, out_dir)
    plot_overlap(s, out_dir)
    plot_decoding(s, out_dir)
    if time:
        plot_time(E_by_t, rows, out_dir)
    write_csv(s, ctx, join(out_dir, 'svpc_axis_summary.csv'))
    print('\nfigures + csv written to', out_dir)
    return dict(s=s, ctx=ctx, E_by_t=E_by_t, rows=rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--data-dir')
    ap.add_argument('--out')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--sims-plot', default='0,1')
    ap.add_argument('--no-time', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    if a.data_dir:
        out = a.out or join(os.path.dirname(a.data_dir.rstrip('/')), 'figs', os.path.basename(a.data_dir.rstrip('/')))
        run(a.data_dir, out, K=a.K, sims_plot=tuple(int(x) for x in a.sims_plot.split(',')),
            time=not a.no_time, workers=a.workers)
