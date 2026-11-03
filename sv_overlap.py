"""
Gauge-invariant statistics for the overlap between left and right singular vectors
of the recurrent weight matrix, W = U S Vh, across many simulations.

Why this module exists
----------------------
The SVD is defined only up to per-mode sign flips (u_k, v_k) -> (-u_k, -v_k), so
M = Vh @ U transforms as M[i,j] -> d_i d_j M[i,j]. Averaging M[i,j] across sims
therefore cancels by construction. Everything below is a gauge-invariant statistic,
computed PER SIM first and aggregated afterwards.

Headline statistic: the rank profile. For a column j, a_(k) = k-th largest of
|v_i . u_j| over ALL i, within each sim. A cliff between a_(1) and a_(2) in every
individual sim means "u_j has one dominant right-singular partner".

Usage
-----
    python sv_overlap.py --selftest
    python sv_overlap.py --data-dir outBiasTrue/<folder> [--out figs_dir] [--col 0]

Docs: docs/plans/2026-11-02_sv-overlap-stats.md
"""
import argparse
import glob
import os
import pickle
from os.path import join

import numpy as np
from scipy import stats

# ----------------------------------------------------------------------------------
# colours: one hue for observed data, neutral grey for nulls, one accent for emphasis
# ----------------------------------------------------------------------------------
C_OBS = '#2a6fbb'
C_NULL = '#8a8a8a'
C_ACC = '#c4452d'


# ----------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------
def load_rec_weights(data_dir, classcomb=0, cache=True):
    """Load recurrent weights (n_sim, N, N) from model_state_classcomb*_sim*.pth.
    Cached next to the data as a single .npy because torch.load over NFS is slow."""
    cache_path = join(data_dir, f'rec_weights_classcomb{classcomb}.npy')
    if cache and os.path.exists(cache_path):
        return np.load(cache_path)
    import torch
    paths = sorted(glob.glob(join(data_dir, f'model_state_classcomb{classcomb}_sim*.pth')))
    assert paths, f'no models in {data_dir}'
    Ws = []
    for p in paths:
        m = torch.load(p, map_location='cpu')
        Ws.append((m['h2h.weight'] if 'h2h.weight' in m else m['weight']).numpy())
    Ws = np.stack(Ws)
    if cache:
        try:
            np.save(cache_path, Ws)
        except OSError as e:  # read-only data dir
            print(f'could not cache weights: {e}')
    return Ws


def svd_all(Ws):
    """Returns Us (n,N,N), Ss (n,N), Vhs (n,N,N)."""
    out = [np.linalg.svd(W) for W in Ws]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]),
            np.stack([o[2] for o in out]))


# ----------------------------------------------------------------------------------
# core overlap quantities
# ----------------------------------------------------------------------------------
def compute_VU_overlap(Us, Ss, Vhs, n_components=10):
    """M_full[s,i,j] = v_i . u_j for all i (N of them) and the first K columns j.
    M is the KxK corner. All derived quantities are gauge invariant except M itself."""
    Us, Ss, Vhs = np.asarray(Us), np.asarray(Ss), np.asarray(Vhs)
    K = n_components
    M_full = np.einsum('sin,snj->sij', Vhs, Us[:, :, :K])        # (n_sim, N, K)
    M = M_full[:, :K, :]                                          # (n_sim, K, K)
    return dict(M_full=M_full, M=M, absM=np.abs(M),
                cycle2=M * M.transpose(0, 2, 1),                  # M[i,j] * M[j,i]
                T=Ss[:, :K, None] * M,                            # copying gain, W U = U T
                S=Ss, N=Us.shape[1], K=K)


def rank_profile_null(N, n_draws=20000, seed=0, n_ranks=20):
    """Order statistics of |g|/|g| for Gaussian g in R^N, i.e. an exact Haar column."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((n_draws, N))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    a = np.abs(g)
    srt = -np.sort(-a, axis=1)
    p = a ** 2
    return dict(N=N, mean=srt[:, :n_ranks].mean(0), a1=srt[:, 0],
                a1_max=srt[:, 0].max(), a1_q99=np.quantile(srt[:, 0], 0.99),
                a2=srt[:, 1], pr=1.0 / np.sum(p ** 2, axis=1),
                entry_mean=a.mean(),
                thr_bonf=np.quantile(a.ravel(), 1 - 0.05 / N))


def rank_profile(M_full, col=0, null=None):
    """Per-sim sorted overlap profile of u_col with all N right singular vectors."""
    a = np.abs(M_full[:, :, col])                                 # (n_sim, N)
    srt = -np.sort(-a, axis=1)
    pw = a ** 2
    out = dict(col=col, a=a, srt=srt, a1=srt[:, 0], a2=srt[:, 1],
               ratio=srt[:, 0] / srt[:, 1],
               mass1=srt[:, 0] ** 2, cummass=np.cumsum(srt ** 2, axis=1),
               pr=1.0 / np.sum((pw / pw.sum(1, keepdims=True)) ** 2, axis=1),
               winner=a.argmax(1), n_sim=a.shape[0])
    if null is not None:
        out['n_sig'] = (a > null['thr_bonf']).sum(1)
        # compare each order statistic to ITS OWN null distribution (a(2) vs null a(2), not a(1))
        out['a1_above_null_max'] = float((out['a1'] > null['a1'].max()).mean())
        out['a2_above_null_q99'] = float((out['a2'] > np.quantile(null['a2'], 0.99)).mean())
        out['a2_above_null_max'] = float((out['a2'] > null['a2'].max()).mean())
        # one-sided empirical p of each sim's a1 under the Haar-column null
        out['p_a1'] = (1 + (null['a1'][None, :] >= out['a1'][:, None]).sum(1)) / (1 + len(null['a1']))
    return out


def summarize_rank_profile(rp, null):
    n = rp['n_sim']
    ns = rp['n_sig']
    lines = [
        f"column u_{rp['col']+1}  (n_sim={n}, N={null['N']})",
        f"  a(1..5) mean   : {np.round(rp['srt'][:, :5].mean(0), 3)}   null {np.round(null['mean'][:5], 3)}",
        f"  a(1) min/5pct  : {rp['a1'].min():.3f} / {np.percentile(rp['a1'], 5):.3f}   null max {null['a1_max']:.3f}",
        f"  a(1)>null a(1) max {null['a1'].max():.3f}: {rp['a1_above_null_max']*100:.0f}% of sims;  "
        f"a(2)>null a(2) q99 {np.quantile(null['a2'], .99):.3f}: {rp['a2_above_null_q99']*100:.0f}%, "
        f"> its max {null['a2'].max():.3f}: {rp['a2_above_null_max']*100:.0f}%",
        f"  top-1 sq. mass : mean {rp['mass1'].mean():.3f}  (cumulative top1/2/3/5: "
        f"{np.round(rp['cummass'][:, [0, 1, 2, 4]].mean(0), 3)})",
        f"  participation  : mean {rp['pr'].mean():.2f}  [5-95pct {np.percentile(rp['pr'], 5):.2f}, "
        f"{np.percentile(rp['pr'], 95):.2f}]   null {null['pr'].mean():.1f}",
        f"  a1/a2          : median {np.median(rp['ratio']):.2f}, <1.3 in {(rp['ratio'] < 1.3).sum()}/{n}, "
        f">2 in {(rp['ratio'] > 2).sum()}/{n}",
        f"  n_sig (>{null['thr_bonf']:.3f}) : counts for 0..5 = {np.bincount(ns, minlength=6)[:6]}  "
        f"(exactly one in {(ns == 1).sum()}/{n})",
        f"  winner index   : {_top_counts(rp['winner'])}",
    ]
    return '\n'.join(lines)


def _top_counts(x, k=5):
    v, c = np.unique(x, return_counts=True)
    o = np.argsort(-c)[:k]
    return ', '.join(f'v{int(v[i])+1}:{int(c[i])}' for i in o)


def subspace_overlap(M_full, col, block):
    """||P_span{v_i, i in block} u_col||. Near-saturated for low-rank W -- not evidence
    of localization; reported once for completeness."""
    return np.sqrt((M_full[:, list(block), col] ** 2).sum(1))


def degeneracy_check(Ss, M_full, col=0, i2=1, i3=2):
    """Does the relative gap (s_i2 - s_i3)/s_i2 explain a(1)/a(2) or the winner?"""
    gap = (Ss[:, i2] - Ss[:, i3]) / Ss[:, i2]
    srt = -np.sort(-np.abs(M_full[:, :, col]), axis=1)
    ratio = srt[:, 0] / srt[:, 1]
    rho, p = stats.spearmanr(gap, ratio)
    win = np.abs(M_full[:, :, col]).argmax(1)
    near = ratio < 1.3
    return dict(rho=rho, p=p, gap_mean_near_tie=float(gap[near].mean()) if near.any() else np.nan,
                gap_mean_rest=float(gap[~near].mean()) if (~near).any() else np.nan,
                n_near=int(near.sum()), gap=gap, ratio=ratio, winner=win)


# ----------------------------------------------------------------------------------
# entry-level tests (secondary): per-entry mean |M| vs nulls, BH-FDR
# ----------------------------------------------------------------------------------
def fdr_bh(p):
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def crosssim_null_M(Us, Vhs, K, n_pairs=6000, seed=0, chunk=1000):
    """|Vh_a[:K] @ U_b[:, :K]| over random pairs a != b. Only meaningful if units are
    comparable across sims; empirically checked against Haar in the report."""
    n = len(Us)
    rng = np.random.default_rng(seed)
    a = rng.integers(0, n, n_pairs)
    b = (a + rng.integers(1, n, n_pairs)) % n
    out = np.empty((n_pairs, K, K))
    for s in range(0, n_pairs, chunk):
        sl = slice(s, s + chunk)
        out[sl] = np.abs(np.einsum('pkn,pnj->pkj', Vhs[a[sl], :K], Us[b[sl], :, :K]))
    return out


def test_overlap_entries(absM, N, crosssim=None, n_boot=2000, seed=0):
    """Per off-diagonal entry: mean |M|, ratio to Haar chance, analytic z-test p
    (Haar moments, independent sims), optional cross-sim empirical p, BH q-values."""
    import pandas as pd
    n_sim, K, _ = absM.shape
    mu = np.sqrt(2.0 / (np.pi * N))
    sd = np.sqrt(1.0 / N - 2.0 / (np.pi * N))
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            m = absM[:, i, j].mean()
            z = (m - mu) / (sd / np.sqrt(n_sim))
            row = dict(v=i + 1, u=j + 1, mean_abs=m, ratio=m / mu, p_haar=stats.norm.sf(z))
            if crosssim is not None:
                pool = crosssim[:, i, j]
                null_means = pool[rng.integers(0, len(pool), (n_boot, n_sim))].mean(1)
                row['p_cross'] = (1 + (null_means >= m).sum()) / (1 + n_boot)
                row['cross_mean'] = pool.mean()
            rows.append(row)
    df = pd.DataFrame(rows)
    df['q_haar'] = fdr_bh(df['p_haar'].values)
    if 'p_cross' in df:
        df['q_cross'] = fdr_bh(df['p_cross'].values)
    return df


def gauge_fix_by_column(M, col=0):
    """Gauge convention: choose d_i = sign(M[i,col]) (d_col=+1), return D M D.
    Makes column `col` non-negative; the remaining signed entries then carry
    genuinely gauge-fixed (convention-dependent but consistent) information."""
    d = np.sign(M[:, :, col])
    d[d == 0] = 1
    d[:, col] = 1
    return d[:, :, None] * M * d[:, None, :]


def surrogate_calibration(Ss, null, n=100, K=10, seed=0):
    """Spectrum-matched surrogates W' = U' S Vh' with independent Haar U', V'.
    Pushed through the full rank-profile pipeline; should not look localized."""
    rng = np.random.default_rng(seed)
    N = Ss.shape[1]
    Us, Vhs = [], []
    for _ in range(n):
        Us.append(stats.ortho_group.rvs(N, random_state=rng))
        Vhs.append(stats.ortho_group.rvs(N, random_state=rng))
    W = [U @ np.diag(S) @ Vh for U, S, Vh in zip(Us, Ss[rng.integers(0, len(Ss), n)], Vhs)]
    U2, S2, Vh2 = svd_all(np.stack(W))
    ov = compute_VU_overlap(U2, S2, Vh2, K)
    rp = rank_profile(ov['M_full'], 0, null)
    return dict(fp_rate_a1_p05=float((rp['p_a1'] < 0.05).mean()),
                a1_mean=float(rp['a1'].mean()), pr_mean=float(rp['pr'].mean()),
                n_sig_nonzero=float((rp['n_sig'] > 0).mean()))


# ----------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------
def _plt():
    import matplotlib
    if os.environ.get('DISPLAY') is None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_rank_profile(rps, null, out_dir, crosssim_rp=None, fname='recurrent_VU_rank_profile.svg'):
    """Headline figure. rps: dict col -> rank_profile (col 0 is the claim, others controls)."""
    plt = _plt()
    rp = rps[0]
    n, N = rp['srt'].shape
    kmax = 12
    ks = np.arange(1, kmax + 1)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4))

    ax = axs[0]
    ax.set_title(r'(a) sorted $|v_i\cdot u_1|$, every sim', fontsize=9)
    for s in range(n):
        ax.plot(ks, rp['srt'][s, :kmax], c=C_OBS, lw=.4, alpha=.25)
    ax.plot(ks, np.median(rp['srt'][:, :kmax], 0), c=C_OBS, lw=2, label='median over sims')
    ax.plot(ks, null['mean'][:kmax], c=C_NULL, lw=1.5, ls='--', label='Haar null (mean)')
    ax.axhline(null['a1_max'], c=C_NULL, lw=.8, ls=':', label='Haar null max of a(1)')
    ax.axhline(null['thr_bonf'], c=C_ACC, lw=.8, ls=':', label='Bonferroni threshold')
    ax.set_xlabel('rank k of partner within sim'); ax.set_ylabel(r'$a_{(k)}$')
    ax.legend(fontsize=6.5, frameon=False)

    ax = axs[1]
    ax.set_title('(b) sims sorted by a(1)/a(2); rows = rank', fontsize=9)
    o = np.argsort(-rp['ratio'])
    im = ax.imshow(rp['srt'][o, :10], aspect='auto', interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=max(rp['a1'].max(), .01))
    ax.set_xlabel('rank k'); ax.set_ylabel('simulation (sorted)')
    ax.set_xticks(range(10)); ax.set_xticklabels(range(1, 11))
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    ax = axs[2]
    ax.set_title('(c) how localized, per sim', fontsize=9)
    ax.hist(rp['ratio'], bins=np.linspace(1, max(4, np.percentile(rp['ratio'], 99)), 25),
            color=C_OBS, alpha=.85)
    ax.axvline(np.median(rp['ratio']), c=C_ACC, lw=1)
    ax.set_xlabel('a(1)/a(2)  (null median %.2f)' % np.median(null['a1'] / null['a2']))
    ax.set_ylabel('# sims')
    ax2 = ax.inset_axes([.55, .45, .4, .45])
    ax2.hist(rp['n_sig'], bins=np.arange(-.5, 8.5), color=C_OBS, alpha=.85)
    ax2.set_title('# partners above null', fontsize=6.5); ax2.tick_params(labelsize=6)

    ax = axs[3]
    ax.set_title(r'(d) control columns $u_j$: mean sorted profile', fontsize=9)
    cmap = plt.get_cmap('Blues')
    for c, r in sorted(rps.items()):
        ax.plot(ks, r['srt'][:, :kmax].mean(0), lw=2 if c == 0 else 1.2,
                c=C_ACC if c == 0 else cmap(.35 + .12 * (len(rps) - c)), label=f'$u_{{{c+1}}}$')
    if crosssim_rp is not None:
        ax.plot(ks, crosssim_rp['mean'][:kmax], c=C_NULL, ls='-.', lw=1, label='cross-sim null')
    ax.plot(ks, null['mean'][:kmax], c=C_NULL, ls='--', lw=1.2, label='Haar null')
    ax.set_xlabel('rank k'); ax.set_ylabel(r'$\langle a_{(k)}\rangle$')
    ax.legend(fontsize=6.5, frameon=False, ncol=2)

    from matplotlib.ticker import MaxNLocator
    for a in (axs[0], axs[3]):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    for a in axs:
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout(w_pad=3)
    fig.savefig(join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_VU_partner(rp, Ss, deg, out_dir, fname='recurrent_VU_partner.svg'):
    """Identity of the dominant partner and the (rejected) degeneracy explanation."""
    plt = _plt()
    n = rp['n_sim']
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))

    ax = axs[0]
    ax.set_title(r'(a) identity of dominant partner of $u_1$', fontsize=9)
    v, c = np.unique(rp['winner'], return_counts=True)
    ax.bar([f'v{int(i)+1}' for i in v], c, color=C_OBS)
    ax.set_ylabel('# sims'); ax.set_xlabel('index of argmax |v.u1|')

    ax = axs[1]
    ax.set_title('(b) index raster, sims grouped by winner', fontsize=9)
    o = np.lexsort((-rp['a1'], rp['winner']))
    im = ax.imshow(rp['a'][o, :10], aspect='auto', interpolation='nearest', cmap='Blues', vmin=0)
    ax.set_xlabel(r'right singular vector index $i$'); ax.set_ylabel('simulation (grouped)')
    ax.set_xticks(range(10)); ax.set_xticklabels(range(1, 11))
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    ax = axs[2]
    ax.set_title(r'(c) $a_{(1)}$ vs $a_{(2)}$, colour = $(s_2-s_3)/s_2$', fontsize=9)
    sc = ax.scatter(rp['a1'], rp['a2'], c=deg['gap'], cmap='viridis', s=14)
    lim = [0, max(rp['a1'].max(), .5) * 1.05]
    ax.plot(lim, lim, c=C_NULL, lw=.8, ls=':')
    ax.set_xlabel('a(1)'); ax.set_ylabel('a(2)')
    ax.text(.04, .95, f"Spearman(gap, a1/a2) = {deg['rho']:.2f}, p = {deg['p']:.2g}",
            transform=ax.transAxes, fontsize=7, va='top')
    fig.colorbar(sc, ax=ax, fraction=.046, pad=.03)
    for a in axs:
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_VU_overlap_stats(ov, tests, out_dir, fname='recurrent_VU_overlap_stats.svg'):
    plt = _plt()
    K, N = ov['K'], ov['N']
    mu = np.sqrt(2.0 / (np.pi * N))
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))

    ax = axs[0]
    ax.set_title(r'(a) mean $|v_i\cdot u_j|$ (chance %.3f)' % mu, fontsize=9)
    X = ov['absM'].mean(0)
    im = ax.imshow(X, cmap='Blues', vmin=0, interpolation='nearest')
    ax.set_xlabel('U id'); ax.set_ylabel('V id'); fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    ax = axs[1]
    ax.set_title(r'(b) $-\log_{10} q$ (Haar z-test, BH), masked at q>0.05', fontsize=9)
    Q = np.full((K, K), np.nan)
    for r in tests.itertuples():
        Q[r.v - 1, r.u - 1] = -np.log10(max(r.q_haar, 1e-300)) if r.q_haar < .05 else np.nan
    im = ax.imshow(Q, cmap='Blues', vmin=0, interpolation='nearest')
    ax.set_xlabel('U id'); ax.set_ylabel('V id'); fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    ax = axs[2]
    ax.set_title(r'(c) mean $s_i|v_i\cdot u_j|$: copying gain $|T|$', fontsize=9)
    im = ax.imshow(np.abs(ov['T']).mean(0), cmap='Blues', vmin=0, interpolation='nearest')
    ax.set_xlabel('U id (from)'); ax.set_ylabel('V id (to)'); fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.tight_layout()
    fig.savefig(join(out_dir, fname), dpi=200)
    plt.close(fig)


def plot_gauge_fixed_mean(ov, out_dir, col=0, fname='recurrent_VU_gaugefixed.svg'):
    """Naive (cancelling) mean M next to the gauge-convention mean D M D."""
    plt = _plt()
    M = ov['M']
    Mg = gauge_fix_by_column(M, col)
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, X, t in ((axs[0], M.mean(0), 'naive mean of M (gauge-dependent)'),
                     (axs[1], Mg.mean(0), f'mean of D M D, gauge: col u_{col+1} >= 0')):
        v = np.abs(X).max()
        im = ax.imshow(X, cmap='PiYG', vmin=-v, vmax=v, interpolation='nearest')
        ax.set_title(t, fontsize=9); ax.set_xlabel('U id'); ax.set_ylabel('V id')
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    fig.tight_layout()
    fig.savefig(join(out_dir, fname), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------------
# self test: every reported statistic must be invariant under the sign gauge
# ----------------------------------------------------------------------------------
def selftest(seed=0):
    rng = np.random.default_rng(seed)
    N, K, n = 40, 6, 5
    Ws = rng.standard_normal((n, N, N)) / np.sqrt(N)
    U, S, Vh = svd_all(Ws)
    d = rng.choice([-1.0, 1.0], size=(n, N))
    U2, Vh2 = U * d[:, None, :], Vh * d[:, :, None]
    assert np.allclose(np.einsum('sij,sj,sjk->sik', U2, S, Vh2), Ws), 'gauge flip changed W'
    a, b = compute_VU_overlap(U, S, Vh, K), compute_VU_overlap(U2, S, Vh2, K)
    nul = rank_profile_null(N, 2000, 0)
    ra, rb = rank_profile(a['M_full'], 0, nul), rank_profile(b['M_full'], 0, nul)
    for key in ('absM', 'cycle2'):
        assert np.allclose(a[key], b[key]), key
    assert np.allclose(np.abs(a['T']), np.abs(b['T'])), 'T'
    assert np.allclose(np.diagonal(a['M'], axis1=1, axis2=2), np.diagonal(b['M'], axis1=1, axis2=2)), 'diag'
    for key in ('srt', 'pr', 'ratio', 'n_sig', 'p_a1'):
        assert np.allclose(ra[key], rb[key]), key
    assert np.allclose(subspace_overlap(a['M_full'], 0, (1, 2, 3)), subspace_overlap(b['M_full'], 0, (1, 2, 3)))
    ta = test_overlap_entries(a['absM'], N)
    tb = test_overlap_entries(b['absM'], N)
    assert np.allclose(ta['p_haar'], tb['p_haar']) and np.allclose(ta['q_haar'], tb['q_haar'])
    # the naive signed mean is NOT invariant -- that is the whole point
    assert not np.allclose(a['M'].mean(0), b['M'].mean(0)), 'signed mean should depend on gauge'
    # a column sums to one in squares (M is orthogonal over all N indices)
    assert np.allclose((a['M_full'][:, :, 0] ** 2).sum(1), 1)
    # null sanity: flat profile, PR ~ N/3
    big = rank_profile_null(160, 20000, 0)
    assert abs(big['pr'].mean() - 160 / 3) < 5, big['pr'].mean()
    assert big['mean'][0] - big['mean'][4] < .08, 'null profile should be flat'
    # BH sanity
    assert np.allclose(fdr_bh([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    print('selftest OK: all statistics are gauge-invariant; null profile flat; PR_null=%.1f (N/3=%.1f)'
          % (big['pr'].mean(), 160 / 3))


# ----------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------
def run(data_dir, out_dir, K=10, col=0, n_controls=5, calibrate=True):
    os.makedirs(out_dir, exist_ok=True)
    Ws = load_rec_weights(data_dir)
    n_sim, N, _ = Ws.shape
    print(f'{n_sim} sims, N={N}, data: {data_dir}')
    Us, Ss, Vhs = svd_all(Ws)
    ov = compute_VU_overlap(Us, Ss, Vhs, K)
    null = rank_profile_null(N, seed=0)
    rps = {j: rank_profile(ov['M_full'], j, null) for j in range(n_controls)}
    for j in range(n_controls):
        print(summarize_rank_profile(rps[j], null))
        print()

    # cross-sim null: is the effect pairing within a net, or a common-subspace artefact?
    cs = crosssim_null_M(Us, Vhs, K)
    a_cross = np.empty((2000, N))
    rng = np.random.default_rng(1)
    for p in range(2000):
        a_, b_ = rng.choice(n_sim, 2, replace=False)
        a_cross[p] = np.abs(Vhs[a_] @ Us[b_][:, col])
    cross_rp = dict(mean=(-np.sort(-a_cross, axis=1))[:, :20].mean(0))
    print('cross-sim null a(1..5):', np.round(cross_rp['mean'][:5], 3),
          ' Haar null:', np.round(null['mean'][:5], 3),
          ' (close => sims share no unit coordinates, as expected)')

    tests = test_overlap_entries(ov['absM'], N, crosssim=cs)
    top = tests.sort_values('mean_abs', ascending=False).head(12)
    print('\ntop off-diagonal entries (v=row, u=col):')
    print(top.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
    print('\nsubspace overlap O[u1, {v2,v3,v4}] = %.3f (near-saturated for low-rank W; not evidence)'
          % subspace_overlap(ov['M_full'], col, (1, 2, 3)).mean())
    diag = np.diagonal(ov['M'], axis1=1, axis2=2)
    print('signed diagonal M[i,i] mean (i=1..K):', np.round(diag.mean(0), 3))

    deg = degeneracy_check(Ss, ov['M_full'], col)
    print(f"\ndegeneracy: Spearman(gap, a1/a2) rho={deg['rho']:.3f} p={deg['p']:.2g}; "
          f"near-ties {deg['n_near']}/{n_sim}, mean gap near-tie {deg['gap_mean_near_tie']:.3f} vs rest {deg['gap_mean_rest']:.3f}")
    if calibrate:
        print('surrogate calibration (Haar U,V, observed S):',
              surrogate_calibration(Ss, null, n=100, K=K))

    plot_rank_profile(rps, null, out_dir, crosssim_rp=cross_rp)
    plot_VU_partner(rps[col], Ss, deg, out_dir)
    plot_VU_overlap_stats(ov, tests, out_dir)
    plot_gauge_fixed_mean(ov, out_dir, col)
    tests.to_csv(join(out_dir, 'recurrent_VU_entry_tests.csv'), index=False)
    print('\nfigures + csv written to', out_dir)
    return dict(ov=ov, null=null, rps=rps, tests=tests, deg=deg)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--data-dir')
    ap.add_argument('--out')
    ap.add_argument('--col', type=int, default=0)
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--no-calibrate', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        selftest()
    if args.data_dir:
        out = args.out or join(os.path.dirname(os.path.abspath(args.data_dir)), 'figs',
                               os.path.basename(os.path.abspath(args.data_dir)))
        run(args.data_dir, out, K=args.K, col=args.col, calibrate=not args.no_calibrate)
