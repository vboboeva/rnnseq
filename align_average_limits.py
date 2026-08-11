"""
align_average_limits.py
=======================

Synthetic stress-test for the align-then-average rank-estimation method used in
Boboeva et al. (2026), "From sequences to schemas".

Purpose
-------
The method aligns an ensemble of independently trained networks into a common
frame using the right singular vectors V of the end-of-sequence activity matrix
H, conjugates the recurrent weights into that frame, and averages across simulations.
The simulation-consistent structured low-rank component survives; the simulation-random
residual cancels. The rank is read off the "elbow" where the aligned-average
spectrum separates from the unaligned-average spectrum.

This script isolates and quantifies one failure mode: when the singular values
of H are (near-)degenerate, the right singular vectors spanning that block are
defined only up to an arbitrary orthogonal rotation Q_k in O(d), effectively
independent across simulations. Averaging the conjugated weights over such rotations
ISOTROPISES the block:

        E_{Q ~ Haar O(d)} [ Q A Q^T ] = (tr A / d) I_d ,

so any anisotropic structure that lives inside a degenerate activity subspace is
diluted across d directions and, unless it carries most of the block's trace, is
erased. Near-degeneracy gives a smooth crossover rather than a cliff, controlled
by the Davis-Kahan ratio

        rho_r = sigma_r / |s'_r|

where sigma_r is the simulation-to-simulation std of the r-th activity singular value and
s'_r the local slope of the mean activity spectrum. Components with rho_r << 1
are protected by a resolved eigengap and recovered cleanly; components with
rho_r >~ 1 fall in the isotropisation regime and are lost.

Design notes
------------
* Everything is built in a fixed canonical basis (the "aligned frame"). Each
  fake simulation applies a global Haar rotation R_k in O(N) to BOTH the weights
  and the activity. R_k cancels exactly in the aligned pathway (that is the whole
  point of gauge-fixing to the activity frame), so it is not the source of the
  failure; it is retained only so that the UNaligned average <W_k> correctly
  cancels the structure, reproducing the paper's before/after-alignment baseline.
* The activity spectrum (Sigma0) and the planted weight structure (W0) are
  controlled SEPARATELY. Their coupling is a single knob: how much the structured
  singular vectors project onto the degenerate plateau of Sigma0. This decoupling
  is what lets the confound appear; if structure and activity spectrum were the
  same object (e.g. y = M x with Sigma = M M^T), the structure would sit in the
  best-separated directions by construction and the failure could never show.

Outputs
-------
Three figures saved to the working directory:
  fig_spectrum_baseline.png   sanity check: aligned vs plain average spectra
  fig_overlap_vs_position.png recovery as structure slides top -> plateau
  fig_crossover.png           recovery vs rho = sigma_r / |s'_r| (the money plot)

Author: prepared for V. Boboeva
"""

import numpy as np
from numpy.linalg import svd, qr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Building blocks
# --------------------------------------------------------------------------- #
def haar_orthogonal(n, rng):
    """A Haar-distributed orthogonal matrix in O(n)."""
    A = rng.standard_normal((n, n))
    Q, Rr = qr(A)
    # fix signs so Q is Haar (Mezzadri 2007)
    Q = Q * np.sign(np.diag(Rr))
    return Q


def build_spectrum(N, top_vals, plateau_val, plateau_width, tail_decay, rng=None):
    """
    Activity eigen-spectrum (variances) with three regions:
      - a few well-separated 'top' eigenvalues  (top_vals)
      - a near-degenerate plateau of given width (plateau_val, plateau_width)
      - a smoothly decaying tail
    Returns lam (length N), sorted descending.
    """
    top = np.asarray(top_vals, dtype=float)
    n_top = len(top)
    plateau = plateau_val * np.ones(plateau_width)
    n_tail = N - n_top - plateau_width
    if n_tail < 0:
        raise ValueError("N too small for requested top + plateau widths.")
    tail = plateau_val * np.exp(-tail_decay * (1 + np.arange(n_tail)))
    lam = np.concatenate([top, plateau, tail])
    return lam


def planted_structure(N, R, S, positions_top, positions_plateau, mix, rng):
    """
    Rank-R structured weight  W0 = sum_r S[r] l_r r_r^T.

    Each structured direction is a unit vector interpolating between a 'top'
    canonical axis (well separated in the activity spectrum) and a 'plateau'
    canonical axis (degenerate), controlled by mix in [0, 1]:
        mix = 0 -> structure sits entirely in the separated top block
        mix = 1 -> structure sits entirely in the degenerate plateau block
    Left and right vectors use the same positions here (a symmetric planting);
    this is not essential but keeps the geometry transparent.
    """
    L = np.zeros((N, R))
    Rr = np.zeros((N, R))
    for r in range(R):
        v = np.zeros(N)
        v[positions_top[r]] = np.cos(0.5 * np.pi * mix)
        v[positions_plateau[r]] = np.sin(0.5 * np.pi * mix)
        v = v / np.linalg.norm(v)
        L[:, r] = v
        Rr[:, r] = v
    W0 = (L * S) @ Rr.T
    return W0, L, Rr


def subspace_overlap(U, Vtrue):
    """
    Mean captured energy of the true structured subspace by the estimated one:
      overlap = (1/R) * || P_U Vtrue ||_F^2  in [0, 1], 1 = perfect recovery.
    U:     N x R  estimated basis (need not be orthonormal columns of interest)
    Vtrue: N x R  true orthonormal-ish basis
    """
    # orthonormalise both
    Uo, _ = qr(U)
    Uo = Uo[:, : U.shape[1]]
    Vo, _ = qr(Vtrue)
    Vo = Vo[:, : Vtrue.shape[1]]
    P = Uo @ Uo.T
    return np.sum((P @ Vo) ** 2) / Vtrue.shape[1]


# --------------------------------------------------------------------------- #
#  One ensemble experiment
# --------------------------------------------------------------------------- #
def run_ensemble(N, p, K, lam, W0, Ltrue, eta, rng_seed=0):
    """
    Generate K fake 'simulations', align each to its activity frame, and average.

    Returns
    -------
    s_aligned : singular values of the aligned average  <V_k^T W_k V_k>
    s_plain   : singular values of the unaligned average <W_k>
    overlapR  : subspace overlap of top-R aligned-average left singular vectors
                with the true planted span Ltrue
    act_sv    : (K, N) activity singular values per simulation (for measuring sigma_r)
    """
    rng = np.random.default_rng(rng_seed)
    sqrt_lam = np.sqrt(lam)

    Wbar_aligned = np.zeros((N, N))
    Wbar_plain = np.zeros((N, N))
    act_sv = np.zeros((K, N))

    for k in range(K):
        Rk = haar_orthogonal(N, rng)  # fake independent training frame

        # weights in simulation frame: structure + residual, then rotate
        Xi = rng.standard_normal((N, N)) / np.sqrt(N)
        Wk_canon = W0 + eta * Xi
        Wk = Rk @ Wk_canon @ Rk.T

        # activity in simulation frame: H_canon has covariance diag(lam)
        G = rng.standard_normal((p, N))
        H_canon = G * sqrt_lam[None, :]  # p x N, Cov ~ diag(lam)
        Hk = H_canon @ Rk.T

        # alignment frame = right singular vectors of Hk
        _, s_k, Vt_k = svd(Hk, full_matrices=False)
        Vk = Vt_k.T  # N x min(p, N)
        act_sv[k, : len(s_k)] = s_k
        # if p < N the activity spans only a subspace; complete the frame with a
        # random orthonormal complement (unobserved directions are maximally
        # unconstrained, which is itself part of the degeneracy story)
        if Vk.shape[1] < N:
            filler = rng.standard_normal((N, N - Vk.shape[1]))
            filler = filler - Vk @ (Vk.T @ filler)
            Qc, _ = qr(filler)
            Vk = np.concatenate([Vk, Qc[:, : N - Vk.shape[1]]], axis=1)

        # conjugate weights into the aligned frame; Rk cancels here by construction
        Wbar_aligned += Vk.T @ Wk @ Vk
        Wbar_plain += Wk

    Wbar_aligned /= K
    Wbar_plain /= K

    s_aligned = svd(Wbar_aligned, compute_uv=False)
    s_plain = svd(Wbar_plain, compute_uv=False)

    # recovery of the structured subspace
    R = Rrank(W0)
    U, _, _ = svd(Wbar_aligned)
    overlapR = subspace_overlap(U[:, :R], Ltrue)

    return s_aligned, s_plain, overlapR, act_sv


def Rrank(W0, tol=1e-9):
    s = svd(W0, compute_uv=False)
    return int(np.sum(s > tol * s[0]))


# --------------------------------------------------------------------------- #
#  rho = sigma_r / |s'_r| measured from the activity spectra
# --------------------------------------------------------------------------- #
def degeneracy_ratio(act_sv, index):
    """
    Empirical Davis-Kahan ratio at a given spectral index:
      sigma  = std across simulations of the index-th activity singular value
      slope  = local slope |d(mean singular value)/d index| (central difference)
      rho    = sigma / |slope|   (in units of 'indices of spread')
    """
    mean_sv = act_sv.mean(0)
    sigma = act_sv[:, index].std()
    i0 = max(index - 1, 0)
    i1 = min(index + 1, len(mean_sv) - 1)
    slope = (mean_sv[i1] - mean_sv[i0]) / (i1 - i0)
    rho = sigma / (abs(slope) + 1e-12)
    return rho, sigma, abs(slope)


# --------------------------------------------------------------------------- #
#  Experiments / figures
# --------------------------------------------------------------------------- #
def fig_baseline():
    """Sanity check: with structure in the separated top block, aligned average
    recovers R structured singular values while the plain average cancels them."""
    N, p, K, R = 60, 400, 60, 3
    lam = build_spectrum(
        N, top_vals=[40, 30, 22], plateau_val=6.0, plateau_width=20, tail_decay=0.15
    )
    pos_top = [0, 1, 2]
    pos_plat = [3, 4, 5]  # unused at mix=0
    S = np.array([6.0, 5.0, 4.0])
    W0, L, _ = planted_structure(N, R, S, pos_top, pos_plat, mix=0.0, rng=np.random.default_rng(0))
    s_al, s_pl, ov, act = run_ensemble(N, p, K, lam, W0, L, eta=1.0, rng_seed=1)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    idx = np.arange(1, min(15, N) + 1)
    ax[0].plot(idx, s_al[:15], "o-", color="crimson", label=r"aligned $\langle \bar W\rangle$")
    ax[0].plot(idx, s_pl[:15], "s--", color="steelblue", label=r"plain $\langle W\rangle$")
    ax[0].axvspan(0.5, R + 0.5, color="orange", alpha=0.15)
    ax[0].set_xlabel("singular value index")
    ax[0].set_ylabel("singular value")
    ax[0].set_title(f"Structure in separated top block\n(recovered overlap = {ov:.2f})")
    ax[0].legend(frameon=False, fontsize=9)

    lam_mean = act.mean(0)
    ax[1].plot(np.arange(1, N + 1), lam_mean, "k.-", ms=4)
    ax[1].set_xlabel("activity component index")
    ax[1].set_ylabel("activity singular value")
    ax[1].set_title("Activity spectrum (top / plateau / tail)")
    ax[1].set_xlim(0, 40)
    fig.tight_layout()
    fig.savefig("fig_spectrum_baseline.png", dpi=140)
    plt.close(fig)
    print(f"[baseline] recovered overlap with structure in top block = {ov:.3f}")


def fig_overlap_vs_position():
    """Slide structure from separated top into degenerate plateau; recovery falls."""
    N, p, K, R = 60, 400, 60, 3
    lam = build_spectrum(
        N, top_vals=[40, 30, 22], plateau_val=6.0, plateau_width=20, tail_decay=0.15
    )
    pos_top = [0, 1, 2]
    pos_plat = [3, 4, 5]
    S = np.array([6.0, 5.0, 4.0])
    mixes = np.linspace(0, 1, 11)
    overlaps = []
    for m in mixes:
        W0, L, _ = planted_structure(N, R, S, pos_top, pos_plat, mix=m, rng=np.random.default_rng(0))
        _, _, ov, _ = run_ensemble(N, p, K, lam, W0, L, eta=1.0, rng_seed=2)
        overlaps.append(ov)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.plot(mixes, overlaps, "o-", color="crimson")
    ax.set_xlabel(r"structure position:  top $\rightarrow$ plateau (mix)")
    ax.set_ylabel("recovered subspace overlap")
    ax.set_ylim(0, 1.02)
    ax.set_title("Recovery collapses as structure\nenters the degenerate subspace")
    fig.tight_layout()
    fig.savefig("fig_overlap_vs_position.png", dpi=140)
    plt.close(fig)
    print("[position sweep] overlaps:", np.round(overlaps, 3))


def build_ramp_spectrum(N, top_vals, ramp_center, ramp_half_gap, ramp_width, tail_decay):
    """
    Spectrum with a separated top block, then a linear RAMP of controllable slope
    (this is the near-degenerate region: slope -> 0 is exactly degenerate), then a
    decaying tail. The ramp descends from center+half_gap to center-half_gap over
    ramp_width indices, so the per-index eigengap is 2*half_gap/ramp_width. Varying
    half_gap sweeps the region from flat (degenerate) to well separated.
    """
    top = np.asarray(top_vals, dtype=float)
    n_top = len(top)
    ramp = np.linspace(ramp_center + ramp_half_gap, ramp_center - ramp_half_gap, ramp_width)
    n_tail = N - n_top - ramp_width
    tail = (ramp_center - ramp_half_gap) * np.exp(-tail_decay * (1 + np.arange(n_tail)))
    return np.concatenate([top, ramp, tail])


def fig_crossover():
    """
    The money plot: place structure inside a near-degenerate ramp and sweep the
    ramp's local eigengap (its flatness). Flat ramp -> ill-defined eigenbasis ->
    rho large -> recovery lost. Steeper ramp -> resolved eigengap -> rho small ->
    recovery clean. Plot recovery vs rho = sigma_r / |s'_r| and mark rho ~ 1.
    """
    N, K, R = 60, 80, 3
    ramp_center, ramp_width, half_gap = 20.0, 24, 5.0  # per-index eigengap ~ 0.42 (nonzero)
    pos_top = [0, 1, 2]          # unused at mix=1
    # structure sits at three representative ramp indices (offset by top block = 3)
    ramp_idx = [3 + 6, 3 + 11, 3 + 16]
    S = np.array([6.0, 6.0, 6.0])

    lam = build_ramp_spectrum(
        N, top_vals=[70, 52, 40], ramp_center=ramp_center,
        ramp_half_gap=half_gap, ramp_width=ramp_width, tail_decay=0.15
    )
    W0, L, _ = planted_structure(
        N, R, S, pos_top, ramp_idx, mix=1.0, rng=np.random.default_rng(0)
    )

    # sweep sample size: small p -> gap unresolved (large rho), large p -> resolved.
    # average each point over several ensemble repeats to smooth the boundary.
    p_values = np.array([65, 90, 130, 200, 320, 550, 950, 1700, 3200, 6500, 14000])
    n_rep = 6
    rhos, overlaps, ov_err = [], [], []
    probe_index = ramp_idx[1]
    for p in p_values:
        ov_reps, rho_reps = [], []
        for rep in range(n_rep):
            _, _, ov, act = run_ensemble(N, int(p), K, lam, W0, L, eta=1.0, rng_seed=100 + rep)
            rho, sig, slope = degeneracy_ratio(act, probe_index)
            ov_reps.append(ov)
            rho_reps.append(rho)
        rhos.append(np.mean(rho_reps))
        overlaps.append(np.mean(ov_reps))
        ov_err.append(np.std(ov_reps))
    rhos = np.array(rhos)
    overlaps = np.array(overlaps)
    ov_err = np.array(ov_err)

    order = np.argsort(rhos)
    floor = R / ramp_width  # chance-level overlap when the block is isotropised
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    ax.errorbar(
        rhos[order], overlaps[order], yerr=ov_err[order],
        fmt="o-", color="crimson", capsize=2, lw=1.5, ms=5
    )
    ax.axhline(floor, color="grey", ls="--", lw=1)
    ax.text(rhos.min(), floor + 0.02, r"isotropised floor $\approx R/d$", fontsize=8, color="grey")
    ax.set_xlabel(r"degeneracy ratio  $\rho_r = \sigma_r / |s'_r|$")
    ax.set_ylabel("recovered subspace overlap")
    ax.set_ylim(0, 1.05)
    ax.set_title("Estimator validity boundary\n(structure inside a near-degenerate region)")
    ax.annotate(
        "resolved\n(recovered)", xy=(rhos[order][0], overlaps[order][0]),
        xytext=(rhos[order][0], 0.5), fontsize=8, ha="center", color="k"
    )
    ax.annotate(
        "degenerate\n(erased)", xy=(rhos[order][-1], overlaps[order][-1]),
        xytext=(rhos[order][-1], 0.45), fontsize=8, ha="center", color="k"
    )
    fig.tight_layout()
    fig.savefig("fig_crossover.png", dpi=140)
    plt.close(fig)
    print("[crossover] rho :", np.round(rhos[order], 3))
    print("[crossover] ovl :", np.round(overlaps[order], 3))


if __name__ == "__main__":
    print("Running synthetic stress-test of align-then-average rank estimation...\n")
    fig_baseline()
    fig_overlap_vs_position()
    fig_crossover()
    print("\nDone. Figures written to working directory.")
