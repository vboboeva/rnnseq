"""
tier_diagnostics.py
===================

Companion to align_average_limits.py. Addresses the question: the synthetic
validation established WHEN the align-then-average estimator works, but what can
we say about the REAL activity spectra, which show vertical spread across
simulations?

The spread sigma seen in the per-simulation activity spectra is NOT, by itself,
a measure of the danger to the estimator. It mixes at least three mechanisms
that affect the averaging completely differently:

  Tier 1  scale/gain spread : simulations differ by an overall multiplicative
          factor on the spectrum. Singular VECTORS and their ordering are
          identical -> the alignment frame is identical -> averaging is EXACT.
          sigma is large but harmless. Defused by per-simulation normalisation.

  Tier 2  permutation within a quasi-degenerate band : the shared set of singular
          vectors is the same across simulations, but nearby singular values swap
          order, so the SV-sorted ordering relabels them. Averaging in sorted
          order is degraded, but the damage is RECOVERABLE: detect the relabelling
          by matching vectors across simulations and undo it before averaging.

  Tier 3  genuine rotation within an essentially-degenerate subspace : the vectors
          are truly arbitrary linear combinations across simulations. This is the
          only FATAL case (Haar isotropisation E[Q A Q^T] = (tr A / d) I). No
          relabelling helps.

Crucially, the naive ratio rho_r = sigma_r / |s'_r| fails as a proxy in BOTH
directions: it over-states danger for tier 1 (fires when nothing is wrong) and
under-states it for tier 2 (the sorted spectrum barely moves while the vectors
swap). The reliable diagnostics operate on the VECTORS, not the spectrum:

  (i)  gain-normalisation test  -> how much of sigma_r is benign scale (tier 1)
  (ii) matched singular-vector overlap after a global Procrustes alignment
       -> distinguishes tier 2 (overlap ~ 1, recoverable) from tier 3 (overlap
          << 1, fatal), and its quality is the number to report.

This script (a) constructs the three tiers explicitly, (b) shows that rho tracks
none of them while matched overlap tracks recovery, (c) shows that a
vector-matched average repairs tier 2, and (d) provides a function to run the
same diagnostics on the real per-simulation activity matrices.

Nomenclature: "simulation" throughout (matching the manuscript), never "seed";
'rng_seed' denotes only the seed of the random number generator.

Author: prepared for V. Boboeva
"""

import numpy as np
from numpy.linalg import svd, qr
from scipy.optimize import linear_sum_assignment
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Basic building blocks
# --------------------------------------------------------------------------- #
def haar_orthogonal(n, rng):
    """A Haar-distributed orthogonal matrix in O(n)."""
    A = rng.standard_normal((n, n))
    Q, R = qr(A)
    return Q * np.sign(np.diag(R))


def subspace_overlap(U, Vtrue):
    """Mean captured energy of span(Vtrue) by span(U), in [0, 1]."""
    Uo, _ = qr(U)
    Uo = Uo[:, : U.shape[1]]
    Vo, _ = qr(Vtrue)
    Vo = Vo[:, : Vtrue.shape[1]]
    P = Uo @ Uo.T
    return np.sum((P @ Vo) ** 2) / Vtrue.shape[1]


# --------------------------------------------------------------------------- #
#  Explicit construction of the three tiers
# --------------------------------------------------------------------------- #
def make_tier_ensemble(
    N=60, K=80, R=3, band=(24, 36), tier="rotation",
    gain_std=0.35, value_jitter=0.04, rng_seed=0, global_frame=False,
):
    """
    Build K surrogate simulations that share a common canonical structure but
    differ WITHIN a small band of the activity spectrum according to `tier`.

    The spectrum is fully resolvable everywhere except the small band, so the
    band residual is what carries the tier information -- a representative case,
    not the adversarial all-degenerate one.

    global_frame:
      False (default) -- work directly in the common frame that the method's
        conjugation establishes (the R_k that distinguishes independently trained
        networks has already cancelled after V_k^T W_k V_k, as shown for the main
        result). This isolates the within-frame tier phenomenon and lets the
        permutation be read out and undone exactly, which is the cleanest
        demonstration of the taxonomy.
      True -- additionally place each simulation in an independent global frame
        R_k applied to both activity and weights, reproducing the raw
        before-alignment situation. Recovering the tier then also requires the
        confidence-weighted Procrustes step (see analyse_real_simulations), which
        is reliable only when the resolvable bulk pins the frame.

    Returns per-simulation lists of (Vk, Sk, Wk), the true structured basis Ltrue
    (canonical), the canonical weight, and the structure indices.
    """
    rng = np.random.default_rng(rng_seed)
    b0, b1 = band
    band_idx = np.arange(b0, b1)
    d = len(band_idx)

    # fully resolvable, monotone spectrum; the band is a small window within it
    base = np.linspace(60.0, 5.0, N)

    V0 = np.eye(N)  # canonical activity right-singular vectors

    # planted rank-R structure INSIDE the band (the case that stresses the method)
    struct_idx = list(band_idx[np.linspace(0, d - 1, R).round().astype(int)])
    Ltrue = np.zeros((N, R))
    for j, s in enumerate(struct_idx):
        Ltrue[s, j] = 1.0
    S_struct = np.full(R, 6.0)
    W0canon = (Ltrue * S_struct) @ Ltrue.T

    Vk_list, Sk_list, Wk_list = [], [], []
    for k in range(K):
        Rk = haar_orthogonal(N, rng) if global_frame else np.eye(N)
        S = base.copy()
        Pi = np.eye(N)

        if tier == "scaling":
            # tier 1: overall gain; vectors and ordering untouched
            S = np.exp(gain_std * rng.standard_normal()) * base

        elif tier == "permutation":
            # tier 2: vectors pinned by nonzero gaps, but small value jitter makes
            # the nearby band values swap order -> SV-sorting relabels the axes
            S[band_idx] = base[band_idx] + value_jitter * base[band_idx] * rng.standard_normal(d)

        elif tier == "rotation":
            # tier 3: make the band exactly degenerate, then rotate it genuinely
            S[band_idx] = base[band_idx].mean()
            Q = haar_orthogonal(d, rng)
            Pi[np.ix_(band_idx, band_idx)] = Q

        else:
            raise ValueError(tier)

        Vk_canon = V0 @ Pi
        order = np.argsort(-S)                 # a real SVD returns SV-sorted columns
        Vk_canon = Vk_canon[:, order]
        S_sorted = S[order]

        Vk_list.append(Rk @ Vk_canon)
        Sk_list.append(S_sorted)
        Wk_list.append(Rk @ W0canon @ Rk.T)

    return Vk_list, Sk_list, Wk_list, Ltrue, W0canon, struct_idx


# --------------------------------------------------------------------------- #
#  Diagnostics
# --------------------------------------------------------------------------- #
def naive_rho(Sk_list, index):
    """The (unreliable) spectral proxy rho_r = sigma_r / |s'_r|, for reference."""
    Smat = np.vstack(Sk_list)
    mean_s = Smat.mean(0)
    sigma = Smat[:, index].std()
    i0, i1 = max(index - 1, 0), min(index + 1, Smat.shape[1] - 1)
    slope = (mean_s[i1] - mean_s[i0]) / (i1 - i0)
    return sigma / (abs(slope) + 1e-12)


def gain_normalised_spread(Sk_list):
    """
    Tier-1 test: seed-to-seed std of each singular value BEFORE and AFTER
    removing a per-simulation scalar gain (here the spectrum's Frobenius norm).
    A large drop means most of the spread is benign scale.
    """
    Smat = np.vstack(Sk_list).astype(float)
    sigma_raw = Smat.std(0)
    gains = np.linalg.norm(Smat, axis=1, keepdims=True)
    Snorm = Smat / gains * gains.mean()
    sigma_norm = Snorm.std(0)
    return sigma_raw, sigma_norm


def confidence_weights(mean_s):
    """
    Per-component weight for the Procrustes alignment: large singular value AND
    large local eigengap. Flat (degenerate) regions get ~zero weight, so they
    cannot pin the global frame -- which is what lets the within-band residual
    reveal genuine rotation instead of having it absorbed by the alignment.
    """
    N = len(mean_s)
    gap = np.zeros(N)
    gap[1:-1] = np.abs(mean_s[:-2] - mean_s[2:]) / 2
    gap[0] = abs(mean_s[0] - mean_s[1])
    gap[-1] = abs(mean_s[-2] - mean_s[-1])
    w = mean_s * gap
    return w / (w.max() + 1e-12)


def procrustes_align(Vk, Vref, w, reg=0.0):
    """
    WARNING -- DO NOT USE TO COMPARE FULL ORTHONORMAL BASES ACROSS INDEPENDENT
    NETWORKS. For two complete orthonormal bases the weighted orthogonal
    Procrustes solution is O = Vref @ Vk^T *independently of the weights*
    (polar(Vref W Vk^T) = Vref Vk^T for any positive W), so O maps any basis
    exactly onto any other and the subsequent matched overlap is identically 1 --
    a tautology that measures nothing. This was a real bug in an earlier version.
    The activity singular vectors of independently trained networks live in
    unrelated global frames and cannot be compared by alignment; use the
    frame-free split_half_reproducibility on the conjugated weights instead.

    This routine is retained only for the common-frame tier demonstration
    (use_procrustes=False there) and for cases where the alignment is genuinely
    restricted to a low-dimensional confident subspace by hard masking (not the
    case here). Returns O @ Vk.
    """
    A = Vref * w[None, :]
    B = Vk * w[None, :]
    M = A @ B.T + reg * np.eye(Vk.shape[0])
    U, _, Wt = svd(M)
    O = U @ Wt
    return O @ Vk


def matched_overlap_and_perm(Vk_aligned, Vref):
    """
    After global alignment, match each aligned vector to a reference vector by
    the optimal assignment on |<vk_i, vref_j>|. Returns, indexed by REFERENCE
    component, the matched overlap (in [0,1]; ~1 = same vector up to relabelling,
    <<1 = genuine rotation) and the permutation mapping reference -> simulation.
    """
    Ov = np.abs(Vk_aligned.T @ Vref)          # [i_sim, j_ref]
    row, col = linear_sum_assignment(-Ov)      # maximise total overlap
    ov_by_ref = np.zeros(Vref.shape[1])
    perm_ref_to_sim = np.zeros(Vref.shape[1], dtype=int)
    ov_by_ref[col] = Ov[row, col]
    perm_ref_to_sim[col] = row
    return ov_by_ref, perm_ref_to_sim


def aligned_average(Vk_list, Sk_list, Wk_list, match=False, use_procrustes=False, ref=0):
    """
    Conjugate each simulation's weights into its activity frame and average.

    match=False        : the paper's estimator -- average the conjugated weights
                         in SV-sorted order.
    match=True         : bring every simulation's activity vectors into
                         correspondence with a reference (optimal assignment) and
                         reorder the conjugated weights before averaging -- the
                         vector-matched estimator that repairs tier-2 permutations.
    use_procrustes     : if True, first remove a global frame by confidence-
                         weighted Procrustes (needed when simulations carry
                         independent global rotations R_k); if False, match the raw
                         vectors directly (correct once conjugation has already put
                         everything in the common frame).

    Returns the averaged weight matrix and the per-reference-component matched
    overlap averaged over simulations (meaningful only when match=True).
    """
    N = Vk_list[0].shape[0]
    Vref = Vk_list[ref]
    w = confidence_weights(np.vstack(Sk_list).mean(0))

    Wbar = np.zeros((N, N))
    ov_accum = np.zeros(N)
    for Vk, Wk in zip(Vk_list, Wk_list):
        Wc = Vk.T @ Wk @ Vk                    # common-frame block; R_k cancels
        if match:
            Vk_al = procrustes_align(Vk, Vref, w) if use_procrustes else Vk
            ov, perm = matched_overlap_and_perm(Vk_al, Vref)
            ov_accum += ov
            P = np.eye(N)[perm]                 # reference-ordered permutation
            Wc = P @ Wc @ P.T
        Wbar += Wc
    Wbar /= len(Vk_list)
    ov_accum /= len(Vk_list)
    return Wbar, ov_accum


def split_half_reproducibility(Vk_list, Wk_list, R, n_splits=80, rng_seed=0):
    """
    Frame-free reliability test. Randomly split the simulations into two halves,
    average the conjugated weights within each half, and measure the subspace
    overlap of the top-R left singular vectors between halves. Repeat.

    This operates entirely on the conjugated weights Wc_k = V_k^T W_k V_k, which
    are frame-free (the global R_k has cancelled), so NO cross-network vector
    alignment is needed -- avoiding the full-basis alignment tautology (see the
    warning in procrustes_align). A genuine low-rank structure reproduces across
    halves (overlap -> 1); an isotropisation artefact does not (overlap -> floor),
    because the near-degenerate block's leading directions are set by residual
    noise that differs between halves.

    Returns mean and std of the overlap across splits.
    """
    rng = np.random.default_rng(rng_seed)
    K = len(Vk_list)
    Wc = np.array([V.T @ W @ V for V, W in zip(Vk_list, Wk_list)])
    ov = []
    for _ in range(n_splits):
        idx = rng.permutation(K)
        h1, h2 = idx[: K // 2], idx[K // 2:]
        U1 = svd(Wc[h1].mean(0))[0][:, :R]
        U2 = svd(Wc[h2].mean(0))[0][:, :R]
        ov.append(subspace_overlap(U1, U2))
    return float(np.mean(ov)), float(np.std(ov))


def reproducibility_vs_rank(Vk_list, Wk_list, R_max=8, n_splits=80, rng_seed=0):
    """Split-half reproducibility as a function of the number of retained
    components R. The reliable rank is the largest R for which reproducibility
    stays high; it complements the elbow / rank-sufficiency criteria."""
    Rs = np.arange(1, R_max + 1)
    means, stds = [], []
    for R in Rs:
        m, s = split_half_reproducibility(Vk_list, Wk_list, R, n_splits, rng_seed)
        means.append(m)
        stds.append(s)
    return Rs, np.array(means), np.array(stds)


def recovery_of(Wbar, Ltrue):
    """
    Fraction of the averaged weight's spectral energy in its top-R singular
    values. Permutation- and frame-invariant: a sharp rank-R structure (whether
    or not it was relabelled) gives ~1; an isotropised block gives ~R/d. This is
    what the paper's elbow criterion effectively reads, and unlike a fixed-frame
    subspace overlap it is not fooled by a benign relabelling of the axes.
    """
    R = Ltrue.shape[1]
    s = svd(Wbar, compute_uv=False)
    return s[:R].sum() / (s.sum() + 1e-12)


# --------------------------------------------------------------------------- #
#  Figure: rho tracks nothing; split-half reproducibility does
# --------------------------------------------------------------------------- #
def fig_three_tiers():
    tiers = ["scaling", "permutation", "rotation"]
    labels = ["Tier 1\nscale/gain", "Tier 2\npermutation", "Tier 3\nrotation"]
    metrics = {"rho": [], "rec_naive": [], "reproducible": []}

    for tier in tiers:
        # realistic independent global frames: the diagnostics must be frame-free
        Vk, Sk, Wk, Ltrue, W0, sidx = make_tier_ensemble(
            tier=tier, rng_seed=1, global_frame=True)
        rho = naive_rho(Sk, index=sidx[1])
        Wbar_naive, _ = aligned_average(Vk, Sk, Wk, match=False)
        repro, _ = split_half_reproducibility(Vk, Wk, R=Ltrue.shape[1])
        metrics["rho"].append(rho)
        metrics["rec_naive"].append(recovery_of(Wbar_naive, Ltrue))
        metrics["reproducible"].append(repro)

    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    rho_disp = np.array(metrics["rho"]) / max(metrics["rho"])
    ax.bar(x - w, rho_disp, w, color="lightgrey", edgecolor="k",
           label=r"naive $\rho_r$ (rescaled)")
    ax.bar(x, metrics["rec_naive"], w, color="#b0c4de", edgecolor="k",
           label="recovered rank-$R$ energy")
    ax.bar(x + w, metrics["reproducible"], w, color="#2e8b57", edgecolor="k",
           label="split-half reproducibility (frame-free)")
    for xi, disp, r in zip(x - w, rho_disp, metrics["rho"]):
        ax.text(xi, disp + 0.02, f"{r:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("metric value")
    ax.set_ylim(0, 1.15)
    ax.set_title("The spectral ratio tracks none of the tiers;\n"
                 "frame-free split-half reproducibility tracks reliability")
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig("fig_three_tiers.png", dpi=140)
    plt.close(fig)
    print("tier         rho    rankR_energy  split_half_repro")
    for t, r, rn, rp in zip(tiers, metrics["rho"], metrics["rec_naive"],
                            metrics["reproducible"]):
        print(f"{t:11s} {r:6.2f}    {rn:6.3f}        {rp:6.3f}")


# --------------------------------------------------------------------------- #
#  Real-data diagnostic (frame-free)
# --------------------------------------------------------------------------- #
def analyse_real_simulations(H_list, W_list, R_report=3, R_max=8,
                             make_fig=True, fname="fig_realdata_diagnostic.png"):
    """
    Frame-free diagnostics on the ACTUAL per-simulation data.

    Parameters
    ----------
    H_list : list of (p x N) end-of-sequence activity matrices H^(k) (Methods 4.3).
    W_list : list of (N x N) recurrent weight matrices W_h^(k), same order as H_list.
             Required because the only frame-free object is the conjugated weight
             Wc_k = V_k^T W_k V_k; the activity vectors alone cannot be compared
             across independently trained networks (see procrustes_align warning).
    R_report : structured rank claimed in the manuscript (annotated).

    Reports:
      - the activity spectrum as FRACTION OF VARIANCE, s_r^2 / sum s^2 (matching the
        manuscript's spectrum figure), with the across-simulation spread; because
        this is a ratio it is already invariant to a per-simulation gain (tier 1),
        so any residual spread here is NOT benign scale;
      - the benign-scale fraction of the RAW singular-value spread, for reference;
      - split-half reproducibility of the recovered rank-R structure vs R -- the
        frame-free reliability test that an isotropisation artefact fails.
    """
    Vk_list, Sk_list = [], []
    for H in H_list:
        _, s, Vt = svd(H, full_matrices=False)
        Vk_list.append(Vt.T)
        Sk_list.append(s)
    ncomp = len(Sk_list[0])

    # variance fractions (scale-invariant -> tier-1 gain already removed)
    frac = np.vstack([s**2 / (s**2).sum() for s in Sk_list])
    frac_mean = frac.mean(0)
    frac_sd = frac.std(0)

    # benign-scale fraction of the RAW spread, for reference
    sigma_raw, sigma_norm = gain_normalised_spread(Sk_list)
    benign = 1.0 - np.nan_to_num(sigma_norm / (sigma_raw + 1e-12))

    # frame-free reliability: split-half reproducibility vs R
    Rs, repro, repro_sd = reproducibility_vs_rank(
        Vk_list, W_list, R_max=R_max)

    print("\nBenign-scale fraction of the raw singular-value spread "
          f"(top {R_report}): {100*benign[:R_report].mean():.1f}%")
    print("Split-half reproducibility of the recovered structure:")
    for R, m, sd in zip(Rs, repro, repro_sd):
        flag = "  <- reported R" if R == R_report else ""
        print(f"  R={R}: {m:.3f} +/- {sd:.3f}{flag}")

    if make_fig:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        idx = np.arange(1, ncomp + 1)
        ax = axes[0]
        ax.fill_between(idx, np.maximum(frac_mean - frac_sd, 1e-9),
                        frac_mean + frac_sd, color="grey", alpha=0.3,
                        label=r"$\pm$ across-simulation SD")
        ax.plot(idx, frac_mean, "k-", lw=1.3, label="mean")
        ax.set_yscale("log")
        ax.set_xlabel("activity component index")
        ax.set_ylabel("fraction of variance  $s_r^2/\\sum s^2$")
        ax.axvspan(0.5, R_report + 0.5, color="orange", alpha=0.12)
        ax.set_title("Activity spectrum (matches manuscript units)")
        ax.legend(frameon=False, fontsize=8)

        ax = axes[1]
        ax.errorbar(Rs, repro, yerr=repro_sd, fmt="o-", color="#2e8b57",
                    capsize=3, lw=1.5)
        ax.axvline(R_report, color="orange", lw=6, alpha=0.25)
        ax.axhline(0.9, color="grey", ls=":", lw=1)
        ax.set_xlabel("retained rank $R$")
        ax.set_ylabel("split-half subspace reproducibility")
        ax.set_ylim(0, 1.05)
        ax.set_title("Frame-free reliability of the recovered structure")
        fig.tight_layout()
        fig.savefig(fname, dpi=140)
        plt.close(fig)
        print(f"\nsaved {fname}")

    return dict(frac_mean=frac_mean, frac_sd=frac_sd, benign_frac=benign,
                repro_R=Rs, repro=repro, repro_sd=repro_sd)


def _synth_H_from(Vk, Sk, p, rng):
    """Build a (p x N) activity matrix whose right singular vectors are Vk and
    singular values Sk (random left factor), so SVD recovers (Vk, Sk)."""
    N = Vk.shape[0]
    A = rng.standard_normal((p, N))
    A, _ = qr(A)
    A = A[:, :N]
    return (A * Sk[None, :]) @ Vk.T


def _demo_real_on_surrogate():
    """
    Demonstrate analyse_real_simulations on surrogate data with independent global
    frames and a genuine low-rank structure, contrasting a reliable case (structure
    in resolved components) with a degenerate case (structure rotated in a flat
    band). Replace with the real H^(k) and W_h^(k) to run on data.
    """
    rng = np.random.default_rng(7)
    p = 400
    for tag, tier in [("reliable", "scaling"), ("degenerate", "rotation")]:
        Vk, Sk, Wk, Lt, W0, si = make_tier_ensemble(
            tier=tier, rng_seed=3, global_frame=True)
        H_list = [_synth_H_from(V, S, p, rng) for V, S in zip(Vk, Sk)]
        print(f"\n=== surrogate: {tag} case ({tier}) ===")
        analyse_real_simulations(H_list, Wk, R_report=3,
                                 fname=f"fig_realdata_{tag}.png")


if __name__ == "__main__":
    print("Constructing the three tiers and testing the diagnostics...\n")
    fig_three_tiers()
    print("\nDemonstrating the real-data diagnostic on surrogate data"
          " (swap in real H^(k) and W_h^(k) to run on data)...")
    _demo_real_on_surrogate()
    print("\nDone.")
