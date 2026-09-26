"""
Reproduce Fig. 2C / 2D of the paper for a range of sequence lengths.

For each L (suffix length; total input length = cue_size + L) this:

  1. loads Win, Wh, Wout from every sim*.pth in the trained run folder,
  2. aligns them into a common coordinate frame (Methods Eq. 16):
         R^(k)      = V^(k).T          from  H^(k) = U S V.T   (final hidden acts)
         Win_bar    = R^(k) @ Win^(k)
         Wh_bar     = R^(k) @ Wh^(k) @ R^(k).T
         Wout_bar   = Wout^(k) @ R^(k).T
  3. plots the aligned dominant blocks, one row per simulation (Fig. 2C),
  4. averages the aligned matrices across sims and plots the mean matrices
     and their singular-value spectra (Fig. 2D).

Usage
-----
    python plot_fig2c.py                # L = 0..10, folder from --folder / par
    python plot_fig2c.py --L 0 1 2 3
    python plot_fig2c.py --folder test_rollout --rotation hidden
"""

import argparse
import os
from glob import glob
from os.path import basename, isdir, join

import matplotlib.pyplot as plt
import numpy as np

import par
from plot_weights import (
    align_weights_multi_sim,
    average_aligned_weights,
    load_types_set,
    load_weights_multi_sim,
    plot_mean_aligned_weights,
)

# dominant blocks kept for the flattened Fig-2C view
N_IN, N_REC, N_OUT = 3, 5, 5


# --------------------------------------------------------------------------- #
# Fig 2C : aligned dominant blocks, one row per simulation
# --------------------------------------------------------------------------- #
def _flatten_blocks(all_Win_p, all_Wh_p, all_Wout_p):
    """(title, X, n_blocks, block_width) for Win, Wh, Wout; X is (n_sim, flat entries)."""
    return [
        (r"$\bar{W}_{\mathrm{in}}^{(k)}$",
         np.stack([w[:N_IN].ravel() for w in all_Win_p]), N_IN, all_Win_p[0].shape[1]),
        (r"$\bar{W}_{\mathrm{h}}^{(k)}$",
         np.stack([w[:N_REC, :N_REC].ravel() for w in all_Wh_p]), N_REC, N_REC),
        (r"$\bar{W}_{\mathrm{out}}^{(k)\top}$",
         np.stack([w.T[:N_OUT].ravel() for w in all_Wout_p]), N_OUT, all_Wout_p[0].shape[0]),
    ]


def plot_aligned_blocks_per_sim(all_Win_p, all_Wh_p, all_Wout_p, figs_dir,
                                L=None, types_set=None, cmap="PiYG", vmax=4.0):
    """Fig 2C: rows = simulation id, cols = flattened dominant weight entries."""
    specs = _flatten_blocks(all_Win_p, all_Wh_p, all_Wout_p)

    fig, axs = plt.subplots(1, 3, figsize=(9, 2.6), constrained_layout=True)
    for ax, (title, X, n_blocks, block_w) in zip(axs, specs):
        im = ax.imshow(X, aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Simulation ID  $k$")
        ax.set_yticks(range(X.shape[0]))
        ax.set_xticks([])
        for i in range(1, n_blocks):
            ax.axvline(i * block_w - 0.5, ls="--", c="k", lw=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    _suptitle(fig, "Fig 2C  aligned dominant blocks", L, types_set, len(all_Win_p))
    _save(fig, figs_dir, _name("fig2C_aligned_blocks", L))


# --------------------------------------------------------------------------- #
# Fig 2D : singular-value spectra of the aligned mean recurrent weights
# --------------------------------------------------------------------------- #
def plot_mean_spectra(all_Win_p, all_Wh_p, all_Wout_p, figs_dir,
                      L=None, types_set=None, n_components=12, shade=3.5):
    mean_Win, mean_Wh, mean_Wout = average_aligned_weights(all_Win_p, all_Wh_p, all_Wout_p)

    panels = [
        ("input  $W_{\\mathrm{in}}$", all_Win_p, mean_Win),
        ("recurrent  $W_{\\mathrm{h}}$", all_Wh_p, mean_Wh),
        ("output  $W_{\\mathrm{out}}$", all_Wout_p, mean_Wout),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    for ax, (title, all_W, mean_W) in zip(axs, panels):
        S_ind = np.stack([np.linalg.svd(w, compute_uv=False) for w in all_W])
        k = min(n_components, S_ind.shape[1])
        x = np.arange(1, k + 1)
        ax.axvspan(0, shade, color="wheat", alpha=0.5, lw=0)
        ax.plot(x, S_ind.mean(0)[:k], "--", c="b", lw=2, label=r"$\langle$SV$(W^{(k)})\rangle$")
        ax.fill_between(x, np.percentile(S_ind, 5, 0)[:k],
                        np.percentile(S_ind, 95, 0)[:k], color="b", alpha=0.2, lw=0)
        S_mean = np.linalg.svd(mean_W, compute_uv=False)[:k]
        ax.plot(x, S_mean, "-o", c="r", ms=3, lw=1, label=r"SV$(\langle \bar{W}\rangle)$")
        ax.set_title(title)
        ax.set_xlabel("index")
        ax.set_ylabel("singular value")
    axs[0].legend(fontsize=7)
    _suptitle(fig, "Fig 2D  spectra of aligned mean weights", L, types_set, len(all_Win_p))
    _save(fig, figs_dir, _name("fig2D_mean_spectra", L))
    return mean_Win, mean_Wh, mean_Wout


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _name(stem, L):
    return f"{stem}_L{L}.svg" if L is not None else f"{stem}.svg"


def _suptitle(fig, what, L, types_set, n_sim):
    bits = [what]
    if L is not None:
        bits.append(f"L={L} (seq len {par.cue_size + L})")
    bits.append(f"{n_sim} sims")
    if types_set:
        bits.append(", ".join(types_set))
    fig.suptitle("  |  ".join(bits), fontsize=9)


def _save(fig, figs_dir, filename):
    os.makedirs(figs_dir, exist_ok=True)
    path = join(figs_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path}")


def _count_sims(data_dir, classcomb_idx):
    return len(glob(join(data_dir, f"model_state_classcomb{classcomb_idx}_sim*.pth")))


# --------------------------------------------------------------------------- #
# per-L driver
# --------------------------------------------------------------------------- #
def run_for_L(L, figs_root, min_sims=2, epoch_time_ids=(-1, -1)):
    data_dir = join(par.folder, par.run_folder_name(L=L))
    if not isdir(data_dir):
        print(f"  L={L}: folder missing, skip  ({data_dir})")
        return None
    n_sim = _count_sims(data_dir, par.classcomb)
    if n_sim < min_sims:
        print(f"  L={L}: only {n_sim} sim(s), skip (need >= {min_sims})")
        return None
    print(f"  L={L}: {n_sim} sims  ->  {basename(data_dir)}")

    all_Win, all_Wh, all_Wout = load_weights_multi_sim(data_dir=data_dir, L=L)
    all_Win_p, all_Wh_p, all_Wout_p = align_weights_multi_sim(
        all_Win, all_Wh, all_Wout, data_dir=data_dir, L=L,
        epoch_time_ids=epoch_time_ids,
    )
    types_set = load_types_set(data_dir=data_dir, L=L)
    figs_dir = join(figs_root, basename(data_dir))

    plot_aligned_blocks_per_sim(all_Win_p, all_Wh_p, all_Wout_p, figs_dir,
                                L=L, types_set=types_set)
    mean_Win, mean_Wh, mean_Wout = plot_mean_spectra(
        all_Win_p, all_Wh_p, all_Wout_p, figs_dir, L=L, types_set=types_set
    )
    plot_mean_aligned_weights(mean_Win, mean_Wh, mean_Wout, figs_dir,
                              L=L, types_set=types_set, n_simulations=len(all_Win))
    return dict(L=L, n_sim=len(all_Win), types_set=types_set,
                mean_Win=mean_Win, mean_Wh=mean_Wh, mean_Wout=mean_Wout,
                all_Win_p=all_Win_p, all_Wh_p=all_Wh_p, all_Wout_p=all_Wout_p)


# --------------------------------------------------------------------------- #
# Fig 2C across a range of L  (rows = L, one row-block of sims per L)
# --------------------------------------------------------------------------- #
def plot_aligned_blocks_L_range(results, figs_root, cmap="PiYG",
                                filename="fig2C_aligned_blocks_L_range.svg"):
    """Combined Fig 2C: one row per L, three columns (Win, Wh, Wout)."""
    results = [r for r in results if r is not None]
    if not results:
        print("  no L runs for combined Fig 2C, skipping")
        return
    nL = len(results)
    per_L = [_flatten_blocks(r["all_Win_p"], r["all_Wh_p"], r["all_Wout_p"]) for r in results]

    # a shared symmetric colour scale per column (Win/Wh/Wout differ in magnitude)
    col_vmax = [
        max(np.percentile(np.abs(specs[j][1]), 99) for specs in per_L) or 1.0
        for j in range(3)
    ]

    fig, axs = plt.subplots(nL, 3, figsize=(9, 1.5 * nL + 0.8),
                            squeeze=False, constrained_layout=True)
    col_im = [None, None, None]
    for i, (r, specs) in enumerate(zip(results, per_L)):
        for j, (title, X, n_blocks, block_w) in enumerate(specs):
            ax = axs[i][j]
            col_im[j] = ax.imshow(X, aspect="auto", interpolation="nearest", cmap=cmap,
                                  vmin=-col_vmax[j], vmax=col_vmax[j])
            for b in range(1, n_blocks):
                ax.axvline(b * block_w - 0.5, ls="--", c="k", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks(range(X.shape[0]))
            ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(title, fontsize=11)
            if j == 0:
                ax.set_ylabel(f"L={r['L']}  (seq {par.cue_size + r['L']})\nsim $k$",
                              fontsize=7)
    for j in range(3):
        fig.colorbar(col_im[j], ax=axs[:, j], location="bottom",
                     fraction=0.03, pad=0.02, shrink=0.9)
    fig.suptitle("Fig 2C  aligned dominant weight blocks across L", fontsize=12)
    _save(fig, figs_root, filename)


# --------------------------------------------------------------------------- #
# summary across L
# --------------------------------------------------------------------------- #
def plot_spectrum_vs_L(results, figs_root, n_components=12):
    results = [r for r in results if r is not None]
    if not results:
        return
    fig, ax = plt.subplots(figsize=(4, 3), constrained_layout=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(results)))
    for c, r in zip(cmap, results):
        S = np.linalg.svd(r["mean_Wh"], compute_uv=False)[:n_components]
        ax.plot(np.arange(1, len(S) + 1), S, "-o", ms=3, c=c,
                label=f"L={r['L']}")
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"SV of $\langle \bar{W}_{\mathrm{h}}\rangle$")
    ax.set_title("aligned mean recurrent spectrum vs L")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, figs_root, "fig2D_mean_Wh_spectrum_vs_L.svg")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=None,
                    help="run root (default: par.folder; falls back to test_rollout)")
    ap.add_argument("--L", type=int, nargs="+", default=list(range(0, 11)))
    ap.add_argument("--rotation", choices=["weights", "hidden"], default="weights",
                    help="align via Wh right-singular vectors ('weights', default, "
                         "cleaner here) or final-hidden-activity SVs ('hidden', paper Eq.16)")
    ap.add_argument("--classcomb", type=int, default=None)
    ap.add_argument("--min-sims", type=int, default=2)
    args = ap.parse_args()

    if args.folder:
        par.folder = args.folder
    elif isdir("test_rollout"):
        par.folder = "test_rollout"
    par.common_rotation = args.rotation
    if args.classcomb is not None:
        par.classcomb = args.classcomb

    figs_root = join(par.folder, "figs")
    print(f"folder   : {par.folder}")
    print(f"rotation : {par.common_rotation}")
    print(f"L values : {args.L}")
    print(f"figs     : {figs_root}\n")

    results = [run_for_L(L, figs_root, min_sims=args.min_sims) for L in args.L]
    plot_aligned_blocks_L_range(results, figs_root)
    plot_spectrum_vs_L(results, figs_root)
    done = [r["L"] for r in results if r is not None]
    print(f"\ndone for L = {done}")
    print(f"figures under {figs_root}/")


if __name__ == "__main__":
    main()
