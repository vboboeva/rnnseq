"""
Run all analysis plots for a range of sequence lengths L.

For each L (default: every unique L in parameters.txt) this:
  1. runs lowrank.py  (full per-run analysis plots for that L)
  2. makes the Win/Wh/Wout multi-sim panel + mean aligned panel
  3. makes the Fig 2C aligned-blocks panel + Fig 2D mean-spectra panel

Then once at the end:
  * combined dimensionality grid   SVDs_and_dimensionality_allL{min}toL{max}.svg
  * combined Fig 2C across L        fig2C_aligned_blocks_L_range.svg
  * mean recurrent spectrum vs L    fig2D_mean_Wh_spectrum_vs_L.svg

Missing / still-training L folders are skipped with a warning instead of crashing,
so this is safe to re-run while a training job is in progress.

Usage
-----
    python run_all_plots.py                              # L from parameters.txt, par.folder
    python run_all_plots.py --folder test_rollout
    python run_all_plots.py --folder test_rollout --L 0 1 2 3 4 5 6
    python run_all_plots.py --folder test_rollout --skip-lowrank   # only the weight/Fig2C plots
"""

import argparse
import runpy
import sys
import traceback
from glob import glob
from os.path import basename, isdir, join

import par
from lowrank_utils import (
    L_values_from_parameters,
    plot_SVDs_dimensionality_all_LMin_to_LMax,
)
from plot_weights import (
    plot_weights_multi_sim_from_run,
    plot_mean_aligned_weights_from_run,
)
import plot_fig2c


EPOCH_TIME_IDS = (-1, -1)


def _run_dir(L):
    return join(par.folder, par.run_folder_name(L=L))


def _n_sims(L):
    return len(glob(join(_run_dir(L), f"model_state_classcomb{par.classcomb}_sim*.pth")))


def run_lowrank_for_L(L):
    """Execute lowrank.py as a script after setting par.L."""
    par.L = L
    sys.modules.pop('lowrank', None)        # force the top-level body to re-run
    runpy.run_path('lowrank.py', run_name='__main__')


def main(L_values=None, folder=None, epoch_time_ids=EPOCH_TIME_IDS,
         min_sims=2, skip_lowrank=False):
    if folder:
        par.folder = folder

    if L_values is None:
        L_values = L_values_from_parameters()

    figs_root = join(par.folder, 'figs')
    print(f"Data root : {par.folder}")
    print(f"L values  : {L_values}")
    print(f"figs      : {figs_root}\n")

    fig2c_results = []
    for L in L_values:
        run_dir = _run_dir(L)
        if not isdir(run_dir):
            print(f"--- L={L}: folder missing, skip ({run_dir})")
            continue
        n = _n_sims(L)
        if n < min_sims:
            print(f"--- L={L}: only {n} sim(s), skip (need >= {min_sims})")
            continue

        print(f"\n{'=' * 60}\nL={L}  ({n} sims)  {basename(run_dir)}\n{'=' * 60}")

        if not skip_lowrank:
            try:
                run_lowrank_for_L(L)
            except Exception:
                print(f"!!! lowrank.py failed for L={L}:")
                traceback.print_exc()

        try:
            plot_weights_multi_sim_from_run(L=L)
            plot_mean_aligned_weights_from_run(L=L, epoch_time_ids=epoch_time_ids)
        except Exception:
            print(f"!!! weight panels failed for L={L}:")
            traceback.print_exc()

        try:
            fig2c_results.append(
                plot_fig2c.run_for_L(L, figs_root, min_sims=min_sims,
                                     epoch_time_ids=epoch_time_ids)
            )
        except Exception:
            print(f"!!! Fig 2C/2D failed for L={L}:")
            traceback.print_exc()

    print(f"\n{'=' * 60}\nCombined plots across L\n{'=' * 60}")
    try:
        dim_filename = plot_SVDs_dimensionality_all_LMin_to_LMax(
            L_values=L_values, figs_dir=figs_root, epoch_time_ids=epoch_time_ids,
        )
    except Exception:
        dim_filename = None
        print("!!! combined dimensionality plot failed:")
        traceback.print_exc()

    plot_fig2c.plot_aligned_blocks_L_range(fig2c_results, figs_root)
    plot_fig2c.plot_spectrum_vs_L(fig2c_results, figs_root)

    done = [r["L"] for r in fig2c_results if r is not None]
    print(f"\nDone for L = {done}")
    print(f"  {figs_root}/")
    if dim_filename:
        print(f"    {dim_filename}")
    print(f"    fig2C_aligned_blocks_L_range.svg")
    print(f"    fig2D_mean_Wh_spectrum_vs_L.svg")
    print(f"  per-L SVGs live in the L* subfolders of figs/")


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", default=None,
                    help="run root (default: par.folder as set in par.py)")
    ap.add_argument("--L", type=int, nargs="+", default=None,
                    help="L values (default: unique L in parameters.txt)")
    ap.add_argument("--min-sims", type=int, default=2,
                    help="skip an L folder with fewer than this many sim*.pth (default 2)")
    ap.add_argument("--skip-lowrank", action="store_true",
                    help="only make the weight / Fig 2C / Fig 2D plots, not the full lowrank.py set")
    return ap.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(L_values=args.L, folder=args.folder,
         min_sims=args.min_sims, skip_lowrank=args.skip_lowrank)
