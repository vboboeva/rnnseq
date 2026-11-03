# Quantifying the consistent u₁·v₂ overlap in recurrent-weight SVDs

> **Status 2026-11-03: implemented** in `sv_overlap.py` (+ wiring in `lowrank.py`, flag
> `par.compute_sv_overlap`). Run: `python sv_overlap.py --selftest` and
> `python sv_overlap.py --data-dir outBiasTrue/<folder>`. Where the text below disagrees
> with this box, **this box wins** — these are corrections found while building.
>
> 1. **Wrong null number (mine).** "Null max of a(1) = 0.230" is the null *mean* of a(1).
>    The true max over 20 000 Haar draws is **0.417** (q99 0.308). Observed `min a(1) = 0.472`
>    still exceeds it in 100/100 sims, but the margin is small, not 2×.
> 2. **Wrong comparison for a(2) (mine).** "a(2) beats the null max in 84/100" compared a(2)
>    to the a(1) null. Against a(2)'s own null: above its q99 (0.254) in **75/100**, above
>    its max (0.302) in **66/100**. The "usually a second above-chance partner" conclusion
>    survives; the number was wrong. Each order statistic is now compared to its own null.
> 3. **Cross-sim null is not "primary".** Sims have independent unit coordinates, so
>    `Vh⁽ᵃ⁾U⁽ᵇ⁾` is Haar-like: measured cross-sim `a(1..5)` = `0.230 0.204 0.190 0.181 0.173`
>    vs Haar `0.230 0.205 0.191 0.181 0.174`. It is kept as a check, but the Haar null is
>    the operative one. For the same reason there is no cross-sim reference vector, so
>    `gauge_fix_all(refs)` was replaced by `gauge_fix_by_column` (gauge convention:
>    column u₁ ≥ 0).
> 4. **Controls are not flat.** Columns u₂…u₅ are weakly localized too (a(1) above null max
>    in 14–24% of sims, participation ratio 24–34 vs 55), with winners at v₁₃–v₁₈ — not a
>    clean "only u₁ is special". u₁ is far stronger (PR 4.0, a(1) 0.69) but the control
>    panel should be read as graded, not binary.
> 5. **Calibration**: spectrum-matched Haar surrogates give a false-positive rate of 4.75%
>    at α=0.05 (n=400; an earlier n=100 run gave 1%, which was noise).
> 6. **Not yet done**: replication on `test_ablate_sv` / N=320 / N=64 / `initRich`;
>    `lowrank.py` in this worktree still looks for a `_noise0.000` folder suffix that the
>    `outBiasTrue` directories do not have, so the wiring there is untested end-to-end.
>    Observed in the a(1)-vs-a(2) scatter: the two are strongly anti-correlated, which
>    hints that u₁ rotates within a (v₂,v₃) plane at a sim-dependent angle — a hypothesis,
>    untested.

Target: worktree `/ceph/akrami/Vizhe/RNN_worktrees/uv-proj`, branch `uv-proj`.
Style: **exploratory diagnostic first** — get the statistics right and readable, polish later.

## Context

100 seeds of an RNN with recurrent matrix `h2h` of size N=160. Writing `W = U S Vᵗ`,
the spectrum is highly reproducible across seeds but the singular vectors are not.
The existing figure `recurrent_V_dot_U.svg` (`plot_alignment_SVs`,
`lowrank_utils.py:148`) plots `M = Vh @ U`, entry `M[i,j] = v_i·u_j`, for one sim and
as a flattened sim×entry raster. By eye `|M[1,0]| = |v₂·u₁|` is consistently large
(occasionally `|v₃·u₁|`), which would mean `W` copies the mode-1 readout direction into
the mode-2 input direction. Averaging `M` over seeds destroys it, and commit 311e572
says "need to fix alignment".

### This is a gauge symmetry, not an alignment bug

The SVD is only defined up to a per-mode sign flip `(u_k, v_k) → (−u_k, −v_k)`, i.e.
`U → U D`, `V → V D` with `D = diag(±1)`. Hence

```
M = Vᵗ U  →  D M D        i.e.   M[i,j] → d_i d_j M[i,j]
```

Every **off-diagonal** entry of `M` carries a sign that is pure gauge, so
`mean_sims M[i,j] → 0` by construction regardless of how many seeds are run. No choice
of reference vector can fix this. The analysis must be built from **gauge-invariant**
statistics plus an explicit null.

The current code also compounds it: `lowrank.py:146` feeds the *unaligned*
`all_rec_Us`/`all_rec_Vhs` into `Vh @ U`, and `align_singular_vectors`
(`lowrank_utils.py:68`) sign-aligns `U` and `V` to a hidden-activity reference
*independently*, applying different sign vectors to each — which breaks the gauge
rather than fixing it.

### The effect is real and large — measured on both datasets (100 sims, N=160)

0-indexed throughout: `u₁ = U[:,0]`, `v₂ = Vh[1,:]`, so the entry of interest is `M[1,0]`.
Chance level is `E|M[i,j]| = sqrt(2/(πN)) = 0.0631`.

**A. `outBiasTrue/TaskRNNClass_N160_..._nepochs101_ntypes7_fractrain1.0_initNone_..._datasplit0`**
(the worktree's own data — the primary config):

```
mean |M| (rows = V id, cols = U id)
        u0     u1     u2     u3     u4     u5
 v0   0.050  0.069  0.064  0.050  0.049  0.051
 v1   0.534  0.126  0.107  0.079  0.060  0.076
 v2   0.398  0.104  0.131  0.076  0.081  0.068
 v3   0.186  0.079  0.083  0.170  0.069  0.060
 v4   0.171  0.065  0.075  0.058  0.176  0.063
 v5   0.131  0.073  0.076  0.055  0.052  0.185

naive signed mean: M[1,0] = -0.021, M[2,0] = -0.063   <- near-total cancellation
S mean[:6] = [7.10 5.27 4.90 4.66 4.50 4.37],  sd = [0.25 0.25 0.19 0.16 0.14 0.14]
argmax_i |M[i,0]|:  i=1 in 59/100,  i=2 in 33/100,  i=3 in 6,  i=4 in 2
```

**B. `test_ablate_sv/TaskRNNClass_N160_..._fractrain1.0_..._maxrankNone_0`** (main tree,
for comparison): `mean|M[1,0]| = 0.717` (≈11× chance), naive signed mean `−0.153`,
`argmax_i |M[i,0]|` = 86/10/4 at i=1/2/3, `S mean[:6] = [6.75 4.47 4.14 3.96 3.82 3.72]`.

### The claim is about *within-sim localization* — so sort within sim, then aggregate

The question is not "how big is entry (1,0) on average" but "in each seed, does `u₁` have
one dominant partner?" Averaging at a fixed index answers the wrong question and
*understates* the effect, because it mixes seeds where v₂ won with seeds where v₃ did.
The correct operation is to sort the column `a_i = |v_i·u₁|` **within each sim** and
aggregate the order statistics. Measured on dataset A (all N=160 indices, not just the
top 10 — note `Σ_i M[i,0]² = 1` exactly since `M` is orthogonal):

```
                   a(1)   a(2)   a(3)   a(4)   a(5)   a(6)   a(7)   a(8)
mean               0.693  0.362  0.234  0.182  0.148  0.128  0.117  0.107
sd                 0.106  0.115  0.079  0.055  0.040  0.030  0.024  0.020
5th pct            0.525  0.181  0.122  0.109  0.097  0.089  0.083  0.077
Haar null (mean)   0.230  0.205  0.191  0.181  0.174   ...  <- essentially FLAT

min over sims of a(1) = 0.472        null max  a(1) = 0.230
top-1 squared mass a(1)^2: mean 0.492, median 0.483, 5th pct 0.276
cumulative squared mass, top 1/2/3/5/10: 0.492 0.636 0.697 0.757 0.819
participation ratio 1/Σp²  (p = normalized squared column):
    observed  mean 3.97, median 3.59, 5-95pct [1.94, 7.45]        null 54.7
dominance ratio a(1)/a(2): mean 2.27, median 1.82, min 1.01; >1.5 in 65%, >2 in 46%
count of v above Bonferroni null threshold 0.281:
    ==1 in 30/100,  ==2 in 39/100,  ==3 in 28/100,  ==4 in 3/100   (median 2)
Spearman[(s2-s3)/s2 , a(1)/a(2)] = 0.071, p = 0.48
```

Seven things fall out, all of which the plan must capture:

1. **One dominant partner, in every single seed.** `a₍₁₎ ≥ 0.472` in 100/100 sims against
   a null *maximum* of 0.230 — the right comparison for an order statistic, and the
   relevant chance level here (not the per-entry 0.063). The best partner alone carries
   **49%** of `u₁`'s squared norm. The null rank profile is flat, so the cliff is real.
2. **The profile is sharply localized**: participation ratio ≈ 4.0 vs 54.7 under chance,
   i.e. `u₁` effectively touches ~4 of 160 right singular vectors instead of ~N/3.
3. **"Only ONE" is not supported, and should not be claimed.** `a₍₂₎ = 0.362` also
   exceeds the null maximum, in 84/100 sims; counting partners above a Bonferroni
   threshold gives exactly one in only **30/100** seeds (median 2), and `a₍₁₎/a₍₂₎` is
   below 1.3 in 19/100. The defensible statement is **"one clearly dominant partner
   (~1.9× the runner-up, 49% of the mass), usually accompanied by a second above-chance
   one"**. The figures must show the rank profile so a reader can see this directly
   rather than take a single number on trust.
4. **Near-degeneracy does NOT explain the v₂/v₃ swap** — Spearman ρ = 0.071, p = 0.48
   between the `(s₂−s₃)/s₂` gap and `a₍₁₎/a₍₂₎`; near-tied seeds have mean gap 0.061 vs
   0.070 for the rest. So the 59/33 split and the second partner are **reproducible
   structure, not a labelling artefact of a degenerate block.** This kills the
   degeneracy hypothesis and makes the second partner worth characterising in its own
   right rather than explaining away.
5. The naive signed mean collapses to `−0.021` / `−0.153`, confirming the gauge
   diagnosis quantitatively. On dataset A the cancellation is essentially total.
6. **Subspace overlap is the wrong headline here and is demoted.** `W` is known to be
   close to low-rank, so `span{v₂,v₃,v₄}` is where the mass lives regardless and
   `O[0,{1,2,3}] = 0.79` is near-vacuous — worse, it *hides* the within-sim one-ness
   that is the actual claim. Keep it only as a one-line sanity figure with that caveat
   stated.
7. The **diagonal** `M[i,i]` is elevated *and* consistently negative (`−0.05 … −0.18`).
   Diagonal entries are gauge-invariant *including* their sign, so this genuinely
   survives averaging and should be reported alongside.

Goal: a figure + numbers stating "in 100/100 seeds `u₁` has a single dominant partner
with `|v·u₁| ≥ 0.47` (chance max 0.23), carrying 49% of `u₁` and ~1.9× the runner-up;
the overlap profile has an effective width of ~4 of 160 modes vs ~55 under chance; the
dominant partner is v₂ in 59/100 seeds and v₃ in 33/100, and this is not explained by
near-degeneracy of s₂,s₃."

## Gauge-invariant statistics

All of these are computed **per sim first**, then aggregated — never averaged at a fixed
index, which is what destroyed the effect in the original figure.

| Quantity | Definition | Invariant? | What it answers |
|---|---|---|---|
| `a₍k₎` | k-th largest of `\|v_i·u_j\|` over i, within a sim | ✅ sign cancels; index-free | **Primary / headline.** The rank profile. A cliff at k=1→2 *is* the one-dominant-partner claim, shown per-sim with no averaging. |
| `a₍₁₎/a₍₂₎` | dominance ratio | ✅ | How much the top partner beats the runner-up. Distribution over seeds, honestly showing the 19/100 near-ties. |
| `PR` | `(Σ_i a_i²)² / Σ_i a_i⁴` = `1/Σ p_i²` with `p_i = M[i,j]²` | ✅ | **Effective number of v's that `u_j` touches.** Normalized (`Σ_i M[i,j]² = 1`), so directly comparable to the null `≈N/3`. |
| `n_sig` | count of i with `\|v_i·u_j\|` above the null Bonferroni threshold | ✅ | The literal "how many partners?" count — the statistic that adjudicates "only ONE". |
| `absM[i,j]` | `\|v_i·u_j\|` | ✅ sign cancels | Entry-level strength; needed for the *identity* of the partner (the 59/33 split), kept separate from its magnitude. |
| `C[i,j]` | `M[i,j]·M[j,i]` (`M ⊙ Mᵗ`) | ✅ signed | Is the coupling symmetric (>0) or antisymmetric (<0)? Signed ⇒ still averages. |
| `M[i,i]` | `v_i·u_i` | ✅ signed | Self-alignment; explains the negative diagonal above. |
| `T = S M` | `T[i,j] = s_i·M[i,j]` | ✅ in magnitude | `W U = U T`, so `\|T\|` *is* the copying-gain matrix of `W` in the `U` basis. |
| `O[j,B]` | `sqrt(Σ_{i∈B} M[i,j]²)` | ✅ signs and in-block rotation | Demoted — near-vacuous under low-rank `W`; reported once with that caveat. |

The magnitude question (how strong is the dominant overlap?) and the identity question
(which v is it?) are deliberately kept apart: the first is answered by rank statistics
that need no index, the second by the partner histogram. Conflating them is what made
the original averaged figure uninterpretable.

Interpretation: `W u_j = Σ_i s_i u_i (v_i·u_j) = Σ_i T[i,j] u_i`. So
`|T[1,0]| = s₂·|M[1,0]| ≈ 5.27 × 0.534 ≈ 2.8` is the gain with which one application of
`W` moves activity from `u₁` into `u₂` — the "copying" claim made quantitative, and
comparable against `s₁ = 7.10` on the diagonal. Powers of `T` give multi-step copying,
which is what the recurrent dynamics `h → W h` actually does.

## Null models

1. **Analytic Haar null** (reference lines, closed-form p). For a real Ginibre matrix
   `U` and `V` are independent Haar-O(N), so `M = VᵗU` is Haar on O(N) and each entry is
   a coordinate of a uniform point on `S^{N−1}`: `M[i,j]² ~ Beta(1/2,(N−1)/2)`,
   `E|M[i,j]| = sqrt(2/(πN))`. Exact p: `scipy.stats.beta.sf(m**2, 0.5, (N-1)/2)`.
   For `O[j,B]` with `|B|=b`: `O² ~ Beta(b/2,(N−b)/2)`.
2. **Cross-sim pairing null** — the *primary* empirical null. `M_cross = Vh⁽ᵃ⁾ @ U⁽ᵇ⁾`
   for `a ≠ b`: keeps the true marginal distributions of `U` and of `V` but destroys
   their within-run pairing. This tests exactly the right hypothesis — *is the u₁–v₂
   overlap a property of the paired SVD of one trained network, or an artefact of
   leading singular vectors clustering in a common subspace across seeds?* The repo's
   own "common rotation" result says seeds do share structure, so the analytic null
   alone would be too permissive; this null is what makes the claim honest.
3. **Component-label permutation null** (secondary). Within each sim permute `U`'s
   first-K columns; tests whether the *specific* `(1,0)` pair is special rather than
   off-diagonal overlaps being generically large.
4. **Spectrum-matched surrogate** (calibration self-test). `W' = U' S V'ᵗ` with the
   observed `S` and fresh `ortho_group.rvs(N)`, pushed through the whole pipeline; must
   yield no FDR-significant entries. Verifies the test is calibrated rather than merely
   producing small numbers.

## Approach

New module `sv_overlap.py` in the worktree, imported by its `lowrank.py`. Pure
numpy/scipy — no new dependencies. **`statsmodels` is not installed**, so hand-roll
Benjamini–Hochberg (~6 lines). scipy is 1.7.1: `stats.beta`, `stats.ortho_group`,
`stats.spearmanr` are available; `stats.permutation_test` is **not** (added in 1.8) —
write the permutation loop directly.

### Worktree setup (do this first — the worktree is not currently runnable)

`par.py` is untracked (`.gitignore` is `*`), so the worktree has **no `par.py`** and
`lowrank.py` there dies on `import par`. Also the worktree's `folder_name` ends in
`_noise{train_noise:.3f}` (not `_maxrank{max_rank}`) and its data lives in
`outBiasTrue/`, not `test_ablate_sv/`. So:

- Copy `/ceph/akrami/Vizhe/RNN/par.py` into the worktree and set `folder='outBiasTrue'`,
  `n_epochs=101`, `n_types=7`, `frac_train=1.0`, `init=None`, `train_noise=0.0`,
  `datasplit=0` to match dataset A above (verified: 100 `model_state_classcomb0_sim*.pth`).

### `sv_overlap.py` — computation

- `compute_VU_overlap(all_Us, all_Ss, all_Vhs, n_components=10)` → dict with
  `M (n_sim,K,K)`, `absM`, `cycle2 = M * M.transpose(0,2,1)`, `T = S[:,:K,None]*M`,
  `S`, `N`, **and `M_full (n_sim, N, K)`** — the rank-profile statistics need whole
  columns over all N indices, not the K×K corner, so that `Σ_i M[i,j]² = 1` holds and
  `PR` is comparable to the `N/3` null. Only the first K *columns* are needed, so this
  is `(100,160,10)`, not a full `(100,160,160)` stack.
- `rank_profile(M, col=0)` → **the headline computation.** Per-sim sorted column
  `a₍k₎ = sort_desc(|M[:,col]|)` over **all N** indices; returns the `(n_sim, N)` array
  plus mean/sd/percentiles of each order statistic, `a₍₁₎/a₍₂₎`, `a₍₁₎²`, cumulative
  squared mass, and `PR`. Also `n_sig` = count above the null Bonferroni threshold.
  Everything downstream reads off this. Run for control columns `j=1..4` too, so "mode 1
  is special" is testable rather than assumed.
- `rank_profile_null(N, n_draws=200000, seed=0)` → order statistics of `|g|/‖g‖` for
  Gaussian `g` (exact Haar column), giving the flat null profile, the null max, the
  99.9th percentile and the Bonferroni threshold `1−0.05/N`. **Use the null's order
  statistics, not the per-entry chance level 0.063** — comparing an observed *maximum*
  to a per-entry mean would badly inflate the apparent effect.
- `subspace_overlap(M, col, block)` → per-sim `O[col,block]`, reported **once** with the
  caveat that `W` is near low-rank so this is close to saturated by construction
  (`O[0,{1,2,3}] = 0.79`) and is not evidence for localization. Kept only to forestall
  the question.
- `gauge_fix(U, Vh, ref_U, n_components)` → **one** sign vector `d_k = sign(u_k·ref_k)`
  applied to **both** `U` and `Vh`, so the gauge is respected. This is the corrected
  analogue of `align_singular_vectors` (`lowrank_utils.py:68`); reuse the
  `return_signs=True` variant that already exists on branch `lowrank-deltas`
  (`lowrank_utils.py:69`, commit 50a6462) rather than writing a third copy. Reference =
  per-component across-sim mean direction, refined in two passes so the result does not
  depend on which sim seeds the reference.
- `haar_null_stats(N, K)` → `{mean_abs, pct5, pct95, sf}` from `scipy.stats.beta`,
  for both single entries (`b=1`) and blocks (`b=|B|`).
- `crosssim_null_absM(all_Us, all_Vhs, K, n_draws=20000, seed=0)` → stacked
  `|Vh⁽ᵃ⁾U⁽ᵇ⁾|` over random `a≠b` pairs.
- `fdr_bh(pvals)` → BH q-values.
- `test_overlap_entries(absM, crosssim_null, haar)` → per-entry mean, chance-ratio
  `mean|M| / E_null|M|`, empirical one-sided p computed mean-vs-mean
  (`(1+#{null_mean ≥ obs_mean})/(1+n_draws)`, resampling null means at the same n_sim),
  analytic Haar p, and BH-FDR over the `K(K−1)` off-diagonal entries. Returns a pandas
  DataFrame (pandas 1.3.3 available) for printing/CSV.
- `partner_stats(M, col=0, K=10)` → per-sim `argmax_{i≠col}|M[i,col]|` and its value;
  counts per partner index; binomial p for the modal partner vs the null argmax rate;
  χ² goodness-of-fit against the null argmax distribution. Run for `col=0` (expect
  59/33/6/2 on dataset A) and symmetrically for `row=1`.
- `degeneracy_check(all_Ss, M)` → per-sim relative gap `(s₂−s₃)/s₂` vs `a₍₁₎/a₍₂₎` and
  vs the winning index, Spearman ρ and p. **Already measured: ρ = 0.071, p = 0.48 — the
  degeneracy hypothesis is rejected.** Keep the test in the code as the documented
  falsification (it is the obvious reviewer question), and report the negative result
  explicitly rather than quietly dropping it.
- `surrogate_calibration(all_Ss, K, n=100, seed=0)` → the null self-test of §4.

### `sv_overlap.py` — figures

Repo idiom: `fig, axs = plt.subplots(figsize=...)` → `fig.tight_layout()` →
`fig.savefig(join(FIGS_DIR,'....svg'), dpi=300)` → `plt.close(fig)`; `PiYG` with
symmetric limits for signed matrices, sequential for magnitudes, 5/95 percentile bands.
Exploratory pass: readable multi-panel figures, not yet size-tuned for the paper.

- `plot_VU_overlap_stats(...)` → `recurrent_VU_overlap_stats.svg`, 4 panels:
  (a) `mean|M|` heatmap (K=10) with the chance level marked on the colorbar;
  (b) `−log10 q` heatmap, non-significant entries masked;
  (c) per-sim strip/violin of the top entries (`|M[1,0]|`, `|M[2,0]|`, `|M[3,0]|`, …)
  over the cross-sim null distribution and the Haar 95th percentile;
  (d) mean signed `cycle2` in `PiYG` plus the signed diagonal.
- `plot_rank_profile(...)` → `recurrent_VU_rank_profile.svg` — **the headline figure.**
  Its whole job is to show per-sim one-ness without averaging it away:
  (a) **all 100 sorted curves** `a₍k₎` vs `k` as thin translucent lines, median
  overlaid, with the flat Haar null band and the null-max line. The cliff at k=1→2 is
  visible in every individual seed — this is the panel that replaces the aggregate.
  (b) **sorted raster**, 100 sims × first 10 *ranks*, rows ordered by `a₍₁₎/a₍₂₎`:
  one bright pixel per row, no index confound, no averaging.
  (c) histograms of `a₍₁₎/a₍₂₎`, `PR`, and `n_sig` with their null distributions —
  including the 19/100 near-ties and the `n_sig` median of 2, so the figure states the
  limits of the claim rather than hiding them.
  (d) same rank profile for control columns `j=0..4`, showing mode 1 is the special one.
- `plot_VU_partner(...)` → `recurrent_VU_partner.svg` — the *identity* question, kept
  separate from magnitude: (a) partner-index histogram for `u₁` (59/33/6/2) with null
  band; (b) **index raster**, 100 sims × first 10 *v-indices*, rows grouped by winner —
  shows index-consistency on top of one-ness; (c) scatter `a₍₁₎` vs `a₍₂₎` coloured by
  the `s₂,s₃` gap with the null-max lines marked, which is the panel that visually
  demonstrates the rejected degeneracy explanation (ρ=0.071, p=0.48).
- `plot_mode_transfer(...)` → `recurrent_mode_transfer.svg`: `mean|T|` heatmap (the
  copying-gain matrix) + bar plot of the largest off-diagonal gains with bootstrap CIs.
  This is the panel that states the mechanism.
- `plot_gauge_fixed_mean(...)` → `recurrent_VU_gaugefixed.svg`: signed `mean M` after
  gauge fixing next to the naive cancelling mean, so the figure itself makes the point
  that the cancellation was a gauge artefact.

Also print a summary table (chance-ratio, q, partner counts, subspace overlap) so the
numbers can go straight into a caption.

### `lowrank.py` (worktree) — wiring

The `Vh @ U` computation already exists at `lowrank.py:146`. Add after it:

```python
from sv_overlap import (compute_VU_overlap, rank_profile, rank_profile_null,
                        gauge_fix_all, haar_null_stats, crosssim_null_absM,
                        test_overlap_entries, partner_stats, subspace_overlap,
                        degeneracy_check, plot_rank_profile, plot_VU_overlap_stats,
                        plot_VU_partner, plot_mode_transfer, plot_gauge_fixed_mean)

K = 10
ov    = compute_VU_overlap(all_rec_Us, all_rec_Ss, all_rec_Vhs, n_components=K)
rnull = rank_profile_null(par.n_hidden, seed=0)
rp    = {j: rank_profile(ov['M_full'], col=j, null=rnull) for j in range(5)}
plot_rank_profile(rp, rnull, FIGS_DIR)          # headline

null  = crosssim_null_absM(all_rec_Us, all_rec_Vhs, K, seed=0)
haar  = haar_null_stats(par.n_hidden, K)
tests = test_overlap_entries(ov['absM'], null, haar)
print(tests.to_string())
print(degeneracy_check(all_rec_Ss, ov['M_full']))
plot_VU_partner(ov, rp[0], all_rec_Ss, null, FIGS_DIR)
plot_VU_overlap_stats(ov, tests, null, haar, FIGS_DIR)
plot_mode_transfer(ov, FIGS_DIR)
plot_gauge_fixed_mean(ov, gauge_fix_all(all_rec_Us, all_rec_Ss, all_rec_Vhs, K), FIGS_DIR)
```

Note `all_rec_Ss` is not currently unpacked in the worktree's `lowrank.py` (only
`all_rec_Us` and `all_rec_Vhs` are) — add it from `all_rec_SVDs`. Guard the block with
a new `par.compute_sv_overlap = True` flag, in the style of the existing
`par.compute_uc`, since `lowrank.py` is a flat top-to-bottom script.

### Files

- **new** `sv_overlap.py` — all of the above.
- **new** `par.py` — copied and adjusted (untracked, as elsewhere).
- **modify** `lowrank.py` — ~15 lines after line 146.
- Leave `plot_alignment_SVs` and `align_singular_vectors` in place; the new module
  supersedes the former, and the latter stays for the activity-rotation use it was
  written for.

## Verification

Interpreter: `/nfs/nhome/live/vboboeva/.conda/envs/work/bin/python` (numpy 1.21.2,
scipy 1.7.1, torch 2.0.1, matplotlib 3.4.2, pandas 1.3.3). `/usr/bin/python3` has no
torch — do not use it. Loading 100 models over NFS takes ~3–5 min; cache `M`/`S` to a
pickle on first load.

1. **Gauge-invariance unit test** — in `sv_overlap.py`'s `__main__` (the repo already
   uses this pattern at `lowrank_utils.py:688`): random `W`, SVD, apply a random
   `D = diag(±1)` to `U` and `Vh`, assert `absM`, `cycle2`, `|T|`, `O[j,B]`, `M[i,i]`
   and every reported p-value are unchanged to machine precision. **The most important
   check** — it proves the statistics are gauge-invariant rather than merely plausible.
2. **Null calibration** — `surrogate_calibration()`: 100 spectrum-matched surrogates
   through the full pipeline; assert no off-diagonal entry survives FDR at q<0.05 and
   that the uncorrected false-positive rate is ≈5% at α=0.05. If this fails, nothing
   downstream is trustworthy.
3. **Reproduce the measured numbers** — on dataset A confirm the rank profile
   `a₍₁..₅₎ ≈ [0.693, 0.362, 0.234, 0.182, 0.148]`, `min a₍₁₎ ≈ 0.472`, `a₍₁₎² ≈ 0.492`,
   `PR ≈ 3.97` (null 54.7), `n_sig == 1` in 30/100, partner counts 59/33/6/2, and naive
   signed mean `≈ −0.02`. Any large discrepancy means a bug, not a discovery.
4. **Null-profile sanity** — assert the simulated Haar rank profile is flat
   (`a₍₁₎ ≈ 0.230`, `a₍₅₎ ≈ 0.174`) and that `PR_null ≈ N/3`. If the null profile is not
   flat, the cliff interpretation is unsafe.
5. **Degeneracy (documented negative result)** — reproduce Spearman ρ ≈ 0.071, p ≈ 0.48
   between the `(s₂−s₃)/s₂` gap and `a₍₁₎/a₍₂₎`, and compare the gap distributions of
   dataset A (59/33 split, `s₂/s₃ = 5.27/4.90`) and dataset B (86/10, `4.47/4.14`).
   Report the null result explicitly — it is the first thing a reader will propose, and
   ruling it out is what makes the second partner a finding rather than noise.
6. **Replication** (optional follow-up, not blocking) — other `outBiasTrue` configs
   (N=320, N=64, `initRich`, `ntypes3`) and, in the main tree, `test_ablate_sv`
   `datasplit` 1–4 and `maxrank3`/`maxrank5`. Under rank truncation `K` must be clipped
   to the rank — assert that rather than silently producing garbage. The N=320 / N=64
   configs are especially valuable: chance scales as `1/sqrt(N)`, so the chance-ratio
   should grow with N if the overlap is a fixed geometric property.
7. **Reproducibility** — all nulls draw from an explicitly seeded
   `np.random.default_rng`, so re-running gives identical q-values.
