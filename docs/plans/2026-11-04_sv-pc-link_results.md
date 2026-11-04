# Results: weight SVs vs activity PCs vs transition labels (outBiasTrue, N=160, 100 sims)

Code: `sv_pc_link.py` (branch `sv-pc-link`). Plan: `2026-11-04_sv-pc-link.md`.
Figures, CSV and the full printout are in
`outBiasTrue/figs/TaskRNNClass_N160_nlatent10_L0_m2_alpha10_nepochs101_ntypes7_fractrain1.0_objCE_initNone_transferrelu_cuesize4_delay0_datasplit0/`
as `svpc_{scatter_sv,scatter_pc,overlap,decoding,time}.svg`, `svpc_axis_summary.csv`, `svpc_run.log`
(same directory as the `recurrent_VU_*.svg` figures).

Setup: last saved epoch (key 100), last timestep, 630 train sequences per sim, 100 sims.
Labels: first / previous / latest = same(0) vs different(1) for letter pairs 1-2, 2-3, 3-4;
the 7 classes are the 7 non-constant combinations. SV = right singular vectors of
`h2h.weight`; PC = **centred** PCA of the activity. Everything is sign-gauge free.
`eta2` = squared correlation of an axis coordinate with a binary label.

## What holds

| claim | result (100 sims) |
|---|---|
| SV1 separates by the **latest** transition | mean eta2 0.76 (median 0.79); SV1 is the best SV for `latest` in 100/100 sims; SV1's best 3-vs-4 class split *is* the latest split in 89/100 |
| **previous** lives on SV2, sometimes SV3 | best SV for `previous` is SV2 in 60%, SV3 in 34%, SV4 in 5%; eta2 0.69 (SV2), 0.57 (SV3), still 0.25-0.31 on SV4-6 |
| the SV2/SV3 identity swap matches the earlier U.V overlap | same 60/34 pattern as the v2/v3 partner counts (59/33) |
| PC1/PC2 do **not** carry the latest transition | PC1 = **first** (eta2 0.92, best split = first in 100/100), PC2 = previous (0.87, 100/100) |

## What does not match the working picture

1. **PC3 is not class-structured here.** Between-class share of variance: PC1 0.93,
   PC2 0.94, **PC3 0.03, PC4 0.04**, PC5/6 about 0.10. The latest transition is not on
   PC3 (eta2 about 0); its best-eta2 PC has median index 9 (beyond PC6 in 88% of sims), i.e.
   it sits on low-variance PCs. So "a third PC separates the 7 clusters" is **not**
   reproduced in this dataset: PC1+PC2 give 4 groups (first x previous), and splitting
   those into the 7 classes needs the latest transition, which is on SV1 (2% of the
   variance) rather than on any of PC1-4. This may reflect a different dataset or PCA
   convention than the one behind the original observation; worth checking which.
2. **The first transition is on no top SV.** For `first`, the winning SV index is beyond SV6
   in 100% of sims (median 84); eta2 is 0.08-0.11 on SV1-6. Its axis is PC1 (37% of the
   variance), which is spread diffusely over the SV spectrum.
3. **SV1 ~ PC1 is false.** |cos(SV1,PC1)| = 0.06 (chance 0.079). SV2-PC2 = 0.51,
   SV3-PC2 = 0.38; PC2's cumulative share in the top 3 SVs is 0.53 (chance 0.02), top 10:
   0.71. PC1's is 0.05 / 0.08 against chance 0.02 / 0.06, and 0.35 in the top 40
   (chance 0.25).

## The left singular vectors tell the copying story

Control with **left** singular vectors: **|cos(u1, PC2)| = 0.93** (mean over sims), while
|cos(v1, PC2)| = 0.05. This is consistent with W reading the latest transition from v1 (SV1) and writing it along
u1, which is the previous-transition axis PC2 and overlaps v2/v3 (an inference from overlaps and the time-resolved check below, not a causal test). It is the mechanism
suggested by the U.V result (u1 overlaps v2 or v3); here it shows up in the activity.

Time-resolved check (mean eta2 of SV_k coordinate at time t with the lag-l transition):

| t | lag0 on SV1 | lag1 on SV2 / SV3 | lag2 on SV3 |
|---|---|---|---|
| 1 | 0.93 | - | - |
| 2 | 0.87 | 0.70 / 0.54 | - |
| 3 | 0.76 | 0.69 / 0.57 | **0.09 (chance-level)** |

The prediction "information steps down the SV index one step per timestep" holds for the
first hop (SV1 -> SV2/SV3) at t=2 and t=3, but **fails at the second hop**: lag2 (the first
transition) is not on SV3 (0.09 vs 0.08-0.11 on every other SV). The "identity rank 1 of
60" at t=3 is carried entirely by lag0 and lag1 and must not be quoted as support for a
full SV1 -> SV2 -> SV3 chain. Caveat: the network is trained on the final step only.

## Energy

Variance fraction along SV1: 0.02; along PC1: 0.37. Cumulative over the top 10: SVs 0.26,
PCs 0.75 (the upper bound), chance 0.06. Span(SV1..r) vs span(PC1..r): mean cos^2 0.23 at
r=3 (chance 0.02), 0.25 at r=10 (chance 0.06). So the weight SVs capture the
class/transition information well (SV1 carries the latest transition), but not the
high-variance directions of the activity.

## Controls and limits

- Centring: the mean activity carries 0.44 of the mean |h|^2, so an uncentred SVD (as in
  `lowrank.py`) would make its PC1 mostly the mean direction; |cos(SV1, mean direction)| = 0.04.
- Left vs right singular vectors give very different overlaps with the PCs (above); the
  right vectors are the "read-in" basis and the one used for the projections elsewhere.
- Single dataset (full-rank, N=160, 100 sims). Rank-constrained and other-N runs were not done.
- With 7 classes each label is one of C(7,3)=35 class splits, so a single-label permutation
  p-value cannot go below 1/35 (the plan's "5040 permutations, floor 1/5040" was wrong).
  The best-split and rank statistics replace it.
- eta2 and the splits use sequence-level variance; ranks are among the 35 splits.
