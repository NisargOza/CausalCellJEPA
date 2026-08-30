# Contextual multiteacher reporting evaluation v1

## Evidence boundary

This is a reporting-only evaluation of the validation-frozen v4
`availability_static` checkpoint. Its weights and selection were fixed using K562
training and perturbation-OOD validation outcomes before this run. K562 test and RPE1
perturbed outcomes cannot alter that decision. Results use eight population resampling
repeats and perturbation-condition as the statistical unit. A new untouched context is
still required for confirmation.

The control-conditioned `context_query` candidate was rejected during frozen validation
selection and is not substituted after seeing these results. The selected v4 model adds
explicit teacher-availability features to the v3 ESM+GO model but does not use the proposed
control-state query.

## Four-regime result

The table reports decoded all-gene effect Pearson, DE AUPRC, magnitude absolute error,
and pathway NES Pearson. Rankings compare v4 with 13 previously frozen internal models;
no-change has undefined correlation, so correlation denominators contain 13 models.

| Regime | Effect Pearson (rank) | DE AUPRC (rank) | Magnitude error (rank) | Pathway Pearson (rank) |
|---|---:|---:|---:|---:|
| K562 IID | 0.17674 (3/13) | 0.05522 (6/14) | 4.24739 (9/14) | 0.21348 (5/13) |
| K562 perturbation OOD | 0.20529 (2/13) | 0.13395 (2/14) | 2.53820 (9/14) | 0.25354 (5/13) |
| RPE1 context OOD | 0.23171 (5/13) | 0.21319 (5/14) | 4.67009 (7/14) | 0.26926 (6/13) |
| RPE1 double OOD | 0.21677 (4/13) | 0.18269 (5/14) | 4.23697 (7/14) | 0.26847 (4/13) |

Target-gene exclusion leaves the effect conclusion unchanged: Pearson is `0.17718`,
`0.20611`, `0.23212`, and `0.21854`, with ranks 3rd, 2nd, 5th, and 4th. v4 is therefore
competitive on held-out K562 perturbations but is not the best model across regimes or
endpoints.

## Paired comparison with v3

Decoded effect Pearson changes by only `+0.00004` for IID, `+0.00031` for
perturbation OOD, `-0.00006` for context OOD, and `+0.00009` for double OOD. These
are practically negligible; no regime has both a bootstrap interval excluding zero and
a false-discovery-adjusted Wilcoxon result below `0.05`.

Magnitude error improves by approximately `0.0017`--`0.0018` in all four regimes,
with bootstrap intervals above zero. The latent population evidence moves in the opposite
direction: MMD, Sinkhorn, and energy distance are worse in every regime, while latent
effect Pearson is inconclusive in every regime. The headline comparison contains 13 wins,
23 inconclusive results, and 12 losses; all 152 paired endpoints contain 33 wins, 101
inconclusive results, and 18 losses.

## Interpretation

The `0.003349` K562 validation-loss improvement that selected v4 does not translate into
a broad sealed-test gain over v3. Teacher-availability features yield very small decoded
changes and slightly better magnitude calibration, but consistently worse latent
distribution distances. The proper conclusion is a mixed, near-tied revision—not
superiority or state of the art.

The result also closes the proposed control-conditioned fusion direction under this
protocol: its validation gain was below the preregistered complexity margin, and the
selected static availability control did not improve transfer materially. The next useful
comparison is an independently implemented modern model under the same frozen split,
rather than further tuning v4 against already viewed outcomes.

## Audit trail

The exact checkpoint selection is frozen in
`manifests/contextual_multiteacher_dynamics_selection_v1.json`. The 30,498 condition and
pathway records, 152 v4-v3 paired comparisons, 14-model rankings, runtime provenance,
artifact hashes, and negative results are pinned in
`manifests/evaluation_contextual_multiteacher_v1.json`.
