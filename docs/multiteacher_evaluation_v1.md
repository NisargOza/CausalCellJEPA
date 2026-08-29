# Multiteacher reporting evaluation v1

## Evidence boundary

This is a reporting-only evaluation of the validation-frozen
`attention_dropout_025` checkpoint. K562 test, RPE1, HepG2, and Jurkat outcomes
were already viewed before this revision and did not alter its weights, selection,
threshold, or hyperparameters. Results are condition-level across eight population
resampling repeats. A new untouched context is still required for a confirmatory claim.

The revision transfers diverse frozen teachers and modality-attentive fusion from the
ProtJEPA design, but its protein-function benchmark values are not directly comparable
to perturbational single-cell prediction. This report therefore makes no frontier or
state-of-the-art claim from cross-paper numbers.

## Frozen training decision

The ESM+GO effect anchor passed its CPU gate with K562 validation Pearson `0.29165`,
compared with `0.25217` for ESM alone. On GPU, full attention reached validation loss
`0.690051`; 25% teacher dropout reached `0.684799`. The absolute improvement
`0.005252` cleared the preregistered `0.005` displacement threshold, selecting
`attention_dropout_025` without test-outcome reuse.

## Internal benchmark position

The table reports decoded all-gene effect Pearson and its rank across every comparable
frozen internal reference. Rank denominators exclude the undefined no-change
correlation. DE AUPRC includes all 13 models.

| Regime | Effect Pearson | Pearson rank | DE AUPRC | DE rank | Pathway NES Pearson rank |
|---|---:|---:|---:|---:|---:|
| K562 IID | 0.17670 | 3 / 12 | 0.05527 | 5 / 13 | 5 / 12 |
| K562 perturbation OOD | 0.20497 | 2 / 12 | 0.13376 | 2 / 13 | 5 / 12 |
| RPE1 context OOD | 0.23177 | 4 / 12 | 0.21324 | 4 / 13 | 5 / 12 |
| RPE1 double OOD | 0.21669 | 4 / 12 | 0.18273 | 4 / 13 | 4 / 12 |

The multimodal model is competitive on effect direction and DE recovery, especially
for held-out K562 perturbations. It is not uniformly best. Direct-gene ESM leads the
K562 perturbation-OOD gene metrics; the prior anchored/linear-effect family remains
stronger on RPE1 correlations; state-space alternatives lead IID metrics. Magnitude
error ranks only ninth of 13 in both K562 regimes and seventh of 13 in both RPE1
regimes.

## Paired comparison with the prior anchored model

Decoded IID performance improves clearly: all-gene effect Pearson increases by
`0.01554`, DE AUPRC by `0.00572`, and pathway NES Pearson by `0.01304`; each survives
false-discovery correction. For held-out K562 perturbations, effect Pearson increases
only `0.00194`. Its paired Wilcoxon result is just below `q=0.05`, while its bootstrap
interval crosses zero, so it is treated as fragile rather than a decisive win.

RPE1 decoded effect Pearson decreases by `0.00473` in context OOD and `0.00766` in
double OOD; neither survives multiple-testing correction. The latent population view
is more cautionary: RPE1 effect correlation trends lower and MMD/energy distance are
significantly worse, despite consistently better magnitude calibration. K562
perturbation-OOD latent effect Pearson is a tie.

Retrieval remains weak. Top-1 retrieval is `0.43%` for IID, `1.01%` for K562
perturbation OOD, `0.14%` for context OOD, and `0.50%` for double OOD. The multimodal
revision improves reciprocal rank and top-5 retrieval over the prior anchor, but not
enough to claim target-level identification quality.

## Attention audit and architectural implication

The learned fusion is nearly balanced for 929 GO-annotated targets: 50.8% ESM and
49.2% GO on average. For the 68 targets without retained GO terms, ESM rises only to
54.8%. This modest fallback shows that dropout prevented complete teacher dominance,
but the global action-only query did not learn strong reliability specialization.

The evidence points to context transfer—not raw action direction—as the next
bottleneck. A justified follow-up is control-population-conditioned teacher attention:
the baseline cell state queries frozen teacher tokens, with an explicit teacher
availability mask and an ESM fallback. This remains identity-free and outcome-safe,
while allowing the same perturbation representation to be calibrated differently in
an unseen cellular state. Any such candidate must be selected on permissible
validation data and tested confirmatorily in a new untouched context.

## Audit trail

Exact checkpoints and training logs are frozen in
`manifests/multiteacher_dynamics_training_v1.json`; the locked selection is in
`manifests/multiteacher_dynamics_selection_v1.json`. All evaluation artifacts,
13-model rankings, 320 paired comparisons, attention diagnostics, hashes, runtime
provenance, and negative results are pinned in
`manifests/evaluation_multiteacher_v1.json`.
