# Architecture revision v2: control-OOD residual gate

This is a **post-hoc exploratory** revision proposed after the sealed v1 results were
inspected. It cannot replace the confirmatory v1 evaluation or support a new
confirmatory claim on the same held-out outcomes.

The v1 anchored model transfers perturbation-effect direction well, but its learned,
exactly centered population residual has worse RPE1 Sinkhorn and MMD than the frozen
linear anchor. A control-only diagnostic found complete separation between sampled
K562 and RPE1 control-population means (exploratory AUC 1.00). The closest RPE1 score
was still roughly eight times the K562 99th-percentile threshold.

The fixed intervention computes the mean diagonal-standardized squared distance of
the input control-population mean from 3,000 sampled K562 control-population means.
The 99th percentile of scores on a separate 3,000 K562 control populations is the
retention threshold; the distance between its 95th and 99th percentiles is the fixed
decay temperature. Residual confidence is one below threshold and decays
exponentially above it. Only the exactly mean-centered learned population residual is
scaled. The frozen linear-anchor mean, learned mean correction, and control state are
unchanged.

Calibration reads only `control_train` K562 latent states. It does not read target
identities, perturbed outcomes, sealed test outcomes, or RPE1 cells. At inference the
gate uses the observed control population, never a cell-line label. Formula and
threshold are frozen before outcome evaluation; results will not be used to retune
them. Because the v1 outcomes motivated this revision, all v2 results remain
exploratory even with this leakage boundary.

Success is limited to latent distribution transfer: RPE1 Sinkhorn, MMD, energy
distance, and covariance-shift error should improve toward the frozen linear-anchor
baseline while K562 degradation remains small. Gene-centroid and decoded mean-effect
metrics are mathematically invariant, so they will not be rerun or claimed as gains.

This intervention is informed by unlabeled-target correlation alignment in
[Deep CORAL](https://arxiv.org/abs/1607.01719) and the unpaired population-transfer
framing of [CellOT](https://www.nature.com/articles/s41592-023-01969-x). It is not an
implementation of either method.

## Frozen exploratory results

The complete four-regime, eight-repeat CPU evaluation was run from clean commit
`05ae778`. K562 residual confidence averaged 0.9956 (IID) and 0.9980
(perturbation OOD); RPE1 confidence was effectively zero in every population. The
gate therefore preserved the learned v1 residual on source controls and made the
predeclared linear-anchor fallback on shifted controls.

Against ungated v1, the RPE1 context/double-OOD metrics changed as follows:

| Metric (lower is better) | Context OOD | Double OOD |
| --- | ---: | ---: |
| Sinkhorn | 0.147121 → 0.125024 | 0.146503 → 0.122700 |
| MMD | 0.023352 → 0.016732 | 0.023086 → 0.016472 |
| Energy distance | 0.141909 → 0.071765 | 0.141144 → 0.070967 |
| Covariance-shift error | 0.218772 → 0.162538 | 0.219849 → 0.159686 |

All eight paired bootstrap intervals exclude zero in the beneficial direction after
multiplicity correction. On K562, Sinkhorn worsened by only 0.000288 (IID) and
0.000146 (perturbation OOD), while MMD and energy distance improved slightly. Mean
effect Pearson, centroid error, and magnitude ratio remained invariant to numerical
precision as required.

The gated model beats the primary dynamics model on RPE1 Sinkhorn, MMD, energy
distance, centroid error, and effect Pearson. It remains slightly worse on covariance
error (0.162538 versus 0.160446; 0.159686 versus 0.157776). Because RPE1 confidence
collapses to zero, the gated model is numerically tied with the linear ESM anchor on
RPE1; this validates the fallback but does not demonstrate transferred learned
heterogeneity. Results remain post-hoc and require a new untouched context or dataset
for confirmation.
