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
