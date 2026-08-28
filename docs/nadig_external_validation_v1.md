# Preregistered Nadig external validation

This confirmation tests the already frozen CausalCellJEPA models on the HepG2 and
Jurkat CRISPRi Perturb-seq screens from Nadig et al. The two cellular contexts, their
controls, and every perturbed expression outcome were untouched during all prior
training, selection, architecture revision, and residual-gate calibration.

The protocol is frozen before either H5AD expression matrix is downloaded or opened.
Both datasets are test-only. Eligible targets must already have a frozen action
embedding and at least 32 measured cells. No target, context, metric, or result may be
used to tune the JEPA encoder, dynamics weights, ESM anchor, residual gate, decoder,
thresholds, or sampling. Results must be reported for both contexts even if negative.

Eight deterministic repeats independently sample 32 outcome cells and 32 controls per
perturbation. The trimmed scPertEval files do not retain experimental batch labels, so
batch matching is unavailable and this limitation must remain attached to every
claim. The perturbation-condition is the statistical unit. Latent population metrics
and decoded transcriptomic metrics mirror the frozen Replogle evaluation, including
target-gene-excluded effect correlation and mandatory simple baselines.

The primary question is whether the frozen control-OOD fallback preserves the v1 mean
effect while improving population-distance robustness in two genuinely new cell
contexts. These results can externally support the post-hoc gate mechanism, but they
cannot retroactively make its original RPE1 analysis confirmatory because the datasets
differ in experiment, preprocessing, and available batch metadata.

Sources: the peer-reviewed [Nadig et al. study](https://doi.org/10.1038/s41588-025-02169-3)
and the public [scPertEval processed-dataset documentation](https://github.com/Virtual-Cell-Research-Community/scPertEval/blob/main/docs/user-guide/datasets.md).
