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

## Frozen external result

The full test was executed at clean commits `e5f41c8` (latent) and `12a0ed3`
(decoded transcriptomics). It includes 827 HepG2 and 952 Jurkat conditions, six frozen
models, and eight repeats. These outcomes were not used for fitting or selection.

Latent-space results show a real tradeoff rather than a uniformly superior model:

| Context | Model | Effect Pearson | Magnitude error | Sinkhorn | MMD | Energy | Covariance error |
|---|---|---:|---:|---:|---:|---:|---:|
| HepG2 | gated anchor | 0.1456 | 0.1810 | 0.2697 | **0.0216** | **0.0838** | 0.1837 |
| HepG2 | ungated anchor | 0.1456 | 0.1810 | **0.2340** | 0.0316 | 0.1589 | 0.2083 |
| HepG2 | primary CausalCellJEPA | 0.1394 | **0.0555** | 0.2878 | 0.0376 | 0.1272 | **0.1804** |
| Jurkat | gated anchor | 0.1415 | 0.2025 | 0.3187 | **0.0244** | **0.0901** | 0.2225 |
| Jurkat | ungated anchor | 0.1415 | 0.2025 | **0.2493** | 0.0337 | 0.1367 | 0.2231 |
| Jurkat | primary CausalCellJEPA | **0.3010** | **0.0621** | 0.3173 | 0.0275 | 0.0998 | **0.2183** |

The control-OOD gate almost completely falls back to the linear anchor: mean retained
residual confidence is `9.7e-13` in HepG2 and `5.2e-4` in Jurkat. It therefore improves
MMD and energy over the ungated anchor but is numerically tied with linear ESM. The
ungated residual wins Sinkhorn, while the original primary model retains much better
magnitude calibration, the best covariance error, and a large Jurkat direction gain.

The frozen linear decoder gives the same expected gene-effect centroid for the gated,
ungated, and linear-anchor models because their learned population residual is exactly
mean-centered. Their decoded results are therefore numerically identical:

| Context | Model | All-gene Pearson | Target-excluded Pearson | Magnitude error | DE AUPRC | Pathway Pearson |
|---|---|---:|---:|---:|---:|---:|
| HepG2 | anchor / linear ESM | **0.1840** | **0.1847** | 3.9008 | **0.1395** | **0.2496** |
| HepG2 | primary CausalCellJEPA | -0.0110 | -0.0112 | **3.4167** | 0.1079 | -0.0374 |
| Jurkat | anchor / linear ESM | **0.1879** | **0.1891** | 3.3791 | **0.1165** | **0.2788** |
| Jurkat | primary CausalCellJEPA | 0.0411 | 0.0416 | **3.3503** | 0.0924 | 0.0093 |

The anchor's decoded direction gains over the primary model are strongly supported at
the perturbation level: all-gene Pearson improves by 0.1950 in HepG2 (95% bootstrap CI
`[0.1789, 0.2114]`) and 0.1468 in Jurkat (`[0.1347, 0.1589]`). Its HepG2 magnitude error
is worse by 0.4841 (`[-0.5002, -0.4682]` when expressed as improvement). Transcriptomic
retrieval remains weak for every model (top-1 at or below 0.21%).

DE labels use cell-level Welch tests because batch labels are unavailable; 105 HepG2
and 113 Jurkat conditions have no qualifying DE genes. DE results are secondary and do
not have the batch-aware interpretation of the frozen Replogle analysis.

This external result does not establish frontier superiority. It shows that the
transferable linear mean-effect prior and the learned distributional residual solve
different objectives, and that the current gate discards too much residual structure.
Because these outcomes have now been examined, any architecture revision they motivate
is exploratory on HepG2/Jurkat and requires a new untouched context for confirmation.
Exact result hashes are frozen in `manifests/nadig_external_evaluation_v1.json`.
