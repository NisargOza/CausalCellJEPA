# CausalCellJEPA

CausalCellJEPA tests whether a frozen JEPA cell-state space, a biological action
embedding, baseline-population context, and an explicitly unpaired distributional
transition model improve perturbation-by-context OOD prediction. The fixed scientific
specification is [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md).

## Current milestone

The minimum latent-space Replogle experiment, common transcriptomic comparisons,
three-seed Stage 2 robustness evaluation, and validation-preregistered anchored revision
are complete. The leakage-resistant data/split foundation includes:

- 10,000-count normalization followed by `log1p`;
- HVG fitting on an explicit training-visible mask;
- deterministic 70/10/20 target partitions with a stable SHA-256 manifest;
- cell roles for IID, perturbation-OOD, context-OOD, and double-OOD evaluation;
- explicit exclusion of RPE1 perturbed outcomes and held-out targets from fit data.

The official Replogle Figshare article `20029387`, version 1, is downloaded under
git-ignored `data/raw/` and pinned by byte size plus local SHA-256. The published K562
MD5 also matches exactly. Full backed-matrix validation produced
[`manifests/replogle_v1.json`](manifests/replogle_v1.json):

- 558,299 released cells passed schema, author-QC, raw-count, and batch-control checks;
- 997 targets met the 64-cell threshold in both contexts;
- 698/100/199 targets were frozen into train/validation/sealed-test partitions;
- 7,226 genes were shared and 3,000 leakage-safe HVGs were selected from 114,029
  training-visible cells;
- no RPE1 perturbed outcome or held-out target outcome entered HVG fitting.

Raw dataset files remain git-ignored and are not duplicated by preprocessing.

## Repository layout

```text
configs/                 Tunable experiment configuration
manifests/               Frozen dataset, split, and HVG provenance
scripts/                 Download and preprocessing entry points
src/causalcelljepa/      Reusable data and model code
tests/                   Unit and integration tests
data/                    Git-ignored source data and pinned resources
resources/               Versioned derived biological masking collections
artifacts/               Git-ignored checkpoints, logs, and cached latents
```

## Stage 1 implementation

The sparse cell tokenizer and from-scratch Cell-State JEPA are implemented. The locked
encoder uses 512 gene/value tokens, 192-dimensional tokens, 32 Perceiver queries, three
six-head latent blocks, and a 256-dimensional cell state. Its 2.45M encoder parameters
and 0.39M predictor parameters are within the proposal budgets. A real four-cell CPU
smoke batch completed tokenization, student/EMA-teacher inference, normalized Smooth-L1
plus variance/covariance loss, and backward propagation with zero teacher gradients.

Stage 1 currently uses the strict 114,029-cell admission policy: K562 controls and
dynamics-training outcomes plus RPE1 controls. This deliberately excludes every held-out
outcome and all RPE1 perturbed outcomes. The proposal leaves this representation-stage
choice somewhat ambiguous; changing it requires explicit human review.

Biological masking uses the official, CC BY 4.0 Gene Ontology archive release
`2026-08-05` ([archive DOI](https://doi.org/10.5281/zenodo.21844811)). The date-pinned
[`go-basic.obo`](https://release.geneontology.org/2026-08-05/ontology/go-basic.obo) and
[`HUMAN-uniprot.gaf.gz`](https://release.geneontology.org/2026-08-05/annotations/gaf/HUMAN-uniprot.gaf.gz)
sources are checked against fixed byte sizes and SHA-256 digests. The deterministic
derivation excludes negated, ND/RCA, obsolete, and do-not-annotate assertions; propagates
only through `is_a` and `part_of`; intersects with the frozen HVGs; and retains programs
with 3--500 HVGs. This produces 4,328 programs covering 2,742 HVGs. Source metadata,
derivation rules, and the derived GMT hash are recorded in
[`manifests/go_bp_2026-08-05.json`](manifests/go_bp_2026-08-05.json).

The deterministic Stage 1 validation split is frozen in
[`manifests/stage1_v1.json`](manifests/stage1_v1.json): 108,406 admitted training cells
and 5,623 admitted validation cells. Checkpoints contain online/teacher/predictor weights,
optimizer and scheduler state, the exact training cursor, all RNG states, effective
configuration, and source/split/HVG/GMT/code provenance. DataLoader randomness is isolated
from model randomness, and an interrupted real-data CPU run matched an uninterrupted run
tensor-for-tensor after resume. Stage 1 then completed on one NVIDIA RTX A6000 at clean
commit `f1e5dce`: early stopping selected epoch 0 after 20,328 optimizer steps, all 20,334
train/validation records were finite, and the 35.4-minute run peaked at 311,888,896 CUDA
bytes. The downloaded canonical teacher is frozen and matches the best-checkpoint EMA
tensors exactly.

The frozen teacher subsequently embedded exactly 340,684 cells carrying non-excluded roles,
including sealed outcomes only after encoder freezing. The 412 MiB cache completed on an
RTX A6000 in 99.3 seconds; every value was finite, all role/split constraints were re-audited,
and the downloaded SHA-256 matched the remote artifact. `scripts/cache_latents.py` streams
the fixed 256-dimensional states plus cell, context, target, batch, raw-row, and role metadata
into this provenance-rich HDF5 cache and refuses CPU fallback.

## Stage 2 action resources

The primary frozen ESM-2 `esm2_t6_8M_UR50D` action prior is pinned in
[`configs/action.yaml`](configs/action.yaml). `scripts/action_resources.py` keeps acquisition
separate from model execution, verifies the official checkpoint and UniProt release `2026_02`,
and freezes all target mappings in [`manifests/action_v1.json`](manifests/action_v1.json).
Conservative Ensembl/name agreement maps 995 of 997 targets to reviewed canonical proteins.
`ALG1L` has no unique reviewed mapping and `SEM1` is ambiguous, so both remain explicit
training-split unknown actions; no validation or sealed-test target is unresolved.

`scripts/actions.py` mean-pools frozen layer-6 residue representations on CPU. Canonical
proteins longer than the official 1,022-residue extraction cap are processed in non-overlapping
chunks and combined with a residue-count-weighted mean, retaining the full sequence while
bounding attention memory. A nine-protein real-data gate—including the 4,388-residue VPS13D
sequence—was finite and bitwise reproducible. The full 994-unique-protein CPU pass finished in
148 seconds and produced a finite 997-by-320 cache with clean commit provenance. Twelve
stratified targets recomputed within `1.68e-8`; its SHA-256 is
`b1a615f02a69629342f178ed9c5ce81fa1165528c225da8eefc9e59de98d514d`.

## Stage 2 dynamics and latent evaluation

[`configs/dynamics.yaml`](configs/dynamics.yaml) records the locked set sizes, architecture,
loss coefficients, and the engineering-only optimizer/regularization choices. Training-visible
K562 controls and dynamics outcomes alone fit latent normalization, median-distance scaling,
and the near-null direction threshold in
[`manifests/dynamics_v1.json`](manifests/dynamics_v1.json). Validation, sealed-test, and RPE1
perturbed outcomes are excluded from every fitted statistic.

The condition-level sampler draws an independent 32-cell outcome set and a 32-cell control set
with identical gem-group multiplicities; it never constructs control/outcome pairs and never
passes batch or cell-line IDs to the model. The 5.87M-parameter primary model implements a
two-block permutation-invariant context encoder, frozen-ESM action projection, explicit
context/action/product/difference interaction, and three conditioned residual transition
blocks. Its delta head starts exactly at the no-change baseline. The unpaired objective is the
proposal's debiased Sinkhorn divergence plus `0.25` Gaussian MMD, `0.50` direction, and `0.10`
magnitude terms.

All 698 training and 100 K562 perturbation-OOD validation conditions passed batch-matching and
admission audits. CPU and bounded CUDA gates preceded the full RTX A6000 run. Validation-only
selection chose Sinkhorn blur ratio `0.20`; the frozen selected checkpoint is recorded in
[`manifests/dynamics_selection_v1.json`](manifests/dynamics_selection_v1.json). A capacity-
matched pseudo-paired pointwise-MSE comparator was independently smoke-tested, trained, and
frozen in [`manifests/pseudo_paired_v1.json`](manifests/pseudo_paired_v1.json).

The sealed four-regime evaluation uses eight independently sampled populations per target and
treats perturbation-condition as the statistical unit. All 57,408 original records reproduced
exactly after adding the pseudo-paired model, yielding 71,760 five-model records. The full model
strongly improves population calibration and latent distribution distances over pseudo-pairing
in every regime, while pseudo-pairing has higher latent effect-direction correlations in every
regime. Cross-context effect direction remains weak and does not beat the linear ESM baseline.
This metric-dependent, partly negative result is frozen in
[`manifests/evaluation_pseudo_v1.json`](manifests/evaluation_pseudo_v1.json).

Two additional Stage 2 seeds passed exact CPU/CUDA replay and early-stopped independently on
K562-only validation at losses `0.4402` and `0.4380`. Under identical frozen evaluation
populations, all three seeds retain strong K562 effect prediction. The additional seeds also
raise RPE1 effect Pearson correlation to `0.247`--`0.259` in context-OOD and
`0.252`--`0.258` in double-OOD, above both linear ESM and pseudo-paired baselines; the primary
seed remains much weaker at `0.059` and `0.071`, exposing material seed sensitivity. All three
seeds improve RPE1 magnitude and covariance error over linear ESM and improve distribution and
calibration metrics over pseudo-pairing, but all three lose to no-change/linear baselines on
RPE1 Sinkhorn and MMD. The multi-seed result therefore strengthens the effect-direction and
pseudo-pairing conclusions without supporting universal transferred distributional superiority.
Exact artifacts and paired statistics are frozen in
[`manifests/evaluation_stage2_replication_v1.json`](manifests/evaluation_stage2_replication_v1.json).

Passing the two new seeds through the frozen transcriptomic decoder does not rescue the
gene-level claim. Both improve on the primary seed in every regime, but context-OOD effect
Pearson remains `-0.104` and `-0.044`, and double-OOD remains `-0.109` and `-0.049`, versus
positive linear-ESM values of `0.236` and `0.224`. All three JEPA seeds also have negative RPE1
pathway correlations. Target-gene exclusion leaves the result unchanged. The replicated models
do improve magnitude absolute error over linear ESM, demonstrating a direction-calibration
tradeoff rather than total model failure. This decisive gene/pathway result is frozen in
[`manifests/evaluation_stage2_replication_transcriptomics_v1.json`](manifests/evaluation_stage2_replication_transcriptomics_v1.json).

## Transcriptomic readout

[`configs/readout.yaml`](configs/readout.yaml) defines a separate linear decoder from normalized
frozen latents to the same 3,000 normalized log-expression HVGs. The decoder cannot update the
dynamics model. Its fit and ridge-selection cells are restricted to `control_train`,
`control_inference`, and `dynamics_train`; sealed and RPE1 perturbed outcomes are excluded. A
4,096-cell real-data CPU smoke preceded the full CPU pass. The resulting 340,684-by-3,000
expression cache was independently scanned and raw-row audited. Ridge selection on 5,701
permitted validation cells improved MSE from `0.21562` to `0.19813`; the decoder then refit on
all 114,029 permitted cells. Cache/checkpoint hashes and leakage checks are frozen in
[`manifests/readout_v1.json`](manifests/readout_v1.json).

[`configs/transcriptomics.yaml`](configs/transcriptomics.yaml) fixes decoded effect metrics,
target-gene-excluded and retrospective top-DE scopes, batch-level DE calls, perturbation
retrieval, and GO pathway agreement. A four-target K562 validation smoke exercised all five
models and paired statistics without reading sealed or RPE1 perturbed test roles. The full
four-regime transcriptomic evaluation is followed by a diagnostic decoder-ceiling audit that
decodes observed outcome latents; that audit is explicitly not a predictive baseline.

## Mechanism and representation ablations

[`configs/ablations.yaml`](configs/ablations.yaml) fixes the first three required matched Stage 2
ablations: removing the pooled global context, replacing the Set Transformer summary with a
control mean, and removing effect-direction loss. Each preserves the same frozen state/action
caches, training/validation target roles, transition capacity, selected Sinkhorn blur, optimizer,
and schedule. All three passed exact CPU/CUDA replay, full training, and the sealed four-regime
evaluation. The full model is best calibrated after context transfer, while removing global
context improves transferred direction; the result is a direction-calibration tradeoff rather
than uniform support for every proposed mechanism.

The remaining required internal comparisons replace ESM-2 with learned target identity and the
JEPA state with matched 256-D PCA or reconstruction-autoencoder states. All three passed bounded
CPU/CUDA replay and full L40 training before evaluation through a common 3,000-HVG expression
endpoint. PCA uses its frozen inverse projection, the autoencoder uses its frozen reconstruction
decoder, and the learned-ID model shares the leakage-safe JEPA readout. Perturbation-level paired
analysis finds no uniform JEPA-state or ESM-2-action advantage: PCA/autoencoder improve K562
direction and calibration, JEPA is better calibrated after RPE1 transfer but retains poor
transferred direction, and learned IDs match or beat ESM-2 on most decoded endpoints. Removing
the perturbed target gene does not change those conclusions. Exact artifacts and hashes are
frozen in
[`manifests/evaluation_remaining_comparators_v1.json`](manifests/evaluation_remaining_comparators_v1.json).

The required direct gene-space comparison is implemented separately as an ESM-2-to-expression-
effect low-rank ridge model. It fits K562 dynamics-training effects, selects ridge strength only
on K562 perturbation-OOD validation targets, and evaluates the same gene, DE, retrieval, pathway,
and paired metrics without passing through the JEPA state or transcriptomic decoder. Its full
evaluation outperforms the latent world model on effect direction, DE recovery, magnitude error,
and pathway agreement in all four regimes; the negative result is frozen in
[`manifests/direct_gene_v1.json`](manifests/direct_gene_v1.json).

An exploratory post-test action student extends that baseline with the frozen contextual
ESM-2/GO representation and linear/RBF kernels while retaining its frozen rank-64 expression
basis. Fitting uses only K562 dynamics-training effects and selection uses only the designated
K562 perturbation-OOD validation role. Because this variant was developed after primary test
results were inspected, any later test result is exploratory and requires external confirmation.
The frozen validation-only selection chose the GO-with-availability RBF student and reduced MSE
by 15.23% relative to the direct ESM ridge reference while increasing mean effect Pearson from
0.2313 to 0.2700. This is evidence to proceed to exploratory evaluation, not a test-set or
state-of-the-art claim; exact checkpoint and report hashes are in
[`manifests/kernel_gene_selection_v1.json`](manifests/kernel_gene_selection_v1.json).
Its frozen exploratory evaluation is strongest among evaluated models on K562 IID and unseen-
target effect direction, magnitude calibration, DE recovery, and pathway direction, and ranks
first on retrieval in every regime. It does not preserve the direction advantage after RPE1
transfer: double-OOD Pearson is 0.1964 versus 0.2136 for direct ESM, although magnitude error
improves from 3.7318 to 3.4971. The exact mixed result is frozen in
[`manifests/kernel_gene_evaluation_v1.json`](manifests/kernel_gene_evaluation_v1.json).

## Post-primary architecture revision

The completed diagnostics motivate an explicit experimental revision without changing the
locked primary result. [`docs/architecture_revision_v1.md`](docs/architecture_revision_v1.md)
preregisters an anchored, decomposed transition: a frozen rank-64 ESM-to-latent mean-effect
prior; a strictly positive, bounded action-only gain; a bounded action-only nonlinear
correction; and an exactly mean-centered set-transformer residual for population
heterogeneity. This prevents contextual population reshaping from freely reversing the shared
mean action effect while retaining unpaired set-to-set modeling.

The anchor uses 698 K562 dynamics-training targets and selects ridge strength on the 100 K562
perturbation-OOD validation targets. Its deterministic artifact is pinned in
[`manifests/latent_effect_anchor_v1.json`](manifests/latent_effect_anchor_v1.json); no sealed
K562 test or RPE1 perturbed outcome is read. Three fixed correction caps (`0`, `0.25`, `0.50`)
share seed, loss, optimizer, and stopping protocol. Exact CPU resume and two-condition
selection-path smoke gates pass for all three. All three then passed a bounded CUDA gate and
full early-stopped A6000 training, frozen in
[`manifests/anchored_dynamics_training_v1.json`](manifests/anchored_dynamics_training_v1.json).

The preregistered K562-validation-only comparison selected `anchor_only`. Its decoded
all-gene effect Pearson is `0.1969`, an 83.6% relative improvement over the original primary
checkpoint's `0.1072`. The `0.25` correction reaches `0.2005`, but that improvement is below
the fixed `0.01` practical-tie margin and its latent Sinkhorn is worse (`0.2948` versus
`0.2807`), so the simpler candidate wins the frozen tie-break. The revision does not improve
every endpoint: selected decoded magnitude error is worse than the primary and latent effect
direction is substantially lower. Exact metrics, checkpoint hash, condition-level artifact
hashes, and confirmation that no sealed K562 or RPE1 perturbed outcome was read are frozen in
[`manifests/anchored_dynamics_selection_v1.json`](manifests/anchored_dynamics_selection_v1.json).

The subsequent sealed four-regime evaluation uses eight population repeats per target and
does not alter that selection. The anchored model raises decoded all-gene Pearson over the
primary from `0.081` to `0.161` (IID), `0.092` to `0.203` (perturbation-OOD), `-0.130` to
`0.236` (context-OOD), and `-0.134` to `0.224` (double-OOD). It is tied with the frozen linear
ESM anchor and ranks first on RPE1 all-gene Pearson, target-excluded Pearson, and pathway
correlation among the 11 models with defined correlations. It also beats the direct-gene ESM
baseline on RPE1 Pearson by `0.0088` and `0.0108` with target-level bootstrap intervals above
zero. This is not uniform superiority: direct-gene ESM remains stronger on K562 direction,
the anchored model ranks seventh on RPE1 magnitude error, and its centroid metrics are
numerically identical to the linear anchor. The 292 cross-model paired comparisons, frozen
artifact hashes, and conservative interpretation are in
[`manifests/evaluation_anchored_v1.json`](manifests/evaluation_anchored_v1.json).

The later ESM+GO multiteacher revisions are reporting-only extensions of that frozen
primary analysis. Validation selected v4 `availability_static`: the proposed
control-conditioned teacher query did not clear its fixed complexity margin. In the full
four-regime report, v4 ranks second of 13 defined models on K562 perturbation-OOD effect
Pearson (`0.2053`) and second of 14 on DE AUPRC (`0.1340`), but only fourth to fifth on
RPE1 effect endpoints and seventh on RPE1 magnitude error. Relative to v3, decoded effect
changes are at most `0.00031`; magnitude calibration improves slightly, while latent MMD,
Sinkhorn, and energy distance worsen in every regime. This mixed result does not support
uniform superiority or a state-of-the-art claim. The complete 14-model comparison is frozen
in [`manifests/evaluation_contextual_multiteacher_v1.json`](manifests/evaluation_contextual_multiteacher_v1.json)
and interpreted in
[`docs/contextual_multiteacher_evaluation_v1.md`](docs/contextual_multiteacher_evaluation_v1.md).

## Modern baseline feasibility

The proposal asks for GEARS, CPA, CellOT, and State where feasible. Their official interfaces do
not define one interchangeable four-regime task. [GEARS](https://github.com/snap-stanford/GEARS)
supports Replogle and custom splits but explicitly does not support cross-cell-type transfer;
[CellOT](https://github.com/bunnech/cellot) learns condition-specific transport maps rather than
an unseen-action model and reports hours for an example CPU fit;
[CPA](https://github.com/theislab/cpa) supports external embeddings and context transfer but
requires a separate training/tuning stack; and [State](https://github.com/ArcInstitute/state)
supports Replogle plus explicit few-shot/zero-shot split files but requires its own preprocessing
and GPU training pipeline. The locked minimum experiment fails to clearly beat simple,
direct-gene, PCA, and autoencoder baselines, which activates the proposal's rule that broad
external-model scaling is not justified. Those baseline runs remain deferred rather than being
used to search for a more favorable result. A separately labeled, small architecture revision is
now authorized and uses K562 validation only. If external review requires a modern neural
baseline, State is the first
recommended extension, followed by a K562-only GEARS comparison. The auditable decision record is
[`manifests/modern_baseline_feasibility_v1.json`](manifests/modern_baseline_feasibility_v1.json).

The user subsequently authorized the State extension. [`configs/state_baseline.yaml`](configs/state_baseline.yaml)
pins State `0.11.3` and cell-load `0.10.4`; its exporter physically excludes sealed-test
outcomes and every RPE1 perturbed outcome, retains RPE1 controls only as inference inputs, and
provides the same frozen 386-dimensional ESM+GO action features for a fair unseen-target test.
Both single-chunk and streaming multi-chunk exports pass the official cell-load reader. A tiny
real-data CPU run instantiated State-small (57.4M parameters), completed one optimizer step with
finite validation loss, and wrote all expected checkpoints before any GPU allocation. The clean
full export contains 120,156 K562 fit/validation cells and separate K562/RPE1 control templates;
its hashes and leakage audit are frozen in
[`manifests/state_baseline_input_v1.json`](manifests/state_baseline_input_v1.json).

The bounded State-small A6000 run completed all 10,000 steps with finite losses. Its checkpoint
is frozen by minimum K562 perturbation-OOD validation loss (`13.9905`), before any sealed or RPE1
perturbed outcome is used. Reporting-only inference samples the same eight deterministic
32-control populations as the common benchmark, supplies only control expression and the frozen
action vector to State, and stores gene-effect predictions before the four-regime scorer opens
test outcomes. This preserves the proposal's condition-level paired comparison and target-gene-
excluded endpoints without treating control and perturbed cells as paired.

The complete reporting-only evaluation does **not** support the bounded State-small model as a
replacement architecture. All-effect Pearson is `0.0239` IID, `0.0281` perturbation-OOD,
`0.0017` context-OOD, and `-0.0013` double-OOD; magnitude absolute error rises from `5.94` and
`7.60` in K562 to `16.85` and `17.27` in RPE1. State ranks 14th of 14 defined models for both
K562 effect-correlation regimes and 15th of 15 for magnitude error in every regime. Against the
validation-frozen v4 model, 104 paired endpoints are losses, 6 inconclusive, and 2 wins by the
bootstrap-95% rule. This bounded run used 32-cell sets, batch size 4, and 10,000 steps rather than
State-small's much larger official defaults, so it establishes a feasible modern baseline—not a
definitive ceiling on State. Exact hashes, ranks, and leakage provenance are frozen in
[`manifests/state_baseline_v1.json`](manifests/state_baseline_v1.json).

## Development

```bash
uv sync --dev
uv run python scripts/download.py
uv run python scripts/prepare.py
uv run python scripts/resources.py
uv run python scripts/audit_stage1.py
uv run python scripts/smoke_stage1.py
uv run python scripts/smoke_cuda_stage1.py
uv run python scripts/cache_latents.py
uv run python scripts/action_resources.py
uv run python scripts/actions.py
uv run python scripts/prepare_dynamics.py
uv run python scripts/smoke_dynamics.py
uv run python scripts/smoke_cuda_dynamics.py
uv run python scripts/train_dynamics.py
uv run python scripts/smoke_pseudo_paired.py
uv run python scripts/train_pseudo_paired.py
uv run python scripts/evaluate_dynamics.py
uv run python scripts/analyze_evaluation.py
uv run python scripts/smoke_readout.py
uv run python scripts/cache_expression.py
uv run python scripts/train_readout.py
uv run python scripts/smoke_transcriptomics.py
uv run python scripts/evaluate_transcriptomics.py
uv run python scripts/audit_readout.py
uv run python scripts/smoke_ablations.py
uv run python scripts/smoke_cuda_ablations.py
uv run python scripts/train_ablations.py
uv run python scripts/smoke_remaining_comparators.py
uv run python scripts/evaluate_remaining_comparators.py
uv run python scripts/analyze_remaining_comparators.py
uv run python scripts/smoke_stage2_replication.py
uv run python scripts/smoke_cuda_stage2_replication.py
uv run python scripts/train_stage2_replication.py
uv run python scripts/smoke_evaluation_stage2_replication.py
uv run python scripts/evaluate_stage2_replication.py
uv run python scripts/analyze_stage2_replication.py
uv run python scripts/smoke_stage2_replication_transcriptomics.py
uv run python scripts/evaluate_stage2_replication_transcriptomics.py
uv run python scripts/analyze_stage2_replication_transcriptomics.py
uv run python scripts/smoke_direct_gene.py
uv run python scripts/evaluate_direct_gene.py
uv run python scripts/fit_kernel_gene_student.py
uv run python scripts/evaluate_kernel_gene.py
uv run python scripts/prepare_string_action.py
uv run python scripts/fit_string_kernel_gene.py
uv run python scripts/prepare_effect_anchor.py
uv run python scripts/smoke_anchored_dynamics.py
uv run python scripts/smoke_anchored_validation.py
uv run python scripts/smoke_cuda_anchored_dynamics.py
uv run python scripts/train_anchored_dynamics.py
uv run python scripts/select_anchored_dynamics.py
uv run python scripts/smoke_anchored_evaluation.py
uv run python scripts/evaluate_anchored.py
uv run python scripts/analyze_anchored_evaluation.py
uv run python scripts/smoke_state_baseline.py
uv run python scripts/prepare_state_baseline.py
uv run python scripts/prepare_state_prediction_metadata.py
# The next command runs inside the pinned official State CUDA environment.
PYTHONPATH=src work/state/.venv/bin/python scripts/predict_state_baseline.py
uv run python scripts/evaluate_state_baseline.py
uv run ruff check .
uv run pytest
```

Stage 1 pretraining, cell-latent caching, Stage 2 dynamics, and major neural baselines require
CUDA. Action embedding, transcriptomic cache construction, linear readout fitting, and most
metric calculations intentionally remain on CPU.

Full pretraining refuses to fall back to CPU. To resume without modifying the frozen
configuration, set `CAUSALCELLJEPA_RESUME_FROM=artifacts/stage1/latest.pt`. CUDA runs set the
deterministic cuBLAS workspace and abort immediately on non-finite losses or gradients.
