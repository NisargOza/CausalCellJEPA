# CausalCellJEPA

CausalCellJEPA tests whether a frozen JEPA cell-state space, a biological action
embedding, baseline-population context, and an explicitly unpaired distributional
transition model improve perturbation-by-context OOD prediction. The fixed scientific
specification is [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md).

## Current milestone

The project currently implements the data/split foundation that every later experiment
depends on:

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
tensor-for-tensor after resume. The canonical EMA teacher cannot be exported until a full
configured run completes; early stopping selects the best admitted-cell validation epoch.

CPU validation completed 100 real-data optimizer steps at batch size 32 with finite losses;
the mean engineering-smoke loss changed from 0.0493 over the first 20 steps to 0.0243 over
the last 20. This is a numerical stability check, not a biological performance result. The
observed throughput was about 106 cells/second, projecting roughly nine CPU hours for the
configured 30-epoch fit including validation. Full Stage 1 pretraining has not been run.

## Development

```bash
uv sync --dev
uv run python scripts/download.py
uv run python scripts/prepare.py
uv run python scripts/resources.py
uv run python scripts/audit_stage1.py
uv run python scripts/smoke_stage1.py
uv run python scripts/smoke_cuda_stage1.py
uv run ruff check .
uv run pytest
```

After committing a clean code/provenance checkpoint, run full Stage 1 pretraining with
`uv run python scripts/pretrain.py`. A CUDA GPU is required for that run. No biological
experiment, full model training, or matched-baseline comparison has been run yet.

Full pretraining refuses to fall back to CPU. To resume without modifying the frozen
configuration, set `CAUSALCELLJEPA_RESUME_FROM=artifacts/stage1/latest.pt`. CUDA runs set the
deterministic cuBLAS workspace and abort immediately on non-finite losses or gradients.
