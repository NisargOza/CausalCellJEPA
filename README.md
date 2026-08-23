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
artifacts/               Git-ignored checkpoints, logs, and cached latents
```

## Stage 1 status

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

## Development

```bash
uv sync --dev
uv run python scripts/download.py
uv run python scripts/prepare.py
uv run ruff check .
uv run pytest
```

No biological experiment or full model training has been run yet. The next implementation
unit is pinning the open GO Biological Process masking resource, followed by the Stage 1
training/checkpoint loop and throughput probe.
