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
tensor-for-tensor after resume. Stage 1 then completed on one NVIDIA RTX A6000 at clean
commit `f1e5dce`: early stopping selected epoch 0 after 20,328 optimizer steps, all 20,334
train/validation records were finite, and the 35.4-minute run peaked at 311,888,896 CUDA
bytes. The downloaded canonical teacher is frozen and matches the best-checkpoint EMA
tensors exactly. These are engineering/training facts; downstream biological performance
has not yet been evaluated.

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
sequence—was finite and bitwise reproducible. The 320-dimensional cache is ready to be built
from a clean commit before the learned 256-dimensional action projection is implemented.

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
uv run ruff check .
uv run pytest
```

Stage 1 pretraining, cell-latent caching, Stage 2 dynamics, and major neural baselines require
CUDA; action embedding is intentionally kept on CPU. No Stage 2 dynamics experiment,
biological evaluation, or matched-baseline comparison has been run yet.

Full pretraining refuses to fall back to CPU. To resume without modifying the frozen
configuration, set `CAUSALCELLJEPA_RESUME_FROM=artifacts/stage1/latest.pt`. CUDA runs set the
deterministic cuBLAS workspace and abort immediately on non-finite losses or gradients.
