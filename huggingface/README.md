---
license: other
library_name: pytorch
pipeline_tag: feature-extraction
tags:
  - biology
  - single-cell
  - perturbation-prediction
  - jepa
  - transcriptomics
  - pytorch
---

# CausalCellJEPA

CausalCellJEPA is a research model for predicting unpaired single-cell perturbation populations.
It combines a frozen JEPA cell-state encoder, biological action features, baseline-population
context, and an action-conditioned set transition. This repository contains compact
`safetensors` exports, exact model metadata, provenance manifests, empirical plots, and the
minimal project code needed to reconstruct the custom PyTorch modules.

The project produced a **mixed result**, not a validated global state-of-the-art result. The
primary model strongly improves population calibration and distribution distances over a
matched pseudo-paired comparator, while the comparator has higher latent effect-direction
correlation. The final Adamson candidate improves over a systematic perturbed-mean baseline but
does not beat its frozen STRING+GO component.

Source code: [NisargOza/CausalCellJEPA](https://github.com/NisargOza/CausalCellJEPA)

## Included components

| Component | Purpose | Scientific status |
|---|---|---|
| `stage1_teacher` | Frozen 256-dimensional EMA cell encoder | Proposal-locked primary |
| `stage2_primary` | Unpaired population dynamics | Proposal-locked primary |
| `transcriptomic_readout` | 256-dimensional latent to 3,000-HVG decoder | Proposal-locked reporting |
| `stage2_multiteacher_v4` | Validation-selected ESM-2 + GO population model | Post-primary exploratory |
| `multiteacher_effect_anchor` | Low-rank biological effect anchor | Post-primary exploratory |
| `external_response_predictor` | Multi-context gene-effect predictor | Exploratory post-test |
| `string_go_predictor` | STRING + GO gene-effect predictor | Exploratory post-test |
| `control_ood_gate` | Control-only reference confidence gate | Frozen before Adamson outcomes |
| `replogle_actions_multiteacher` | Precomputed ESM-2 + GO features for 997 targets | Derived feature cache |
| `replogle_actions_final` | Precomputed ESM-2 + GO + STRING features for 997 targets | Derived feature cache |

Raw single-cell data, optimizer states, temporary checkpoints, State baseline weights, and
third-party ESM-2 weights are intentionally excluded.

## Loading

This is a custom research architecture rather than a Transformers `AutoModel`. Install the
requirements, clone the source project, and load a snapshot:

```bash
pip install -r requirements.txt
git clone https://github.com/NisargOza/CausalCellJEPA.git
```

```python
from huggingface_hub import snapshot_download

repo_dir = snapshot_download("NisargOza/CausalCellJEPA")

# Run from the cloned source repository so causalcelljepa is importable.
from load_components import load_primary_dynamics, load_tensor_component

model, metadata = load_primary_dynamics(repo_dir)
readout, readout_metadata = load_tensor_component(repo_dir, "transcriptomic_readout")
model.eval()
```

`load_components.py` verifies the files listed in `MODEL_MANIFEST.json` before loading them.
The primary dynamics model consumes normalized control populations with shape
`[batch, 32, 256]`, action embeddings with shape `[batch, 320]`, and a boolean action-known
indicator. Exact preprocessing and normalization are recorded in the included configuration and
provenance files.

## Results

### Replogle double OOD

The statistical unit is a perturbation condition after averaging eight deterministic population
resamples. Relative to the matched pseudo-paired model, CausalCellJEPA is better for:

| Metric | CausalCellJEPA | Pseudo-paired | Targets favoring CausalCellJEPA |
|---|---:|---:|---:|
| Magnitude absolute error ↓ | 0.0591 | 0.2274 | 188 / 199 |
| MMD ↓ | 0.0294 | 0.0971 | 198 / 199 |
| Sinkhorn divergence ↓ | 0.1371 | 0.2735 | 199 / 199 |
| Latent effect Pearson ↑ | 0.0715 | 0.1873 | 66 / 199 |

![Replogle target-paired results](assets/figures/empirical/empirical_2_replogle_target_paired_scatter.png)

### Adamson external confirmation

Across 27 preregistered targets, the frozen candidate improves Systema all-gene Pearson over the
perturbed-mean baseline by `+0.2606` (95% target-bootstrap CI `[+0.1563, +0.3700]`). It trails
STRING+GO by `-0.0201` (`[-0.0422, -0.0002]`). The external confirmation therefore failed one of
six preregistered criteria and does not support a global SOTA claim.

![Adamson target-paired results](assets/figures/empirical/empirical_3_adamson_systema_paired_scatter.png)

## Training and evaluation data

- Replogle et al. CRISPRi screens in K562 and RPE1.
- Adamson et al. 2016 Perturb-seq was used for a one-shot external confirmation.
- Exploratory external-response components also used Nadig et al. HepG2 and Jurkat outcomes.
- UniProt, Gene Ontology, STRING-derived features, and ESM-2 representations provide biological
  action information.

The data are not redistributed here. Dataset identifiers, checksums, splits, roles, and leakage
audits are included under `provenance/`.

## Intended use

- Research on single-cell perturbation prediction.
- Reproduction of the reported latent-population and transcriptomic comparisons.
- Feature extraction or controlled evaluation of the released components.

The model is not validated for clinical use, patient-specific treatment selection, diagnostic
decisions, or safety-critical biological intervention design.

## Limitations

- The evidence does not establish global state-of-the-art performance.
- Direction recovery and distributional calibration favor different objectives.
- Adamson outcome-fit-unseen targets remain weak on average.
- The control-gated candidate does not outperform its STRING+GO component.
- The action caches cover the frozen Replogle target vocabulary; new targets require rebuilding
  biological features from the source project.
- Exploratory post-test components must not be interpreted as confirmatory replacements for the
  primary architecture.

## Reproducibility and security

- Tensor weights are stored in `safetensors`; optimizer, scheduler, and RNG states are removed.
- `MODEL_MANIFEST.json` records the SHA-256 of every original frozen artifact and every exported
  file other than the manifest itself.
- Source prediction and evaluation manifests preserve the outcome-use and leakage boundaries.
- The small original effect-anchor checkpoint is retained only because the custom model builder
  verifies its exact frozen hash; load it only with a current PyTorch release and
  `weights_only=True`.

## License

The source repository did not contain a software or model license when this package was created.
Accordingly, this model card uses Hugging Face's `other` license tag and does not invent or imply
an open-source license. Contact the repository owner before reuse or redistribution.

## Citation

This work is currently released as a research artifact. Use the `CITATION.cff` metadata in this
repository and cite the GitHub repository plus the exact model revision used.
