# Final Adamson external confirmation

## Status

This is the terminal external confirmation for CausalCellJEPA. The candidate, target cohort,
reference cohort, metrics, success criteria, and stopping rule were committed before the Adamson
GSE90546 expression matrix was opened. The single full evaluation failed one of six required
criteria. Architecture search is closed, and no same-split frontier comparison is authorized.

This result does not support a global state-of-the-art claim. It supports a narrower conclusion:
the frozen response model predicts substantial Adamson perturbation-specific signal beyond the
dataset's systematic perturbed mean, but the preregistered control gate does not preserve enough
of the best frozen STRING+GO component and does not generalize strongly to outcome-fit-unseen
targets.

## Leakage and execution audit

- Study/sample: Adamson et al. 2016, GSE90546/GSM2406681, K562 CRISPRi.
- Scored cohort: 27 targets and 15,021 cells.
- Systema reference cohort: 55 disjoint, action-ineligible targets and 30,178 cells.
- Controls: 5,241 cells across ten barcode lanes.
- Frozen vocabulary overlap: 2,963 of 3,000 HVGs by exact Ensembl ID.
- Prediction role: controls only; no perturbation outcome was used for fitting, selection, or the
  gate.
- Prediction gate: `0.0003460099 * STRING+GO + 0.9996539901 * external response`.
- Outcome evaluation: exactly one full run, at clean commit `5471a95`.
- Statistical unit: perturbation-condition; cells are never treated as paired observations or
  independent benchmark replicates.
- DEG labels: one-sample tests over ten lane-level target-minus-control effects, BH FDR `0.05`,
  minimum absolute effect `0.10`.

The control-only prediction artifact safely replays with PyTorch's restricted loader. Its SHA-256
is `6aa75109e0b96e4d19536e396289a7b352eb23c8e5f6bde909489995cfda622d`.

## Primary result

| Model | Systema Pearson delta | Target-excluded delta | Centroid accuracy | Control-reference Pearson | Magnitude error | DE AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| Frozen gated candidate | 0.2056 | 0.2081 | 0.5256 | 0.4067 | 2.9450 | 0.3298 |
| External response | 0.2056 | 0.2081 | 0.5256 | 0.4067 | 2.9450 | 0.3298 |
| STRING+GO | **0.2257** | **0.2287** | **0.5427** | 0.4305 | **2.9009** | 0.3432 |
| Direct ESM ridge | 0.0067 | 0.0075 | 0.5057 | 0.3557 | 3.4580 | 0.3124 |
| Mean training effect | -0.0276 | -0.0270 | 0.5043 | 0.3172 | 3.5758 | 0.3074 |
| No change | -0.0564 | -0.0546 | 0.5028 | undefined | 4.7053 | 0.1741 |
| Perturbed mean | -0.0550 | -0.0493 | 0.5000 | **0.5509** | 2.9415 | **0.4982** |

The perturbed mean is deliberately strong on conventional control-reference and DE endpoints
because Adamson contains high systematic perturbed-versus-control variation. The Systema metrics
subtract an equal-condition reference centroid estimated only from the 55 reference targets;
they are the primary evidence for perturbation-specific prediction.

The candidate improves over perturbed mean by `0.2606` for all-gene Systema Pearson delta (95%
paired-bootstrap CI `[0.1563, 0.3700]`) and `0.2574` target-excluded (`[0.1529, 0.3650]`). Its
centroid accuracy gain is `0.0256`; the corresponding interval includes zero, but the locked
criterion required only a positive point estimate.

## Generalization strata

| Stratum | Targets | Candidate delta | Target-excluded delta | Centroid accuracy |
|---|---:|---:|---:|---:|
| Outcome-fit seen | 19 | 0.3323 | 0.3360 | 0.6073 |
| Outcome-fit unseen | 8 | -0.0953 | -0.0958 | 0.3317 |

The unseen stratum is much better than its perturbed-mean reference (`-0.3186` and `-0.3196`),
but it remains negative in absolute Systema correlation. The overall positive result is therefore
driven by targets whose Replogle outcomes contributed to response fitting. This limits claims of
global unseen-intervention generalization.

## Locked decision

| Criterion | Result |
|---|---|
| All-gene paired Systema CI lower bound above zero vs perturbed mean | Pass |
| Target-excluded paired Systema CI lower bound above zero vs perturbed mean | Pass |
| Centroid accuracy point estimate above perturbed mean | Pass |
| Mean Systema loss versus best frozen component at most `0.01` | **Fail: `0.02037`** |
| Centroid accuracy loss versus best component at most `0.02` | Pass: `0.01709` |
| Magnitude-error degradation versus best component at most 2% | Pass: about 1.52% |

The external confirmation therefore fails. Selecting STRING+GO after seeing Adamson would be a
post-test architecture change, so it is reported as the best frozen component rather than adopted
as a revised candidate. The preregistered consequence is finalization of a mixed result: no gate
retuning, no second Adamson evaluation, no extra benchmark cycle, and no frontier/SOTA claim.

## Reproducibility anchors

- Protocol: `configs/adamson_external_confirmation.yaml`.
- Runtime: `configs/adamson_external_evaluation.yaml`.
- Preparation manifest self-hash: `0564f07569a899fe933fcc50780743403f30744af7474d4f4058bc85e57ddf51`.
- Prediction manifest self-hash: `584a30cb059f15d44e0000d6c78b0bdf396bff4c0811ed64a86fe8d46480696b`.
- Evaluation manifest self-hash: `aacf3efae7d7dffe41baf01c466f6cda2fd2a7b89b6ca467727a331fc97ebbd0`.
- Systema framework: [Viñas Torné et al., Nature Biotechnology 2025](https://www.nature.com/articles/s41587-025-02777-8), repository commit `aaf5b5353993b48b78543f2f93b3e18ca65df515`.
