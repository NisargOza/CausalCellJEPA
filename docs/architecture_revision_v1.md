# Architecture revision v1: anchored, decomposed population dynamics

## Status and scope

This document preregisters an experimental revision after completion of the locked
CausalCellJEPA protocol. It does not replace, relabel, or modify the primary model or
its negative results. `RESEARCH_PROPOSAL.md`, `REVIEW.md`, and `REVIEW_LOG.md` remain
unchanged.

The revision is justified by a specific completed diagnostic:

- the full model improves latent perturbation-effect direction and magnitude in the
  two RPE1 regimes, but its frozen linear decoder produces negative mean gene-effect
  and pathway correlations;
- a low-rank ESM-to-effect linear model transfers with positive decoded direction;
- removing global context improves transferred direction but worsens distribution and
  magnitude calibration.

These results indicate that the primary model entangles the transferable mean action
effect with context-conditioned population reshaping.

## Literature-grounded design

The revision combines three established lessons rather than claiming a new primitive:

1. Ahlmann-Eltze, Huber, and Anders show that low-rank linear perturbation models are
   unusually strong and should be treated as an inductive prior, not merely a weak
   baseline ([Nature Methods, 2025](https://doi.org/10.1038/s41592-025-02772-6)).
2. State predicts perturbation responses over sets of cells and includes a low-rank,
   cell-preserving linear displacement baseline; its official implementation also uses
   residual prediction over basal state
   ([bioRxiv, 2025](https://doi.org/10.1101/2025.06.26.661135),
   [official repository](https://github.com/ArcInstitute/state)).
3. GEARS demonstrates the value of an explicit perturbation representation and
   transcriptome-wide cross-gene interaction, while also noting that its predictions
   require the same cellular context
   ([Nature Biotechnology, 2024](https://doi.org/10.1038/s41587-023-01905-6)).
4. STAMP separates perturbation prediction into affected-gene identification,
   direction, and magnitude, supporting explicit calibration rather than asking one
   unconstrained residual to solve all three subtasks
   ([Nature Computational Science, 2024](https://doi.org/10.1038/s43588-024-00698-1)).

## Revised transition

For control population `Z_c`, action embedding `a_g`, and the frozen training-only
low-rank action anchor `b(a_g)`, predict

```
Z_hat = Z_c + s(a_g) b(a_g) + r_rho(a_g) + H(Z_c, a_g).
```

The components have deliberately separated responsibilities:

- `b(a_g)` is a frozen rank-64 ridge map from ESM-2 to the K562 training-target latent
  effect. Ridge strength is selected on the frozen K562 perturbation-OOD validation
  targets.
- `s(a_g)` is a strictly positive action-only gain. For corrected candidates it is
  bounded to `[1/4, 4]`, allowing magnitude calibration without reversing the anchor.
- `r_rho(a_g)` is an action-only nonlinear correction. Its norm is hard-bounded by
  `rho * max(||s(a_g)b(a_g)||, null_effect_threshold)`, so it cannot freely overwrite
  the transferable linear prior.
- `H(Z_c, a_g)` is produced by the existing set transformer, then centered exactly
  across cells. It can model context-specific heterogeneity and covariance but cannot
  alter the predicted population mean.

No cell-line ID, target identity, pseudo-pairing, sealed-test outcome, or RPE1 outcome
enters fitting or selection.

## Frozen candidate set

All candidates use seed `20260826`, the original K562 target split, the same optimizer,
the same unpaired loss, and the same early-stopping rule.

| Candidate | Gain range | Mean-correction cap | Purpose |
|---|---:|---:|---|
| `anchor_only` | exactly 1 | 0.00 | Strong linear mean effect plus learned population reshaping |
| `anchor_residual_025` | `[1/4, 4]` | 0.25 | Direction-preserving gain plus conservative correction |
| `anchor_residual_050` | `[1/4, 4]` | 0.50 | More expressive but still bounded correction |

The original primary model and the low-rank linear anchor remain fixed references; they
are not retrained as candidates.

## Validation-only selection rule

Selection is performed once using only K562 `perturbation_ood_validation` outcomes.

1. Each run selects its checkpoint by the original latent validation loss.
2. A candidate is eligible only if all values are finite, its centered heterogeneity
   mean is numerically zero, and its mean correction respects the configured norm cap.
3. The primary architecture-selection metric is mean all-gene decoded perturbation-
   effect Pearson correlation against observed K562 validation expression effects.
4. A corrected candidate must not worsen latent Sinkhorn distance or decoded magnitude
   absolute error by more than 5% relative to `anchor_only`.
5. Among eligible corrected candidates, select the highest decoded effect correlation.
   Improvements below 0.01 are treated as a tie and resolved by lower latent Sinkhorn,
   then the smaller correction cap.
6. If neither corrected candidate passes the guardrails, freeze `anchor_only`.

Only after the selected name, checkpoint hash, and this decision record are frozen may
sealed K562 test or RPE1 outcomes be evaluated. A weak result remains a valid negative
result; no candidate will be chosen or redesigned from sealed outcomes.

## Frozen validation result

All three candidates passed exact CPU checkpoint/resume replay, a bounded CUDA
checkpoint/resume and finite-validation gate, and full early-stopped training on one
NVIDIA RTX A6000. The training artifacts and checks are pinned in
`manifests/anchored_dynamics_training_v1.json`.

| Candidate | Best epoch | Latent validation loss | Decoded effect Pearson | Latent Sinkhorn | Decoded magnitude absolute error |
|---|---:|---:|---:|---:|---:|
| `anchor_only` | 139 | 0.697403 | 0.196934 | 0.280728 | 2.556439 |
| `anchor_residual_025` | 40 | **0.687205** | **0.200451** | 0.294762 | **2.399769** |
| `anchor_residual_050` | 40 | 0.687624 | 0.197097 | 0.293830 | 2.578133 |
| Original primary reference | 28 | 0.568212 | 0.107241 | 0.371557 | 2.439233 |

Every candidate passed the preregistered eligibility and corrected-candidate guardrails.
Their decoded Pearson values fall within the fixed 0.01 practical-tie margin, so the
tie-break selects `anchor_only`: it has lower latent Sinkhorn than either corrected
candidate and the smallest correction cap. Its decoded validation Pearson is 83.6%
higher than the original primary reference, but it is worse on decoded magnitude error
and substantially worse on latent effect direction. The revision therefore improves the
chosen decoded-direction endpoint without establishing uniform superiority.

The immutable decision, selected checkpoint SHA-256, condition-level artifact hashes,
and explicit leakage report are in `manifests/anchored_dynamics_selection_v1.json`.
Selection read exactly 100 K562 perturbation-OOD validation targets over eight fixed
repeats. It did not read sealed K562 test or RPE1 perturbed outcomes.
