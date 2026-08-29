# Architecture revision v3: multimodal action teachers

## Status and evidence boundary

This is a post-primary exploratory revision. It does not alter the primary model,
`RESEARCH_PROPOSAL.md`, or any frozen result. K562 test, RPE1, HepG2, and Jurkat
outcomes have already been viewed and cannot be used to choose this revision. Candidate
selection remains restricted to the existing K562 perturbation-OOD validation targets.
A new untouched context is required for any later confirmatory claim.

The design is adapted from ProtJEPA, a 2026 bioRxiv preprint rather than a peer-reviewed
result ([Ravideshik et al.](https://doi.org/10.64898/2026.08.03.742606)). ProtJEPA's
transferable ideas are diverse frozen biological teachers, modality-aware fusion, and
teacher masking. Its protein-specific ten-teacher system and claims are not copied.

## CPU design audit

The frozen Stage-1 cell latents are not severely anisotropic (sample mean cosine
`0.0289`), while the raw ESM action vectors are (`0.8690`). The existing action anchor
already standardizes ESM per dimension. Standardizing latent effect targets changed
validation Pearson by less than `0.00004` and did not materially improve MSE, so target
whitening is not added.

A GO Biological Process teacher covers 92.3% of eligible Replogle targets. A
validation-only ridge probe improved mean latent-effect Pearson from `0.2522` for the
frozen ESM anchor to `0.2917` with ESM plus rank-64 GO, while MSE decreased from
`6.845e-5` to `6.284e-5`. This probe motivates the revision but makes the
result exploratory rather than prospective confirmation.

## Frozen action construction

The new action cache concatenates two independently frozen modalities:

1. the existing 320-dimensional ESM-2 `esm2_t6_8M_UR50D` protein embedding;
2. a rank-64 latent-semantic embedding of propagated human GO Biological Process
   membership.

GO uses the already pinned ontology and human annotation release, excludes `ND` and
`RCA`, propagates `is_a` and `part_of` ancestry, and retains terms covering 3–500 target
genes. SVD signs are canonically oriented. The construction may read all frozen target
names and public annotations but no expression or perturbation outcome.

## Modality-attentive dynamics

Each modality has its own LayerNorm and linear projection to the 256-dimensional action
space. A learned query scores the projected modalities using the ProtJEPA form
`q^T tanh(W_m h_m)`; softmax-normalized weights produce their fused action. The fused
action conditions the unchanged context/action interaction and every transition block.

The mean-effect prior is the existing frozen low-rank ridge construction refit to the
new 384-dimensional action vector using K562 dynamics-training outcomes only. Its rank
and ridge candidates remain fixed. Population residuals remain exactly mean-centered,
so they cannot overwrite the multimodal mean-effect anchor.

Two candidates are frozen before training:

| Candidate | Action-modality dropout | Purpose |
|---|---:|---|
| `attention_full` | 0.00 | direct attentive fusion |
| `attention_dropout_025` | 0.25 | prevent either frozen teacher from dominating |

When dropout removes both modalities for a sample, ESM is retained, preserving the
sequence-first fallback. All other optimizer, loss, split, sampling, and architecture
settings match revision v1.

## Selection and success criteria

The multimodal anchor must improve K562 validation latent-effect Pearson by at least
`0.02` over the frozen ESM anchor and have lower MSE. Otherwise the experiment stops
before GPU training.

The frozen anchor passed this gate on CPU. It selected ridge `1000`, improved Pearson
by `0.0395`, reduced MSE by 8.2%, and reduced mean magnitude absolute error from
`0.0706` to `0.0611`. The action and anchor artifacts are pinned in
`manifests/multiteacher_action_v1.json` and
`manifests/multiteacher_effect_anchor_v1.json`.

If the anchor passes, both attention candidates receive the same CPU replay and bounded
CUDA smoke gates used previously, then full training. Checkpoints are selected by the
unchanged unpaired latent validation loss. A dropout candidate must improve validation
loss by at least `0.005` to displace `attention_full`; otherwise the simpler full-input
candidate is selected. No sealed or external outcome may alter the candidate, weights,
threshold, or hyperparameters.

The final exploratory report must preserve all negative results. Success requires a
Pareto improvement rather than a single favorable metric: better mean-effect transfer
from the multimodal anchor without material degradation of Sinkhorn, MMD, energy,
magnitude, or covariance relative to the relevant frozen references.

## Frozen training result

Both candidates passed exact CPU checkpoint/replay and bounded CUDA checkpoint/resume
smokes before full training on one RTX A6000. Both full runs stopped under the frozen
15-epoch patience rule. `attention_full` reached minimum validation loss `0.690051` at
epoch 98. `attention_dropout_025` reached `0.684799` at epoch 162, an absolute
improvement of `0.005252`. This narrowly exceeds the preregistered `0.005` displacement
margin, so `attention_dropout_025` is selected without consulting any test or external
outcome.

Relative to the prior selected ESM-only anchored model (`0.697403`), the multimodal
selection improves the same validation objective by 1.81%. At the selected epochs,
direction loss improves from `0.900377` to `0.864534` and magnitude loss from
`0.023692` to `0.022066`. The tradeoff is a 2.31% higher Sinkhorn term (`0.241423`
versus `0.235977`) and a 0.38% higher MMD term (`0.035610` versus `0.035475`). These
are modest but real degradations, so training loss alone is not presented as evidence
of uniform superiority.

Exact checkpoints, logs, validation metrics, transfer hashes, and the locked decision
are frozen in `manifests/multiteacher_dynamics_training_v1.json` and
`manifests/multiteacher_dynamics_selection_v1.json`. Reporting-only evaluation on the
already-viewed contexts is the next step and cannot alter this selection.
