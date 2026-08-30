# Architecture revision v4: control-conditioned biological teachers

## Status and evidence boundary

This is a post-primary exploratory revision motivated by the frozen v3 result. It does
not alter `RESEARCH_PROPOSAL.md`, the primary model, or any prior result. K562 test,
RPE1, HepG2, and Jurkat outcomes have already been viewed and cannot select this
revision. Fitting and selection remain limited to K562 `dynamics_train` and
`perturbation_ood_validation`, respectively. A new untouched context is required for
confirmation.

## Evidence motivating the change

Revision v3 improved K562 IID and held-out-perturbation rankings, but regressed RPE1
latent distribution transfer. Its learned action-only attention was nearly uniform:
50.8% ESM and 49.2% GO for annotated targets, with only a modest ESM increase to 54.8%
when GO was unavailable. This suggests that the static teacher query cannot determine
which biological prior is reliable in a new basal cellular state.

The design follows convergent primary research rather than copying one system:

- [CisTransCell](https://arxiv.org/abs/2606.13713) integrates coding and regulatory
  sequence priors with dynamic control-cell state and uses state-dependent modulation.
- [TxPert](https://doi.org/10.1038/s41587-026-03113-4) separates a basal-state encoder
  from multi-graph perturbation encoders and reports that control matching/aggregation
  improves transfer.
- [State](https://doi.org/10.1101/2025.06.26.661135) uses control-cell sets and
  population attention for cross-context perturbation prediction.
- The [2026 Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026)
  formalizes zero-shot context transfer using only unperturbed cells from unseen cell
  lines plus perturbation target identities.

## Frozen action representation

The biological features remain exactly the v3 320-dimensional ESM-2 and rank-64 GO
teacher vectors. Two public-data availability bits are appended: unique ESM mapping
present and retained GO annotation present. Overall action identity is known if either
teacher is available, allowing GO to rescue a target without a unique ESM mapping.
No expression or perturbation outcome enters this cache.

The projectors still receive only their own biological feature block. Availability
bits are used for hard attention masks and as auditable ridge-anchor covariates; an
unavailable teacher cannot receive nonzero attention. If neither teacher is available,
the computation stays finite and the existing unknown-action path is used.

## Control-conditioned fusion

The unchanged set-transformer control encoder produces a 256-dimensional basal-state
summary. A zero-initialized context projection turns that summary into a query update.
For teacher `m`, the score is

`s_m = (q_global + q_context(control))^T tanh(W_m h_m) / sqrt(d)`.

Availability-masked softmax weights fuse the projected teachers. Zero initialization
starts training from the stable v3-style global query while allowing gradients to
learn context-specific teacher reliability. No cell-line identifier, batch label, or
perturbed outcome is supplied to the query.

## Frozen experiment and rejection criteria

Two matched candidates will use the same action cache, effect anchor, 25% modality
dropout, optimizer, loss, split, sampling, and early stopping:

| Candidate | Context query | Purpose |
|---|---:|---|
| `availability_static` | no | isolate availability-mask and action-cache changes |
| `context_query` | yes | test control-state-dependent teacher reliability |

The refit anchor must not reduce K562 validation latent-effect Pearson by more than
`0.002` relative to the v3 multimodal anchor (`0.291651`) and must not increase MSE by
more than 2%. Otherwise the revision stops before GPU training.

Both candidates must pass exact CPU checkpoint/replay and bounded CUDA
checkpoint/resume gates. Full checkpoints use the unchanged original latent validation
loss. `context_query` must improve that loss by at least `0.003` over
`availability_static` to be selected; otherwise the static matched control wins. The
selected v4 candidate must also beat the frozen v3 loss `0.684799` to justify replacing
v3. Test or external outcomes cannot change these decisions.

## Completed GPU result

Both candidates passed the CUDA checkpoint/resume gate and completed on one RTX A6000.
`context_query` reached `0.681129`, a numerical improvement of only `0.000321` over
`availability_static` at `0.681450`; this fails the locked `0.003` complexity margin.
The selected v4 model is therefore `availability_static`. It improves the v3 validation
loss by `0.003349` (0.49%), so it replaces v3 for subsequent exploratory evaluation.

The result rejects the control-conditioned query as a material improvement; it does not
establish state of the art. Selection used no K562 test, RPE1, HepG2, or Jurkat outcomes.
Exact checkpoints, logs, provenance, and the decision are frozen in
`manifests/contextual_multiteacher_dynamics_training_v1.json` and
`manifests/contextual_multiteacher_dynamics_selection_v1.json`.
