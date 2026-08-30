# Architecture revision v5b: validation-selected linear SALT student

## Decision inherited from v5

The v5 nonlinear ESM-only student passed its frozen representation gates, but the
mandatory ridge control was materially better on the public-validation split. The
ridge validation reconstruction loss was `0.285043`, compared with `0.368391` for
the nonlinear student. This revision promotes that already-selected control rather
than adding nonlinear capacity after seeing an unfavorable result.

The public test split was opened at the end of v5 and is sealed as reporting-only.
No test metric may change the implementation, validation gates, or downstream GPU
decision in this revision.

## Frozen representation

The masked ESM+GO joint teacher and its train-only standardization are replayed from
the exact v5 checkpoint. A regularized affine student maps standardized frozen ESM-2
features to the standardized 256-dimensional joint-teacher target. The regularization
candidate is selected only by public-validation SmoothL1 reconstruction loss. The
ridge candidates and selected value are inherited exactly from v5; there is no new
search and no refit on validation or test targets.

The exported action concatenates raw frozen 320-dimensional ESM with the
256-dimensional ridge prediction. Retaining raw ESM is the residual safeguard against
the source manuscript's reported loss of sequence-native structural information after
cross-modal distillation. GO features supervise the frozen public teacher but are not
required at inference.

## Promotion boundary

The CPU export must deterministically reproduce the prior validation selection, remain
finite and non-collapsed, and improve validation-query GO-neighbor overlap over raw ESM
by the frozen margin. These checks use only public features and public-validation
targets from the representation-learning task; they never read perturbation outcomes.

Only a passing CPU artifact can enter the existing outcome-restricted anchor and K562
perturbation-OOD validation protocol. It must improve validation loss by at least
`0.003` over v4's `0.681450` before it can replace v4. The older nonlinear, contextual,
and multiteacher architectures remain reproducible baselines regardless of the result.
