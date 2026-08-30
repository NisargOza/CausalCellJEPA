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

## Completed CPU export

The deterministic replay selected ridge `100.0` with zero validation-loss replay
error. Validation reconstruction loss was `0.285043`, target cosine was `0.640632`,
and stable rank was `8.40274`. The loss improvement over the nonlinear student was
`0.083348`. Validation-query residual GO-neighbor overlap improved by `0.007527` over
raw standardized ESM, so every frozen CPU gate passed.

The reporting-only public test reproduced loss `0.302580` and target cosine `0.611168`.
Its residual GO-neighbor overlap gain was only `0.001075`, below the validation gate.
This weaker held-out geometry is a material limitation and is retained in the manifest,
but it cannot change or tune a revision whose test split was already viewed. The next
authorized selection boundary remains the pre-existing K562 perturbation-OOD validation
protocol; no claim of broad superiority follows from this CPU result.
