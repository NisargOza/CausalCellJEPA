# Architecture revision v5: SALT-style residual action student

## Status and evidence boundary

This post-primary exploratory revision changes only the perturbation representation.
It does not alter the frozen JEPA cell-state encoder, population objective, target
split, or primary proposal. The student receives frozen ESM-2 features and public GO
features supervise only a frozen teacher. No perturbation outcome or cell-line label
enters either action-pretraining phase.

## Why v4 is not treated as final

Availability-aware ESM+GO fusion reduced K562 perturbation-OOD validation loss from
`0.684799` to `0.681450`, but the 0.49% improvement is incremental. A control-state
teacher query reached `0.681129`, only `0.000321` better than its static control and
well below the locked `0.003` complexity margin. Direct teacher fusion therefore
remains useful, but neither its scale nor its transfer result justifies a state-of-the-art
claim.

## Adaptation from SALT and ProtJEPA

SALT trains an asymmetric student to predict a static frozen-teacher representation.
The user-supplied ProtJEPA manuscript adds two relevant mechanisms: masked-teacher
reconstruction before distillation and per-dimension standardization of cached joint
targets using training-only statistics. Its reported structural-retrieval regression is
also a warning that cross-modal distillation can suppress information already present
in the sequence representation.

Revision v5 consequently has two outcome-free phases:

1. An ESM+GO joint teacher hides exactly one available modality and reconstructs it
   from the other. Both encoders remain frozen; only projectors and decoders train.
2. The joint teacher freezes. Training-split joint targets are standardized per
   dimension, cached, and predicted by a small student whose only input is the frozen
   320-dimensional ESM-2 vector.

The downstream action cache concatenates raw frozen ESM with the student encoder
representation. This residual sequence path is the deliberate deviation from
ProtJEPA: it prevents the distilled branch from having to preserve every useful ESM
direction while still making cross-modal public knowledge available.

ProtJEPA can identify sample-dependent attention because several of ten teachers
remain visible when others are masked. With two teachers, hiding exactly one leaves no
attention choice. The v5 joint teacher therefore uses availability-normalized equal
fusion; pretending to learn a gate in this setting would add an underidentified
parameter rather than biological capacity.

## Frozen controls and gates

The public-data split is a deterministic 80/10/10 target-gene split among targets with
both teachers. Teacher and student checkpoints are selected only on the public
validation split. Required controls are raw ESM and a closed-form ridge predictor of
the standardized teacher target. Geometry, target cosine, and GO-teacher-neighbor
agreement are reported on the public test split.

No GPU dynamics experiment is justified unless tensors are finite, standardization
matches the frozen thresholds, the student avoids collapse, target cosine is at least
`0.50`, and the residual representation improves GO-neighbor overlap over raw ESM by
at least `0.005`. If those gates pass, the frozen residual cache receives the same
outcome-restricted anchor and dynamics protocol as v4. It must improve K562
perturbation-OOD validation loss by at least `0.003` over `0.681450` to replace v4.
Viewed K562 test, RPE1, HepG2, or Jurkat outcomes cannot change either decision.

The older architectures remain frozen baselines even if v5 wins; only the selected
default configuration changes. Deleting them would break reproducibility and make the
incremental evidence impossible to audit.
