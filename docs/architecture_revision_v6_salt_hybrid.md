# Architecture revision v6: residual ESM+GO+joint teacher

## Motivation

The ESM-only SALT ridge student reconstructed the public joint-teacher target well, but
failed the downstream K562 anchor gate. It improved slightly over ESM-only while losing
most of the gain provided by direct GO. The result shows that distillation transfers
some cross-modal structure but cannot recover GO variation that sequence does not
identify.

Revision v6 therefore does not replace a successful teacher with its weaker student.
It preserves raw frozen ESM and direct frozen GO as separate modalities, then adds the
nonlinear masked-teacher joint embedding as a third residual modality. All three are
cached from public, outcome-free resources. The ridge and nonlinear students remain
frozen ablations.

## Frozen representation

The joint branch replays the exact v5 masked-teacher checkpoint. Its ESM and GO inputs
use public-training-only statistics, and its output is standardized using the joint
target mean and variance from the same 742 public-training targets. When both public
teachers exist, the joint state fuses both. When only one exists, the teacher's masked
reconstruction pathway supplies the joint state. No perturbation outcome or cell-line
identifier enters the cache.

The action cache layout is raw ESM `320`, raw GO `64`, standardized joint teacher `256`,
then three availability bits. The original ESM+GO prefix and availability are preserved
exactly, making the added branch auditable.

## Downstream isolation

The low-rank effect anchor will use only the original ESM+GO modalities and their
availability bits. This should reproduce the v4 prior instead of allowing the larger,
partly redundant joint branch to change ridge regularization. The population dynamics
model alone receives all three branches, so any validation improvement isolates the
new nonlinear joint representation.

The hybrid must first pass public-feature CPU checks and the existing K562 anchor gate.
Only then may a bounded CUDA checkpoint/resume smoke run proceed. Full GPU training is
eligible only after that smoke run and must improve K562 perturbation-OOD validation
loss by at least `0.003` over v4's `0.681450`. Sealed test and external outcomes cannot
change the architecture decision.

## Completed public-feature export

The CPU export preserved the ESM+GO prefix and availability bits exactly. Train-split
joint standardization errors were below `1.2e-7`, the validation joint stable rank was
`13.3289`, and the frozen teacher's validation reconstruction replay error was zero.
All public-feature gates passed without reading the previously viewed public test or
any perturbation outcome. The resulting `643`-dimensional cache covers 996 targets and
is eligible for the isolated effect-anchor check.
