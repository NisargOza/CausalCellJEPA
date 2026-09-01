# Publication positioning and figure design

## Recommended title

**CausalCellJEPA: A Multimodal Joint-Embedding Predictive Architecture for Cellular
Perturbation Response Modeling with Biology-Aware Multi-Teacher Fusion**

“Response modeling” is more defensible than “world modeling” because the completed
external evaluation tests intervention-conditioned transcriptomic response, not broad
simulation, planning, or general cellular dynamics. Leakage controls remain a methods
strength rather than a title claim.

## Evidence-calibrated contribution

CausalCellJEPA is a controlled test of four coupled ideas: a frozen JEPA cell-state
space, continuous biological action representations, baseline-population context, and
unpaired distributional dynamics. The strongest defensible contribution is not global
leaderboard dominance. It is the combination of a strict perturbation-by-context OOD
design, condition-level uncertainty, external one-shot confirmation, and an explicit
diagnosis of the direction–calibration and seen–unseen transfer gaps.

The completed evidence supports these statements:

- On Replogle double OOD, the full distributional model is substantially better than
  pseudo-pairing and the matched ablations on magnitude error, MMD, and Sinkhorn, while
  losing on latent effect direction. This supports calibrated population modeling, not
  uniform superiority.
- Two of three Stage 2 seeds beat the linear ESM baseline on latent cross-context effect
  Pearson, but the frozen transcriptomic decoder does not preserve that advantage.
- On Adamson, the frozen candidate beats the perturbed-mean systematic baseline by
  `0.2606` Systema Pearson delta with a paired 95% interval `[0.1563, 0.3700]`; the
  target-excluded improvement is `0.2574` with interval `[0.1529, 0.3650]`.
- The Adamson candidate is `0.0204` below the best frozen STRING+GO component in mean
  Systema, exceeding the preregistered `0.01` tolerance. Absolute Systema performance
  is negative on the eight outcome-fit-unseen targets. Global state of the art is not
  supported.

## Primary-literature map

The table records the papers that directly influenced the manuscript position or figure
design. Peer-reviewed papers and active preprints are labeled rather than treated as
equivalent levels of evidence.

| Work | Status | Relevant lesson for CausalCellJEPA |
|---|---|---|
| [I-JEPA](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | CVPR 2023 | Predict semantic latent targets instead of reconstructing every noisy observation. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 2025 preprint | Separate representation learning from action-conditioned dynamics. |
| [ProtJEPA](https://www.biorxiv.org/content/10.64898/2026.08.03.742606v1) | 2026 preprint | Multi-teacher fusion is credible only with deployment-relevant missing-modality tests and load-bearing ablations. |
| [GeneJEPA](https://www.biorxiv.org/content/10.1101/2025.10.14.682378v1) | 2025 preprint | Latent prediction over gene sets is a viable transcriptomic pretraining objective. |
| [Cell-JEPA](https://arxiv.org/abs/2602.02093) | 2026 preprint | Better state representations need not improve perturbation-effect estimation. |
| [BioM-JEPA](https://arxiv.org/abs/2608.05928) | 2026 preprint | Report representation geometry, biological-program retention, response error, and matched ablations together. |
| [BioJEPA-AC](https://github.com/GPTomics/biojepa) | active software | Action-conditioned cellular JEPA is not itself a novelty claim. |
| [scGen](https://www.nature.com/articles/s41592-019-0494-8) | Nature Methods 2019 | Latent vector transfer is an essential historical baseline. |
| [CPA](https://doi.org/10.15252/msb.202211517) | Molecular Systems Biology 2023 | Disentangled perturbation, dose, and context modeling motivates compositional comparisons. |
| [GEARS](https://www.nature.com/articles/s41587-023-01905-6) | Nature Biotechnology 2024 | Graph priors support unseen-action prediction, but comparisons must use matching splits and metrics. |
| [CellOT](https://www.nature.com/articles/s41592-023-01969-x) | Nature Methods 2023 | Destructive assays justify distributional transport rather than observed cell pairing. |
| [CINEMA-OT](https://www.nature.com/articles/s41592-023-02040-5) | Nature Methods 2023 | Causal language requires explicit confounding assumptions and careful counterfactual interpretation. |
| [PerturbNet](https://doi.org/10.1038/s44320-025-00131-3) | Molecular Systems Biology 2025 | Distributional prediction and continuous unseen-perturbation representations are established ideas. |
| [State](https://www.biorxiv.org/content/10.1101/2025.06.26.661135v2.full) | 2025 preprint | Use permutation-invariant control populations and evaluate context transfer with baseline controls available. |
| [scDFM](https://proceedings.iclr.cc/paper_files/paper/2026/hash/75f029a1289a47aa99c86588239f0c12-Abstract-Conference.html) | ICLR 2026 | MMD plus distributional flow matching reinforces multi-endpoint population evaluation. |
| [Stable-Shift](https://arxiv.org/abs/2606.24940) | 2026 preprint | Low-rank response bases plus STRING, GO, and control statistics are strong unseen-gene priors. |
| [STRAND](https://arxiv.org/abs/2602.10156) | 2026 preprint | Sequence-conditioned transport is a relevant zero-shot alternative, especially beyond protein-coding targets. |
| [CisTransCell](https://arxiv.org/abs/2606.13713) | 2026 preprint | Coding, regulatory, and control-state priors should be distinguished rather than described as one generic modality. |
| [Lingshu-Cell](https://arxiv.org/abs/2603.25240) | 2026 preprint | Broad “cellular world model” claims require whole-distribution generation across diverse tissues and contexts. |
| [Simple linear baseline benchmark](https://www.nature.com/articles/s41592-025-02772-6) | Nature Methods 2025 | Complex models must beat deliberate mean and linear baselines on perturbation effects. |
| [Systema](https://www.nature.com/articles/s41587-025-02777-8) | Nature Biotechnology 2025 | Control-referenced correlations can reward systematic variation; perturbed-reference metrics and centroid accuracy are primary here. |
| [27-method, 29-dataset benchmark](https://www.nature.com/articles/s41592-025-02980-0) | Nature Methods 2025 | Perturbation and context generalization must be shown separately across multiple metrics. |
| [scPerturb](https://www.nature.com/articles/s41592-023-02144-y) | Nature Methods 2024 | Harmonized provenance and energy-distance analyses improve cross-study interpretability. |

## Figure system

`scripts/generate_publication_figures.py` renders every panel from hash-verified frozen
artifacts according to `configs/publication_figures.yaml`. It writes editable SVG,
font-embedded PDF, 300-dpi PNG, and a manifest containing source and output hashes.

1. **Architecture:** two-stage student/EMA-teacher JEPA, frozen biological action
   teachers, control-population encoder, and explicitly unpaired set transition.
2. **Evaluation design:** the four Replogle generalization regimes plus the Adamson
   controls-only prediction/freeze/outcome timeline.
3. **Distributional trade-off:** raw double-OOD perturbation conditions and bootstrap
   intervals for effect direction, magnitude, MMD, and Sinkhorn.
4. **Seed and readout transfer:** latent versus decoded cross-context effects for three
   independently selected JEPA seeds and the linear ESM baseline.
5. **Adamson confirmation:** Systema model comparisons and paired target-level plots
   against the systematic baseline and best frozen component.
6. **Generalization and claim audit:** target-level external scores, paired improvements,
   and the preregistered five-of-six stopping decision.

The plotting policy avoids truncated axes, cell-level pseudo-replication, bar-only
summaries, retrospective target removal, and hidden negative results. Raw points use
perturbation conditions after repeat averaging; every displayed interval resamples
perturbation identities 10,000 times.
