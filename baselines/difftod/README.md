# DiffTOD

Untouched author code: [`upstream/`](upstream/) at commit
`fd46b2887a9c0d9e0e413623e79c72ee32b9b7c1`. Evaluation adaptation remains under `tdrl/`.

Dataset-specific discrete diffusion trajectory planning. The adapter retains SEDD likelihood,
score-entropy training, analytic reverse sampling, five semantic paraphrases and self-BLEU/MBR
consensus, with shared Qwen realization and evaluation.

DiffTOD requires explicit preparation, training, development plan generation, measured budget
selection, immutable selection locking and only then official test evaluation. Those stages cannot
honestly be collapsed into a preselected one-command job on a fresh machine. Follow
`tdrl/README_DIFFTOD.md` from bootstrap through Section 8. Each stage is resumable and Slurm-backed.

The public code is runnable, but a scientific test remains blocked until the model lock and
development-selected configuration are valid on the new machine.
