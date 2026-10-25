DiffTOD nonzero-checkpoint replay repair

AdamW's CPU step tensors alias the supplied optimizer state on restore. The first
probe advances the in-memory source step counter; a second probe restores that
mutated counter. This affects resumed probes, not the uninterrupted fitting path.

The repair deep-copies optimizer checkpoint state before each restore. Training
math, architecture, data, RNG, budget, validation and selection stay unchanged.
AST/source audits reject other fitting changes. Both H100s must reproduce the
original alias, unchanged losses/gradients, exact repaired optimizer/EMA replay,
immutable source state and zero committed updates. The original update-82360
checkpoint and failed resume remain preserved. Resume requires a checksum-bound
TDRL_DIFFTOD_RESUME_AMENDMENT and retains the original checkpoint contract/clock.
