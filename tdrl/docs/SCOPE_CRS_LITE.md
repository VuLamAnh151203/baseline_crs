# SCOPE-CRS-lite reproduction

This package adapts the released SCOPE planning mechanism to the shared CRS
evaluation without changing any TDRL or other-baseline module.

## Method boundary

- Candidate realization uses the frozen local Qwen3.5-9B generation backend,
  with five independent candidates per executed turn.
- Semantic states remain the original projected 1024-dimensional
  Meta-Llama-Guard-2-8B space. This is intrinsic to the released transition
  checkpoints and is not replaced by Qwen embeddings.
- The four released LMSYS-trained deterministic transition seeds are used in
  both directions. The loader fixes the upstream path bug and loads the files
  that are actually released (`model_min_val.pth`); it never silently skips a
  failed seed.
- Native planning settings are retained: depth 6, three seconds of semantic
  MCTS, gamma 0.9, five candidates, and the online Q approximator.
- Because no released CRS task-reward checkpoint exists, each dataset gets a
  goal-conditioned reward head fitted only on its train/dev split. Weak action
  progress and the shared semantic judge provide labels for user-ended states;
  minimum dev BCE is selected before any official test body is opened.
- This is therefore reported as **SCOPE-CRS-lite**, not source-faithful SCOPE.

## Immutable assets

Public transition checkpoints are downloaded as an I/O-only task on the
networked login node, as prescribed by the cluster guide. This does not import
or execute either model. Files are written to:

`/mnt/data/u26anhvl/crs/scope/assets/transition_models/deterministic`

```bash
bash scripts/download_scope_transition.sh
```

If Google Drive supplies three split ZIP archives, upload them to the parent
directory and import/hash them on a CPU node with:

```bash
bash scripts/submit_scope_transition_import.sh
```

Meta-Llama-Guard-2-8B is gated by Meta. Place an authorized, revision-pinned
snapshot at:

`/mnt/data/u26anhvl/crs/scope/assets/Meta-Llama-Guard-2-8B/7d257f3c1a0ec6ed99b2cb715027149dfb9784ef`

Then build its hash manifest with the supplied CPU Slurm job:

```bash
bash scripts/submit_scope_llama_manifest.sh
```

The runtime is offline and fails closed if either asset differs from its
manifest.

## Gates and evaluation

Run these in order after committing the SCOPE files:

```bash
bash scripts/submit_scope_cpu_gate.sh
bash scripts/submit_scope_mini.sh
bash scripts/submit_scope400.sh
```

The fixed-400 job evaluates the immutable first 200 TopDial and first 200
DuRecDial scenarios under both profile and resistant protocols, seed 17,
natural budgets, the shared simulator and the shared semantic judge. The two
datasets run concurrently, one H100 each. Raw dialogue records, request
journals, reward artifacts, method lock, seals, metrics and stage logs are
stored below `/mnt/data/u26anhvl/crs/scope/`.

The result is a single-seed, task-adapted sampled-test baseline. It is not a
claim that the unreleased original task reward was reproduced, and it does not
measure training-seed uncertainty or human evaluation.
