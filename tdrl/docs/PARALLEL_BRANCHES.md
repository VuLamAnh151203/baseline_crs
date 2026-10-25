# Concurrent candidate branches for TDRL Full

User authorization: 2026-10-09, "chạy đồng thời 3 nhánh candidate của Full".

`TDRL_CANDIDATE_WORKERS=3` overlaps the three independent branches. Within each
branch the candidate and alternating user/assistant lookahead remain sequential.
Each branch owns its replay IDs; results are assembled by candidate index, then
frozen-head leaf scoring and graph access run on the controller thread. Preserve
all prompts, seeds, routing, refusals, lookahead two, budgets and tie-breaking.
Failures drain every submitted branch before returning. Both model servers retain
the existing BF16 residency, ports and generation maximum of four sequences.

GDN generation can change with batching; no serial/parallel text equality claim.
CPU tests establish exact decision/seed/request equivalence only for fake models.
Actual H100 smoke checks both fitted MRNFixed and IQE, three complete branches,
real normalized embeddings, frozen budget enforcement, exact generation-journal
and selected-plan replay, inherited concurrent critic replay and exact BF16
controller action/logprob/gradient replay, with zero optimizer updates.

Existing fitting and PPDPP RL checkpoint/RNG/optimizer remain compatible. Preserve
all old selection artifacts and rerun both datasets/heads from fresh
`selection-branches-*` tasks, so compute-based head selection compares one execution
protocol. Global selection and official test stay locked until fresh shards and
policy diagnostics finish. Config/scenario/seed/model contracts are unchanged;
record source/current commits, original manifests and allocation 58159 clock.

Canonical transition (controller uses only stdlib; all inference is Slurm/H100):

```sh
python3 scripts/transition_branch_parallel.py --root <tdrl-root> --bundle <committed-git-bundle> --commit <target-commit> --job-id 58159
```

The transition stops only the verified owned queue controller, lets the current
selection stage finish while RL continues, then requests a durable RL boundary.
Only after both steps exit and pending requests are absent does it update source,
run independent smoke on both GPUs and enable the fresh queue. A failed gate
retains evidence and stops; it never releases or replaces the allocation.

Smoke stage command (one visible full H100/eight CPUs):

```sh
runtime/.venv/bin/python scripts/smoke_branch_parallel.py --slot 0 --shared artifacts/two-gpu-58159 --output artifacts/branch-execution-58159/smoke-0 --checkpoint <durable-PPDPP-checkpoint>
```

Transition phase and audit: `artifacts/branch-execution-58159/transition-status.json`
and, after both smokes pass, `amendment.json`. Microbenchmark speedup is an
engineering measurement, not a test result or whole-pipeline speedup.
