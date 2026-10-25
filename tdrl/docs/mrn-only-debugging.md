# MRN-only TopDial debugging

The 2026-10-12 user scope amendment stops the IQE evaluation while preserving
its journals and complete/partial results. The ongoing MRN evaluation keeps its
existing contract. Fixed PPDPP remains the comparison baseline; it is not retrained.

After `topdial-nearest-debug-v5-58159/eval-0` is complete, run from `tdrl/`:

```sh
python scripts/summarize_topdial_debug.py artifacts/topdial-nearest-debug-v5-58159 --mrn-only
```

This command requires all 256 MRN dialogues plus 128 fixed PPDPP dialogues,
verifies completed source manifests and record hashes, shared model/prompt/data/
seed contracts, PPDPP checkpoint and identical scenario order. It does not read
the stopped IQE evaluation. It reports scenario-paired 10,000-resample CIs both
against PPDPP and for Full minus R, including discordant wins/losses.
These are single-seed post-test development diagnostics, not independent test
confirmation or evidence of training-seed variability.

## Remaining mechanism audit

Full minimizes entry distance plus directed path cost at rollout leaves. Calling
its negative a value makes maximizing value equivalent to minimizing cost; the
optimization direction is not itself a bug. The remaining question is whether
this learned cost ranks actual unpressured target completion correctly.

The frozen `artifacts/topdial-reference-audit-58159/rollout-terminal-audit-inputs.json`
contains source-hashed development branches selected independently of outcomes.
Future semantic auditing uses only its MRN branches, the existing local Qwen
outcome judge, and each observed user prefix with its recorded finish reason.
The judge must not see candidate identity, distance, selected status or outcome
of the real dialogue. Preserve ambiguity; do not replace judgments with text
matching. Measure achieved branches that were discarded and additional rollout
steps after achievement before deciding whether a planner change is warranted.
This audit adds diagnostic inference only; it does not retroactively alter the
completed comparison or silently change Full's generation budget.

The executable audit is now `scripts/audit_rollout_terminals.py`, with frozen
input and comparison-config hashes in `configs/mrn-rollout-terminal-audit-v1.json`.
It judges 70 observed user prefixes from the 48 MRN branches. The local semantic
judge receives observed history and permitted task context, never cost or chosen
candidate metadata. Report both discarded accepted branches and continuations
after hypothetical acceptance/refusal; these are counterfactual diagnostics,
not a causal success estimate. Both original models remain resident on the same
allocated H100. Live 4096-D exact embedding replay and exact judge journal replay
must pass before prefix judgment. Record cold loading, allocated H100 seconds,
actual role calls/tokens, allocator telemetry and source hashes in the manifest.

After allocation/process inspection, launch one exclusive eight-CPU/one-GPU
step in the existing allocation (slot 1 uses ports 18100/18101):

```sh
srun --jobid=58159 --ntasks=1 --cpus-per-task=8 \
  --gres=gpu:nvidia_h100_80gb_hbm3:1 --exclusive --exact \
  runtime/.venv/bin/python scripts/audit_rollout_terminals.py --slot 1 \
  --output artifacts/mrn-rollout-terminal-audit-v1-58159
```

The source extraction remains outside package code and its frozen payload is
gitignored; copy that exact hash-verified payload along with source artifacts
when reproducing. Missing/changed files fail closed. The audit runner is
resumable via the generation journal; unresolved dispatched requests require
inspection before a retry, not automatic regeneration.

Run inference only on an observed free GPU inside allocation 58159 after live
scheduler/process inspection. A failed SSH observation does not establish that
an evaluation stopped or authorize a duplicate worker. No superiority claim is
supported until new paired evidence and the frozen disjoint development
confirmation pass.
