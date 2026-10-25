# MRN retrieval-candidate diagnostic

The v5 discovery diagnostic gave MRN R 27/64 profile and 12/64 resistant
successes, versus Full 32/64 and 8/64. These are generated values from
`artifacts/topdial-nearest-debug-v5-MRN-only-58159/report.json`; paired intervals
versus fixed PPDPP contain zero. The frozen v4 rollout-terminal audit established
that hypothetical acceptance and leaf cost can disagree, but its resistant
sample did not establish discarded acceptance as the cause of v5 SR losses.

This next experiment isolates a separate candidate-generation inconsistency:
R uses the final-response seed stream, while original Full candidate zero uses
another stream. Full therefore need not consider the response R would propose.
The opt-in `preserve_retrieval_candidate=true` makes candidate zero use precisely
R's realization guidance, visible history, decoding and final-response seed.
It applies only to planner revision 2. Default false preserves old request replay.

No additional generation calls are added: Full keeps three candidate branches,
two sequential lookahead steps, concurrent branch execution and the same maximum
12 assistant/simulated-user calls. Candidate one/two and all rollout seed streams
are unchanged. Leaf scoring, reference selection, numerical MRN checkpoint,
calibration quantiles, assistant/simulator/outcome judge and both protocols remain
fixed. The semantic terminal issue is still unresolved by this intervention.

GDN generation depends on batching. Matching prompts/seeds does not establish
serial R and concurrent candidate-zero text equality; we make no such assertion.
Both GPU smoke checks must assert matching prompt hash/seed/decoding, the frozen
call ceiling, exact cached plan/journal replay and existing controller replay.
Smoke inference is recorded separately from evaluation inference.

Canonical command from `tdrl/` in allocation 58159:

```sh
runtime/.venv/bin/python scripts/run_topdial_debug.py \
  --config configs/topdial-retrieval-candidate-debug-v6.json \
  --output-prefix topdial-retrieval-candidate-debug-v6
```

Both eight-CPU/one-GPU workers use MRN only. After both GPU smokes pass, slot zero
runs R and slot one Full, each on the same 64 discovery scenarios and both
protocols (128 dialogues per worker, root seed 17). The reporter verifies both
shards and the 128 frozen PPDPP development records, producing 384 paired records
via `--mrn-split`. Historical IQE/selection/test evidence is retained. The frozen
disjoint confirmation set is not opened by this command. This is post-test
development debugging; neither a smoke nor a point estimate proves superiority.
