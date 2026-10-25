# TopDial numerical-head development diagnostic

This is post-test, development-only debugging, not independent test confirmation.
It compares TDRL-R and Full against the fixed PPDPP checkpoint and its completed
64-scenario development records. Root seed 17, the two user protocols, scenario
order, assistant/simulator/judge contracts, BF16 forward, head formulas and
planner revision 2 remain fixed. Official test resources are never opened.

Source fitting: `artifacts/two-gpu-58159`. Fresh numerical heads:
`artifacts/head-numerics-v2-58159/{MRNFixed,IQE}`. Both have 30,000 training steps,
FP32 auxiliary optimizer/EMA masters and exact H100 update-replay gates. Original
heads and all earlier dialogue evidence are retained.

New distance scales require fresh thresholds: H_edge is the 0.8 quantile of 1,024
ordered training-bank pairs drawn with seed 17; H_cover is the 0.75 quantile on
the same frozen 16 development calibration cases. These rules are frozen before
new dialogue outcomes. Evaluation uses the unchanged, disjoint 64 development
cases and both protocols. Each GPU evaluates one head, R followed by Full; two
distinct-GPU runtime/controller/plan-replay smokes must pass first.

Canonical command, after the old diagnostic steps finish and source is committed:

```sh
runtime/.venv/bin/python scripts/run_topdial_debug.py \
  --config configs/topdial-numerics-debug-v3.json \
  --output-prefix topdial-numerics-debug-v3
```

Expected output: 512 new dialogues plus 128 reused PPDPP records, audited against
their complete manifest, shared provenance, checkpoint and paired outcome
contracts. Report paths:
`artifacts/topdial-numerics-debug-v3-58159/{report.json,report.md}`.
The report includes paired 95% bootstrap intervals with 10,000 resamples.
Higher point estimates alone do not establish improvement, and numerical
remaining-step diagnostics do not establish dialogue success. New fitting and
dialogue compute are recorded separately from reused source evidence.
