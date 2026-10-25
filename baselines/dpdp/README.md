# DPDP

DPDP preserves a fast learned actor-critic policy, confidence routing and slow open-loop PUCT
MCTS. The full controller prepares train/dev data, annotates rewards with local Qwen, performs
offline training, runs MCTS self-play, selects on development and evaluates the frozen checkpoint.

The deadline configuration uses the native depth 5 and 10 simulations while adapting self-play to
50 cases per dataset per round (five rounds). This is a declared compute adaptation and may reduce
SR relative to a larger native self-play budget.

```bash
bash scripts/submit_baseline.sh dpdp mini
bash scripts/submit_baseline.sh dpdp full400
```

Resume the same physical run root with `BASELINE_RUN_ROOT` and `BASELINE_RESUME=1`; never create a
fresh root and call it a resume. Detailed checkpoints, stages and limitations are documented in
`tdrl/README_DPDP.md`.
