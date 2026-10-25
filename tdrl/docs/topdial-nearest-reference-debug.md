# Nearest-reference development diagnostic

Concern: next-move guidance should correspond to the observed local state. Audit
`artifacts/topdial-reference-audit-58159/reference-selection-audit.json` counts
reference choices from checksum-verified completed target-pool records. Many
choices skip the nearest reference and select a one-node goal path.

For a quasimetric, `d(live,g) <= d(live,r) + d(r,g)`. A directed path from r to g
costs at least d(r,g), apart from documented numerical tolerances. Therefore
minimizing entry distance plus remaining path cost over references including g
can prefer g itself, losing the local next move. Coverage still uses nearest
entry distance, so dispatch and guidance can bind to different references.

The opt-in `reference_selection="nearest_entry"` removes this extra ranking.
Retrieval order supplies the closest live-to-reference state; its directed
shortest path and observed next-action guidance remain intact. Full's leaf cost,
MRN/IQE formulas/checkpoints, exact-target pool/fallback, calibration rules,
prompts, decoding, seeds and candidate/lookahead budgets are unchanged. Historical
`entry_plus_path` remains the default for replaying earlier experiments.

Regression: a triangle-consistent fixture verifies that a close intermediate
reference supplies the next move rather than being replaced by a distant goal,
with unchanged coverage and assistant call count. Real H100 exact/transfer,
covered/recovery and journal/plan-replay gates on both GPUs precede evaluation.

Canonical command in allocation 58159:

```sh
runtime/.venv/bin/python scripts/run_topdial_debug.py \
  --config configs/topdial-nearest-debug-v5.json \
  --output-prefix topdial-nearest-debug-v5
```

Expected: 512 new discovery-development dialogues and 128 audited fixed PPDPP
records, reported under `artifacts/topdial-nearest-debug-v5-58159/`. The separate
confirmation membership remains unopened for inference. This is post-test
development debugging; no improvement or independent test confirmation is
established by the audit or regression alone. Weak next-action labels and
unsupported-target transfer remain limitations.
