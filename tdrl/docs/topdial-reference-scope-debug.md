# TopDial reference-scope debugging

Evidence: `artifacts/topdial-reference-audit-58159/audit.json`, derived exclusively
from hash-verified completed MRN development dialogues and the training-bank
index. No inference or official test resource was used in this audit.

`artifacts/topdial-reference-audit-58159/reference-conditioning-breakdown.json`
separates references chosen with an exact target anchor from declared goal-type
transfer. The original aggregate different-target counts include both conditions;
they must not all be described as erroneous routing with an available exact
anchor. The selected new-head discordant trace in
`numerical-head-discordant-traces.json` is a transfer case, so restricting the pool
to exact targets cannot repair that case. Both unsupported and supported scenarios
remain in every comparison.

Planner revision 2 conditions goal anchors on the current declared target, but
its intermediate reference pool still includes every training item of the same
goal type. Consequently, an exact-target goal set does not ensure the selected
next-move reference concerns that target. This is a separate hypothesis from
numerical optimizer/EMA loss; the numerical diagnostic remains unchanged.

The opt-in `make_graph(..., reference_scope="exact_target_when_supported")`
restricts the whole reference pool to training trajectories with the same
normalized declared target whenever an exact goal anchor exists. It keeps
intermediate references, preserves directed distances and edge construction,
and leaves the node budget and next-strategy realization unchanged. Unseen
targets retain explicit goal-type transfer; they are never dropped. The default
`goal_type` scope preserves current experiments.

Status: implementation and control-flow regression passed; no H100 runtime gate
or dialogue comparison has established improvement. The separate frozen config
`configs/topdial-targetpool-debug-v4.json` changes only reference scope relative
to the numerical diagnostic. It retains the same heads, thresholds rules,
development membership, seeds, prompts, budgets and fixed PPDPP evidence. This
config is frozen before numerical-dialogue outcomes are available. Run it only
after the numerical diagnostic completes, with fresh both-GPU replay gates:

```sh
runtime/.venv/bin/python scripts/run_topdial_debug.py \
  --config configs/topdial-targetpool-debug-v4.json \
  --output-prefix topdial-targetpool-debug-v4
```

Expected: 512 new dialogues and 128 hash-audited fixed PPDPP records. Preserve
numerical and target-pool reports separately; interpret repeated use of the same
development cases as exploratory debugging, not independent confirmation.
Both-GPU smoke fixtures are chosen from training-anchor availability alone:
the first exact-anchor and first transfer case in frozen discovery order. Each
head must replay covered and recovery plans for both fixtures. The exact-anchor
gate also rejects any reference from another declared target. Evaluation scenario
order, prompts, root seed and budgets are unchanged by these smoke-only fixtures.
Weak strategy labels remain a separate limitation. A future semantic annotation
experiment must label training next moves only, retain label provenance/costs,
and keep assistant/simulator/judge contracts unchanged.
