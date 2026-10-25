# BF16 head numerical-state repair

The TopDial goal remains improvement of TDRL-R/Full against fixed PPDPP. A
training-only checkpoint audit revealed a numerical defect independently of
new dialogue outcomes. Source: `artifacts/head-numeric-audit-58159/audit.json`.
At the old final checkpoints, one additional tau=.01 BF16 EMA update leaves
99.91% of MRNFixed and 99.79% of IQE target parameters unchanged, although most
online/target values differ. This is a next-update measurement, not a claim that
those elements never changed during training. Optimizer moments are also BF16.

A CPU arithmetic fixture repeatedly applies the same update 1000 times: target
1.0, online 1.0078125, tau .01. BF16 target stays at 1.0; FP32 accumulation reaches
1.0078065. This proves the update-loss mechanism, not its effect on success rate.

Numerical revision 2 preserves model parameters, inputs and head forward in BF16;
MRNFixed/IQE formulas, argument order, loss, step cost, gamma, learning rate,
weight decay, clipping, sample sequence and 30,000-step budget remain unchanged.
Use standard Torch AdamW on FP32 auxiliary optimizer masters/moments, publishing
BF16 forward weights after each step. Accumulate target EMA in an FP32 auxiliary
master and publish BF16 target weights. These numerical states are not additional
inference models or a change of generation/embedding precision. Checkpoints record
revision and all masters; reject old numerical execution for resumed fitting.
Fit fresh, keeping old checkpoints and compute immutable.

Canonical H100 command (one visible H100/eight CPUs in allocation 58159):

```sh
runtime/.venv/bin/python scripts/train_head_numerics.py --head MRNFixed --output artifacts/head-numerics-v2-58159/MRNFixed
```

Repeat for IQE with identical data/budgets. Before fitting, require real frozen
4096-D training embeddings, identical initial BF16 online/target outputs and exact
repeated gradient/optimizer/EMA/RNG replay with strict deterministic execution.
Fixture updates never modify imported scientific checkpoints. After fitting,
check original invariants/tolerances and the same held-out temporal diagnostics.
A later dialogue evaluation must freshly calibrate thresholds and pass runtime/
planner gates with the new checkpoints. No old selection/test cache is reusable.

The ongoing 64-scenario planner-v2 benchmark stays on its committed code and old
fitting. Do not deploy numerical repair into running workers. Any numerical
repair comparison must retain the frozen baseline and disclose this additional
training segment; improved distance diagnostics alone cannot establish beating
PPDPP or independent post-test confirmation.
