# Fixed test subset (user-authorized 2026-10-08)

The scoped PPDPP / TDRL-R / TDRL comparison evaluates **2000 total scenarios**,
sampled uniformly without replacement from the pooled official TopDial and
DuRecDial2.0 test splits. This is a sampled test evaluation, not completion of
the full official-test or paper matrix. Sampling seed **42** is distinct from
the single experimental seed **17** (user amendment later on 2026-10-08).
Both dataset policies and both reachability heads use training seed 17;
development and test also use seed 17. Budgets and training case counts remain unchanged.
Derived request/candidate streams are retained. Scenario-paired bootstrap CIs
measure scenario uncertainty, not training or experimental-seed variability.

Before fitting/selection, freeze `configs/comparison-judge-v2-sample2000-seed17.json`.
After verifying `selection.lock`, audit the full test population for leakage,
sort `(dataset, scenario ID)` pairs, and use Python `random.Random(42).sample`
to draw 2000 pairs. Persist sorted selected IDs, full population membership hash,
sample membership hash, config/selection hashes and dataset counts in
`test/test-sample.json`. Resume rejects a changed manifest. Every method,
budget, evaluation seed and user protocol uses those same IDs. No success,
target category or generated outcome influences inclusion.

Keep natural and matched budgets and both profile/resistant protocols:
**16,000 unique evaluated dialogues** (2000 × 1 seed × 2 protocols × 4 cells).
PPDPP and TDRL-R matched reports reuse their natural records as before.
Reports include the sample manifest and use the selected membership for
completion checks and scenario-paired confidence intervals.

Preflight projects expected dataset composition from public population counts;
actual membership/counts are obtained only after lock. This is a development
timing proxy, not a scientific result or guaranteed ETA. Retain the original
allocation 57711 clock and all previous failed preflights. Recheck actual
method-specific timing during test and stop if the remaining allocation cannot fit.

Canonical command inside the registered allocation:

```sh
runtime/.venv/bin/python -m tdrl repair-ppdpp \
  --source artifacts/low-sr-fix-57711 \
  --output artifacts/ppdpp-sample2000-seed17-57711 \
  --config configs/comparison-judge-v2-sample2000-seed17.json
```
