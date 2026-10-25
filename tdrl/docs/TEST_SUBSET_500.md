# Fixed 500-scenario evaluation (2026-10-08)

This user amendment supersedes the 2000-scenario scope. Sample **500 total**
official test scenarios uniformly without replacement from both datasets, seed
**42**. Reuse identical IDs across every method, budget, protocol and the single
experimental seed **17**. Keep profile/resistant and natural/matched budgets:
**4000 unique evaluated dialogues**. PPDPP and natural TDRL Full run concurrently;
TDRL-R and matched TDRL remain sequential. All fitting/development budgets remain.

Freeze `configs/comparison-judge-v2-sample500-seed17.json` before selection.
The original run drew its seed-42 sample after verifying `selection.lock`.
User amendment 2026-10-10 freezes that exact completed subset: subsequent runs
verify their own selection lock, open official test bodies and replay the packaged
IDs/indices without random sampling. Persist `test/test-sample.json` with the
pinned-resource checksum as well as population/subset hashes.
Report sampled-test results; scenario CIs do not measure training-seed variability.

Preserve compatible complete fitting and the exact committed RL checkpoint,
optimizer/RNG, episode schedule and original training contract. Record source
segment compute and hashes. A fresh controller replay checks the imported policy
before another update. Retain all old artifacts and allocation 57711's clock.

ETA scales measured test timing by 500/2000; training is not scaled. Include the
25% safety factor and one-day reserve. Existing timing is a development proxy
with provisional covered/off-map thresholds and partial SFT, not official-test
performance or a guaranteed completion date. Recheck after fitting/selection.

```sh
runtime/.venv/bin/python -m tdrl parallel-comparison \
  --source artifacts/ppdpp-tdrl-concurrent-v2-57711 \
  --output artifacts/ppdpp-tdrl-sample500-57711 \
  --config configs/comparison-judge-v2-sample500-seed17.json
```


## Cross-machine fixed membership (2026-10-10)

Canonical resource: `src/tdrl/resources/test_scenarios_500_seed42.json`, packaged
with the Python library and loaded through `importlib.resources`, independent of
working directory. SHA-256:
`a56f7a0c909bd1ac460f41904da12f99582e98d4ef0b602f2593f7ab21870cb0`.

The resource records exactly **330 TopDial + 170 DuRecDial2.0** IDs, their original
order, zero-based indices within each dataset's lexicographically sorted ID
population, fixed dataset evaluation order (TopDial then DuRecDial2.0), population/subset checksums and canonical normalized-content SHA-256
for each selected scenario. Indices refer to that declared ordering, not a
machine-dependent input-list order. ID lookup and index verification must agree.

It was frozen from allocation 58159's completed locked sample, not resampled or
selected using outcomes. Source sample/order/selection hashes are retained. The
resource contains ID/index/hash metadata only, no dialogue bodies or outcomes;
fitting and development never consume it. Test bodies still require selection.lock.

All 500-scenario callers use this resource automatically. Altered resource bytes,
missing/additional population IDs, mismatched indices or changed selected scenario
content fail closed. Never silently resample, replace missing cases, accept an
alternate dataset release or mix artifact subsets. The historical 2000/full-test
paths retain their original behavior. Previously completed locks, test manifests
and reports are unchanged; a new scientific run must use a newly locked current
code contract rather than mutating the old lock to resume under this code.
