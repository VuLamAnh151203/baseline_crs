# Concurrent PPDPP / TDRL Full evaluation

User authorization: 2026-10-08. Keep 2000 pooled test scenarios (sample seed 42),
one experimental seed 17, both datasets/protocols, all existing budgets and gates.
At each test scenario, run PPDPP and natural-budget TDRL Full as two independent
dialogue workers. They share the one generation server and one embedding server
on allocation 57711's H100. Then run TDRL-R and matched-budget TDRL sequentially.
No additional model instance, Slurm allocation or concurrent scientific stage.

Each worker owns its planner, history, namespace and output. Request IDs filter
per-turn compute records; shared-list length slices alone would count another
worker's requests. Allocate busy wall time equally among active dialogue workers.
This is an accounting convention, not measured GPU kernel attribution. Keep
dialogue wall latency separate; server cold time and stage overhead are separate.
Do not sum overlapping dialogue wall times as total H100 time. Pair ETA uses
measured pair wall time plus the two remaining serial cells. Failed workers are
drained; preserve unknown requests and every completed record. Completed-pair
resume preserves counters, attribution and timing without reissuing requests.

GDN generation lacks batch invariance in the pinned runtime. Preserve all seeds,
prompts and decoding; record concurrent batching as an execution amendment.
Exact durable journal replay is required; serial/concurrent generated text is
not assumed equal, and speedup comparisons disclose dialogue turn counts.

The continuation verifies unchanged fitting modules and policy-input/forward
interfaces, hashes every imported complete stage, and retains their original
manifests/compute. Partial SFT retains its exact old checkpoint contract,
optimizer, scheduler and RNG; fresh manifests identify the continuation segment.
Fresh dual-model/controller/concurrency smoke precedes another update. Source
fitting is repository-local; no historical experimental evidence is imported.

The concurrency smoke uses two fixed TopDial development scenarios, training
MRNFixed guidance and the partial SFT policy, with covered/off-map probes. It
measures serial and concurrent PPDPP/Full, the remaining serial cells, verifies
counter isolation, exact completed-pair resume and repeated controller gradients.
These probes are runtime diagnostics, not scientific results or selection.
ETA is a small-sample proxy, with safety/reserve and the original allocation
clock; DuRecDial/resistant and selected-head timing require later verification.
Keep tests sealed until the new selection lock. If feasibility fails, retain
measurements and request an explicit decision.

```sh
runtime/.venv/bin/python -m tdrl parallel-comparison \
  --source artifacts/ppdpp-sample2000-seed17-57711 \
  --output artifacts/ppdpp-tdrl-concurrent-57711 \
  --config configs/comparison-judge-v2-sample2000-seed17.json
```
