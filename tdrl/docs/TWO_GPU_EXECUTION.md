# Two-H100 execution amendment

User authorization 2026-10-08 replaces retired allocation 57711 with allocation
58159: two full H100 80GB GPUs, 16 CPUs total, 128G RAM and 20 days. Keep source
artifacts, 120 committed TopDial RL episodes, optimizer/RNG and fitting provenance.
Report source and new allocation segments separately. No claim that all fitting
ran in the new allocation.

Run a bounded dependency queue with at most two Slurm steps. Each step reserves
one exclusive GPU and eight CPUs; set four OpenMP threads. Each interactive
worker serves both pinned BF16 models on its own 80GB H100, using loopback port
pairs 18000/18001 and 18100/18101. Validate distinct GPU UUIDs and independent
dual-residency/controller/concurrency gates before any new update. No model,
precision, quantization, embedding dimension or inference placement fallback.

Overlap offline DuRecDial embedding/head learning with PPDPP fitting. Execute
development head/threshold shards on a free GPU once both datasets' heads exist,
while baseline fitting may continue on the other GPU. Each shard retains all
trials and candidates. Aggregate both datasets with the original macro-SR,
paired-CI and compute tie-break rules; write one immutable selection lock only
after fitting and PPDPP development diagnostics complete.

Test GPU workers independently verify that lock and sample the same 500 total
official scenarios with seed 42. Experimental seed stays 17, with profile and
resistant protocols. One worker evaluates PPDPP, TDRL-R and matched TDRL; the
other evaluates natural TDRL Full concurrently. Assert identical membership,
merge all 4000 records and retain each worker's manifests. Sum per-GPU H100 time;
do not treat overlapping two-GPU wall time as one H100's compute. Hardware
utilization depends on work and is measured, not guaranteed to be 100%.

The queue reserves both GPUs for useful independent tasks whenever dependencies
permit, stops siblings at durable boundaries on failure and retains the persistent
allocation. Completion of this scope does not finish full baseline/human/paper gates.

```sh
python3 scripts/run_two_gpu.py --job-id 58159 \
  --source artifacts/ppdpp-tdrl-concurrent-v2-57711 \
  --output artifacts/two-gpu-58159
```
