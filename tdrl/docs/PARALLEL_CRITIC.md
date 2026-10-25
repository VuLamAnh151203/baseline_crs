# Concurrent PPDPP training critic

User authorization: 2026-10-09, "tối ưu chạy đồng thời các mẫu".

`TDRL_CRITIC_WORKERS=10` submits the existing ten sampled critic requests to the
same loopback vLLM server concurrently. The server retains its existing maximum
of four active generation sequences. Sample indices, derived seeds, request IDs,
prompts, temperature 1.1, validation and arithmetic mean are unchanged. Results
are assembled in index order; failures drain all submitted requests before raising.
No extra training runs or experimental seeds are introduced. Default execution
remains serial until the H100 amendment is explicitly enabled.

Batch-dependent generation can change sampled text and future policy updates.
This is an execution amendment, not serial/concurrent text equivalence. Completed
checkpoints and exact journal records are retained; existing scientific and RNG
contracts remain immutable and source/current commits are disclosed in resumed
stage manifests. Two independently assigned H100 smoke records, zero unresolved
requests and exact cached replay are required before a code-contract migration.

Smoke command (inside the existing allocation, one visible H100/eight CPUs):

```sh
runtime/.venv/bin/python scripts/smoke_critic_parallel.py --slot 0 --output artifacts/critic-smoke-0 --checkpoint <durable-checkpoint>
```

Repeat on slot 1 with its own ports/output. Each smoke reserves policy FP32
optimizer state, checks exact BF16 controller action/logprob/gradient replay,
benchmarks serial and concurrent critic calls with identical request seeds,
checks exact journal replay without additional inference, and obtains a live
4096-dimensional embedding with both Qwen models resident. Smoke timings are
engineering measurements, not success-rate evidence or full-pipeline ETA.

Activation additionally requires `TDRL_CRITIC_AMENDMENT` pointing to an immutable
JSON record with source/current commits and both passed smoke paths/hashes.
Preserve all failed queue logs, completed fitting/development artifacts, checkpoint
hashes, allocation 58159 clock and source/new execution-segment compute.
