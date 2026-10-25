# DuRecDial PPDPP RL interruption audit

Audit date: 2026-10-05. Source allocation: `57166`, source commit `9285587`.
This is a training interruption audit, not an evaluation result.

## Verified state

The DuRecDial checkpoint has `schema_version=1`, `next_episode=243`, 393 optimizer
state entries, CPU RNG, one CUDA RNG state, and provenance exactly matching its
stage contract. Episodes 0 through 242 have committed updates. Of 256 episodes,
13 remain, including episode index 243 (ordinal 244).

Episode `000243/dialogue.json` is partial with four completed turns. Its
`pre-episode-rng.pt` remains available. The unresolved request is
`PPDPP-RL/PPDPP/resistant/260/train:en:000243/turn/4/live`, an embedding request before
the next policy action. Its marker remains in the embedding request ledger.
No update for episode 243 has been committed.

Evidence: `artifacts/debug-57166-DuRecDial/inspection.json` contains the checkpoint,
partial dialogue, RNG and pending-request hashes. Original remote evidence is under
`/home/duclm3/projects/crs_refactor/tdrl/artifacts/allocation-57166/PPDPP-RL/DuRecDial2.0/`.

## Failure and mitigation

The observed exception is `httpx.RemoteProtocolError: Server disconnected without
sending a response`. Available server logs contain no corresponding CUDA/OOM
traceback. Both API servers log SIGTERM shutdown during controller cleanup; those
shutdown messages alone do not establish that a server crash caused the HTTP error.
A stale HTTP keep-alive connection is a hypothesis, not a proven root cause.

The local transport now disables persistent HTTP connections for generation and
embedding, preserving loopback-only routing, decoding, seeds and model precision.
There are no automatic retries. A regression uses a real local HTTP/1.1 server to
check distinct TCP connections and a forced disconnection; the failed request stays
unresolved and a repeated request ID remains blocked. Runtime exceptions now capture
owned API process return codes before cleanup sends termination signals. This
mitigation has CPU/HTTP control evidence; post-change H100 validation is pending.

## Recovery boundary

Algorithmically, retraining the 243 committed episodes is unnecessary. The existing
resume path restores policy, optimizer and RNG, reconstructs saved sampled actions
and their gradients, and checks every strategy before continuing the partial episode.
Its regression checks exact log probabilities and gradients under dropout.

A direct rerun is deliberately blocked: the embedding outcome is unresolved, and
the transport patch changes the git commit embedded in stage/checkpoint provenance.
Do not delete the pending marker, overwrite old contracts, bypass provenance checks,
or silently retry the request. A recovery must retain the failed attempt and its
unknown request outcome, document the narrowly reviewed source amendment, verify
all unchanged model/data/config/prompt/optimizer/seed contracts, and validate replay
on H100 before applying another policy update.

Allocation 57166 has ended. Recovering its artifacts in another allocation spans
allocations and requires an explicit amendment to the repository's one-allocation
protocol. The alternatives are an audited recovery of the 13 remaining episodes,
or a fresh scientific pipeline entirely inside one new allocation. The user authorized audited recovery on 2026-10-05. Implementation and launch status
are documented below; authorization is not completion evidence. Completed banks/head/SFT/TopDial RL
artifacts remain preserved; selection and test have not started.


## Authorized recovery command

`bash scripts/submit_recovery.sh` transfers only committed tracked source. It submits
`scripts/slurm/recovery.sbatch` with one full H100 and a 20-day maximum. The CLI is:

```bash
runtime/.venv/bin/python -m tdrl recover-comparison \
  --source artifacts/allocation-57166 \
  --audit artifacts/debug-57166-DuRecDial/inspection-replay-fix.json \
  --output artifacts/recovery-$SLURM_JOB_ID
```

Original completed stages are hash-verified and referenced without rewriting their
contracts or manifests. Recovery copies the DuRecDial checkpoint, episode snapshots,
RNG and generation journal into a new output; its new inference ledger starts empty.
The original unresolved embedding marker stays intact. Only a matching audited
request ID and payload may be reissued. Both allocation IDs, source hashes, code diff,
imported run IDs and the unknown prior request outcome remain in
`recovery-amendment.json`. New stage counters cover the recovery segment only.

Before any new optimizer update, the allocated H100 reconstructs all four saved
actions twice, verifies their recorded strategies, compares exact log probabilities
and gradient hashes, then restores pre-episode RNG and clears audit gradients.
`h100-replay-audit.json` records this check. Old gradients were never saved, so this
cannot establish numerical identity with an unrecorded pre-failure gradient.

The compatibility provenance in training snapshots remains the original frozen
training contract. New checkpoints additionally record `execution_amendment` with
actual source commit and allocation. `recovery-completion.json` requires 256 updates
and records 243 preserved plus 13 new updates. Selection/test remain gated and their
lock carries the recovery amendment. Any further recovery requires a fresh audit;
this command fails closed if its destination is already used.


## Replay audit failure and fix continuation (2026-10-06)

Allocation `57561` completed Gates 0/1/2 but failed at repeated gradient replay,
before a new update or inference request in the RL recovery stage. Its checkpoint
SHA-256 still equals `4e35e7aa62867bfa6713c6c7f9b0ed86b9bc992c91aa4a75c9690619bd216d70`.
The original audit did not retain separate log-probability and per-parameter gradient
differences on failure; CUDA backward nondeterminism remains a hypothesis until the
new allocated diagnostic runs. Original allocation `57166` and failed recovery
`57561` are both retained and hash-verified by `inspection-replay-fix.json`.

The fix records two default-backward replays and two strict-backward replays,
including exact log-probabilities, RNG hashes, gradient hashes, parameter names,
changed-element counts and maximum absolute differences. Failed audits also retain
these diagnostics. There is no tolerance-based acceptance. Strict replay must be
bitwise identical, match stored strategies, and leave forward values/RNG identical
to the default-backward experiment. Audit gradients are discarded and pre-episode
RNG restored before normal continuation. No optimizer update occurs in the audit.

The explicit numerical execution amendment is controller-only
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True,
warn_only=False)` scoped to PPDPP backward. Both deterministic flags are restored
on exit/exception. New checkpoints record this amendment and the replay-audit hash.
Owned vLLM processes strip the controller-only flags, keeping inference settings.
All model IDs/weights, BF16 precision, policy architecture/objective, optimizer,
seeds, dialogue serialization and prompts remain frozen. This is numerical
stabilization of the remaining updates, not proof of identity with unsaved gradients
from allocation `57166`.

PyTorch 2.13 documents deterministic versus nondeterministic CUDA SDPA backward:
https://github.com/pytorch/pytorch/blob/v2.13.0/docs/source/notes/randomness.md

The canonical recovery command now uses
`--audit artifacts/debug-57166-DuRecDial/inspection-replay-fix.json`. Prior output
directories cannot be overwritten. Gate 2 and the strengthened replay audit must
pass in the replacement allocation before the 13 remaining episodes continue.
H100 validation of this fix is pending at implementation time.


## Persistent allocation request (2026-10-06)

The user replaced pending request `57710` (cancelled before start) with one persistent
allocation: one full H100 80 GB, 16 Slurm CPUs, `--mem=128G`, and
`--time=20-00:00:00`. `scripts/submit_recovery.sh` records its ID in
`logs/active-allocation-job-id.txt`. The audited pipeline runs first. Its exit code is
recorded separately in `artifacts/recovery-JOBID/allocation-holder.json`; scientific
failure remains failure even while the allocation stays RUNNING for debugging.
The holder keeps its allocation until explicit release or the pre-deadline signal.

Run subsequent authorized jobs from this checkout with:

```bash
python3 tdrl/scripts/run_in_allocation.py -- \
  runtime/.venv/bin/python -m tdrl smoke --output artifacts/followup-smoke-JOBID
```

The helper checks the registered allocation's owner, RUNNING state, full H100 type,
CPU/RAM/time contract and committed source. It uses `srun --jobid=JOBID` with all
16 CPUs and the one GPU, without `--overlap` or another `sbatch`. Thus later GPU
steps wait for prior steps to finish. Each helper invocation retains its command,
submitted source commit and exit code under `allocation-steps`. Stage manifests
still govern actual scientific provenance, gates, errors and compute. GPU idle
holding time is allocation consumption, not completed experimental compute.


## Selection request namespace fix (2026-10-06)

The first selection attempt failed before lock/test with `request ID reused with
different inputs`. Its legacy prefix omitted head, threshold and budget contracts.
Different development trials could therefore read each other's generation records;
the shared journal correctly rejected a changed payload. All old selection traces,
cache and metrics are invalidated and retained for audit, including the 16 completed
and one partial dialogue. Fitting remains complete and unaffected.

Evaluation now supplies a namespace digest of the full dialogue contract, config
hash and matched-budget setting. Both embedding/generation and rollout requests
inherit this scope. Decoding seeds still depend on the unchanged scenario/protocol/
turn/role formulas. Identical-contract replay retains its original request IDs;
changed head, thresholds, budget or config receives a distinct scope. Direct RL
requests retain their original IDs. Payload mismatch within a namespace still fails
closed; journal checks have not been weakened.

`continue-selection` verifies all imported fitting artifact hashes, frozen contracts,
the original recovery amendment and the failed selection manifest. It references
completed banks/heads/SFT/RL read-only, keeps the original allocation clock and reruns
Gates 0/1/2 followed by selection/test/report with fresh output/cache:

```bash
python3 tdrl/scripts/run_in_allocation.py -- \
  runtime/.venv/bin/python -m tdrl continue-selection \
  --source artifacts/recovery-57711 --output artifacts/selection-fix-57711
```

The current-code H100 smoke includes two changed prompts under distinct trial
namespaces, with identical scenario/seed, and two exact cache replays. It requires
exactly two actual generation executions. Scientific artifacts record
`selection-amendment.json`; original failing artifacts and checkpoints are not
rewritten. CPU tests cover scope separation, seed pairing, exact resume and retained
payload-mismatch rejection. H100 validation remains pending at implementation time.
