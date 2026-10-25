# EPL-adapted fixed1000 evaluation

This adapter, isolated under EPL-specific source, config, script and artifact
namespaces in the main TDRL checkout, adapts pinned EPL revision
`364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda` to the shared TDRL evaluator. It runs
EPL only; it does not rerun TDRL.

The method keeps an EPL-style RoBERTa policy prior, experience retrieval and P-UCT.
The disclosed task adaptation uses the shared six-action taxonomy, builds experience
memory by self-simulating 50 official training scenarios per dataset, and replaces hosted generation,
simulation and embedding services with the pinned local Qwen models. The deadline
natural budget is five P-UCT simulations, depth five, top-four policy expansion and
top-20 memory retrieval. It is an `EPL-adapted` result, not an unmodified reproduction
of the paper's DuRecDial/INSPIRED scripts.

## Fixed evaluation contract

- `configs/baseline-fixed1000-test-v1.json` and its immutable resource SHA.
- Prefix 400 means 200 TopDial plus 200 DuRecDial scenarios.
- Seed 17, profile/resistant protocols, natural budget and turn limit 10.
- Shared Qwen3.5-9B realization/simulator/judge, Qwen3-Embedding-8B and BF16.
- Test bodies open only after train/dev selection and an immutable EPL selection lock.
- Dataset workers run concurrently on two H100s. Request IDs contain neither prefix
  count nor GPU slot, so completed prefix-400 snapshots are reusable by the extension.

## Storage links in the main checkout

```bash
bash scripts/link_h100_storage.sh \
  /mnt/data/u26anhvl/crs/data \
  /mnt/data/u26anhvl/crs/models
ln -s /mnt/data/u26anhvl/crs/toolchains toolchains
```

The links are ignored by Git. Scientific submission refuses a dirty worktree.

## Prefix 400 end to end

Run the diagnostic mini-E2E first. It uses two H100s for at most four hours and
keeps the real five-epoch RoBERTa training, Qwen self-simulation/embedding,
natural P-UCT budget, two protocols and turn limit. Only data volume is reduced:
four policy-training examples, two policy-development examples, one train-only
self-simulation case and one dialogue-development scenario per dataset. It then
runs exact-replay smoke checks and never opens the official test split:

```bash
bash scripts/submit_epl_mini.sh
```

Its run root is `/mnt/data/u26anhvl/crs/epl/epl-mini-e2e-JOB_ID`.

Submit from the committed `crs_www/tdrl` project root. The login node only invokes
`sbatch`; all Python/model work runs inside the allocation:

```bash
bash scripts/submit_epl400.sh
```

The job requests two H100s for six days and runs, in order:

1. parallel TopDial and DuRecDial policy fitting;
2. train-only experience-memory construction and Qwen embedding indexes;
3. parallel development evaluation;
4. immutable selection lock and measured feasibility gate;
5. fixed prefix-400 evaluation (800 EPL dialogues total).

The run root is printed on submission and saved to
`logs/active-epl-run-root.txt`. Stage logs live under `<run-root>/logs`, while Slurm
stdout/stderr use `logs/epl-e2e400-JOBID.{out,err}` in the project checkout.

## Optional extension to 1000

Do not submit this until prefix 400 is complete. Pass the exact original run root:

```bash
bash scripts/submit_epl1000.sh \
  /mnt/data/u26anhvl/crs/epl/epl-fixed1000-PREFIX400_JOB_ID
```

The extension refuses a different source commit or changed selection/checkpoint/memory.
It reuses completed dialogues and journals, then evaluates only the missing 300
scenarios per dataset. Prefix-400 reports remain separate and immutable.

## Monitoring

```bash
squeue -j JOB_ID -o '%.18i %.28j %.9T %.10M %.30R'
tail -f /mnt/data/u26anhvl/crs/epl/slurm/epl-e2e400-JOB_ID.out
cat /mnt/data/u26anhvl/crs/epl/epl-fixed1000-JOB_ID/status.json
tail -f /mnt/data/u26anhvl/crs/epl/epl-fixed1000-JOB_ID/events.jsonl
```

Queued/submitted work is not completed evaluation evidence. A valid result requires a
complete E2E summary, complete shard reports, sealed sample, selection lock and dialogue
checksum seals.
