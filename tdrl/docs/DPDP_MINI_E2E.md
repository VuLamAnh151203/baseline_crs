# DPDP mini end-to-end

This is a bounded **non-scientific integration run**. It verifies the real DPDP-adapted path on
one deterministic train scenario and one deterministic development scenario per dataset. It never
opens official test data and it does not replace development selection or the 500-scenario run.

Covered stages:

1. deterministic train/dev subsetting and transition preparation;
2. local-Qwen reward annotation;
3. one-epoch RoBERTa actor-critic offline training;
4. one-case MCTS-guided self-play update;
5. eight one-turn development dialogues: two datasets times `profile/resistant` times
   mini `natural/matched` search budgets.

Self-play writes `selfplay-trace.jsonl` with the selected responses/actions, critic votes,
deterministic acceptance result, disagreement flag, terminal reasons, root statistics, complete
MCTS tree and discounted return. A majority `accepted` critic vote terminates that simulated
trajectory even when the deterministic classifier disagrees, preventing repeated accepted rewards.

Every server compute report distinguishes the largest single-worker PyTorch allocator peak from
the whole visible-GPU `nvidia-smi` peak (`gpu_peak_used_bytes`), total memory and observed headroom.

The mini natural search uses three MCTS simulations at depth two. The mini matched search uses two
simulations at depth one. Both force the System-2 integration branch. These are integration-test
budgets, not selected scientific budgets.

Submit from the `tdrl` project root:

```bash
mkdir -p logs
export DPDP_ROBERTA_REVISION=722cf37b1afa9454edce342e7895e588b6ff1d59
sbatch --parsable scripts/slurm/dpdp_mini_e2e.sbatch
```

For job ID `JOB_ID`, monitor with:

```bash
squeue -j JOB_ID
tail -f logs/dpdp-mini-e2e-JOB_ID.out
tail -f logs/dpdp-mini-e2e-JOB_ID.err
```

Completion requires all of the following:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
jq '{status, scope, evaluation_dialogues}' \
  /mnt/data/u26anhvl/crs/dpdp/dpdp-mini-e2e-JOB_ID/mini-e2e-summary.json
jq '{status, dialogues, matrix}' \
  /mnt/data/u26anhvl/crs/dpdp/dpdp-mini-e2e-JOB_ID/evaluation/mini-evaluation-report.json
```

Expected top-level status is `mini_non_scientific`, `evaluation_dialogues` is `8`, and every matrix
row has `status: complete` with `system2` in `selected_systems`. Any exception also writes
`mini-e2e-failure.json` before the Slurm step exits non-zero.

The job uses this physical storage layout and creates every missing directory with `mkdir -p`:

```text
/mnt/data/u26anhvl/crs/
`-- dpdp/
    `-- dpdp-mini-e2e-JOB_ID/
        |-- offline/{TopDial,DuRecDial2.0}/checkpoint-final.pt
        |-- selfplay/{TopDial,DuRecDial2.0}/{checkpoint-final.pt,selfplay-trace.jsonl}
        `-- evaluation/
```

Evaluation receives this same absolute run directory and loads the final self-play checkpoints
from it; it does not resolve checkpoints through the repository `artifacts` symlink.
