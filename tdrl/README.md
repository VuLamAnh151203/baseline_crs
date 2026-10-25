# TDRL (temporary placeholder)

TDRL currently means **Temporal Distance Reinforcement Learning**, not the final method name.
This is a new implementation under `./tdrl`; the parent `../AGENTS.md` applies.

Implemented: typed inference/planner contracts; observed-history serialization; normalized
4096-D loopback embedding adapter; final-only non-thinking generation adapter; MRNFixed and
IQE-maxmean heads; detached target-network Bellman learning and negative separation; directed
graph/retrieval; conditional recovery and routing controls; conservative refusal guard;
provenance journal; immutable selection lock; paired bootstrap; budget/matrix tooling.

A gated three-method runner now covers PPDPP, TDRL-R and TDRL Full. It has CPU control
verification but has **not run on H100**. Official train/dev adapters and audits exist;
scientific checkpoints, selection, test and human evidence remain pending.
See [comparison command and limits](docs/COMPARISON.md), [implementation status](docs/STATUS.md)
and [one-allocation experiment design](docs/EXPERIMENT_PLAN.md).

## Local installation and verification

```bash
cd /media/minhduck/SSD1/Documents/projects/crs_refactor/tdrl
uv venv --system-site-packages .venv --python 3.11
uv pip install --python .venv/bin/python --no-deps -e . ruff==0.16.10
bash scripts/check.sh
.venv/bin/tdrl matrix --output artifacts/proposed-matrix.json
```

The installed local environment reuses PyTorch/NumPy/pytest for CPU control tests. It is not
H100 evidence. For a clean CPU dependency install use `uv sync --locked --extra dev`.
The separate runtime lock uses the PyTorch required by vLLM; both environments must have their
actual versions and lock hash recorded. Do not copy the CPU venv to H100.

## H100 bootstrap and submission

Do bootstrap on the SSH H100 login host before an experiment allocation. Downloads and package
installation are not inference. Copy the **new project only** to a new remote directory;
never import historical outputs or attach to another allocation.

The code checkout remains separate from large data and weights. The remote checkout root defaults
to your private directory `/home/u26anhvl/crs`, with this project under `/home/u26anhvl/crs/tdrl`.
You can override the checkout root with `TDRL_REMOTE_ROOT`. `scripts/link_h100_storage.sh` links the project's
`data/raw`, `data/processed`, and `models` directories to `/mnt/data/duclm3/crs_refactor/tdrl/data/raw`,
`/mnt/data/duclm3/crs_refactor/tdrl/data/processed`, and `/mnt/data/duclm3/crs_refactor/tdrl/models`.
Tracked files such as `data/manifests` remain beside the code. The helper refuses to replace
non-empty directories or links pointing elsewhere, so existing data/weights must be migrated and
verified explicitly before changing those paths. Both submit scripts run this setup automatically.

For automated sync/bootstrap/submission, the default above is used; to override it, set the
checkout root on the machine where this checkout lives:

```bash
export TDRL_REMOTE_ROOT=/your/private/path/crs
bash scripts/submit_comparison.sh
```

For manual bootstrap, after syncing the code, run the following on H100:

```bash
cd /home/u26anhvl/crs/tdrl
bash scripts/link_h100_storage.sh
~/.local/bin/uv sync --project runtime --locked --python 3.11
runtime/.venv/bin/python scripts/bootstrap_models.py
runtime/.venv/bin/python scripts/bootstrap_models.py --lock configs/policy-model.lock.json
runtime/.venv/bin/tdrl verify-models
runtime/.venv/bin/tdrl runtime-audit
mkdir -p logs
# Inspect squeue/sinfo immediately before this command.
sbatch --parsable scripts/slurm/allocation.sbatch
```

The batch requests one full H100 80 GB and at most 20 days. Its controller stops at any
failed gate and starts learning only after a complete real runtime smoke. No allocation
was submitted during the initial implementation. Live submission status and measured ETA
belong in `artifacts/` and the scheduler. `scripts/submit_comparison.sh` performs bootstrap
and submits from a clean commit.

The user explicitly permitted mandatory `openai`/`anthropic` dependencies inside the local
vLLM server on 2026-10-03. Application code imports neither SDK; its transport rejects all
non-loopback hosts and redirects and does not read inference credentials.

## Available stage commands

| Stage | Command / interface | Current limit |
|---|---|---|
| Static/control validation | `bash scripts/check.sh` | CPU evidence only |
| Model cache bootstrap | `runtime/.venv/bin/python scripts/bootstrap_models.py` | Download only |
| Weight verification | `tdrl verify-models` | Requires local model files |
| Source preprocessing | `tdrl prepare --input INPUT --output OUT --dataset TopDial --split train --source-manifest SOURCE.json` | Canonical inputs; official release adapters in comparison runner |
| Full runtime smoke assertions | `tdrl smoke --output artifacts/SMOKE` | H100/Slurm only; no real evidence yet |
| Proposed evaluation matrix | `tdrl matrix --output artifacts/proposed-matrix.json` | Split topology requires audit |
| Timing feasibility | `tdrl budget --measurements MEASURED.json --remaining-hours HOURS` | Requires actual counts/p95 timings |
| Artifact schema validation | `tdrl validate-manifest RUN/manifest.json` | Does not certify scientific validity |
| Reachability optimization | `tdrl comparison --output artifacts/RUN` | Sequential gated three-method scope |
| Selection | `write_selection(...)` / `verify_selection(...)` | Requires all evidence, including baselines |

The model locks pin immutable revisions and upstream weight hashes, but are explicitly marked
`metadata_pinned`; local byte verification is required before inference.
