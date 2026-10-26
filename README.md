# Baseline CRS

Reproducible local-model baselines for target-driven conversational recommendation on
TopDial and DuRecDial2.0. This repository contains the complete `crs_www/tdrl` implementation,
method-specific adapters, fixed scenario resources, shared evaluation code, resumable training
and evaluation controllers, tests, and Slurm launchers.

The packaged source snapshot is `crs_www@b2c515325f9a45a6590a8222d7c84918643607c5`;
hashes for the fixed resource, runtime lock and shared model lock are recorded in
`configs/repository_provenance.json`.

The primary fixed-400 evaluation uses the first 200 TopDial and first 200 DuRecDial2.0 scenario
IDs in `test_scenarios_1000_td500_dr500_v1.json`. Each method runs both the profile-conditioned
and resistant-user protocols with root seed 17, for 800 dialogues per method. Test membership is
committed and never sampled at runtime.

## Included methods

| Launcher name | Public name | Fitting before evaluation | One-command mini/full400 |
| --- | --- | --- | --- |
| `direct_llm_v2` | DirectLLM-v2 | None; inference-only control | Yes |
| `tepl_v2` | T-EPL-v2 | Train-only self-simulation, policy and experience memory | Yes |
| `enpl_domain3` | EnPL-domain3 | Train-only plan-bank preparation; no gradient model fit | Yes |
| `procot` | ProCoT | None; inference-time proactive chain-of-thought planning | Yes |
| `dpdp` | DPDP | Offline actor-critic then MCTS self-play | Yes |
| `scope_crs_lite` | SCOPE-CRS-lite | Dataset reward fit; frozen released transition ensemble | Yes |
| `epl_native_qwen` | EPL-native-Qwen | Native policy/memory pipeline | Yes; full requires a passed mini |
| staged | DiffTOD | Dataset-specific diffusion training and development selection | See its README |
| staged | TRIP | Dataset-specific bidirectional planner training and development selection | See its README |

The proposed TDRL implementation is retained under `tdrl/src/tdrl`, but this repository's
portable launcher is for baselines. A method is not evidence merely because its code or job was
submitted: use only complete reports with verified manifests.

## Clone and environment

```bash
git clone --recurse-submodules https://github.com/VuLamAnh151203/baseline_crs.git
cd baseline_crs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n baseline-crs -c conda-forge --override-channels python=3.11 pip
conda activate baseline-crs
python -m pip install uv==0.8.13
bash scripts/bootstrap_environment.sh "$CONDA_PREFIX/bin/python"
```

The locked H100 runtime is created at `tdrl/runtime/.venv`. Installation and downloads are
bootstrap operations; model inference belongs in a Slurm allocation.

## Data and models

Large datasets, model weights, checkpoints, request journals, and private traces are intentionally
not stored in Git. Create durable storage and link it into the checkout:

```bash
export BASELINE_STORAGE_ROOT=/absolute/scratch/path/baseline_crs
mkdir -p "$BASELINE_STORAGE_ROOT/data" "$BASELINE_STORAGE_ROOT/models"
bash tdrl/scripts/link_h100_storage.sh \
  "$BASELINE_STORAGE_ROOT/data" "$BASELINE_STORAGE_ROOT/models"
```

Populate `tdrl/data/raw` and `tdrl/data/processed` according to
[`tdrl/data/README.md`](tdrl/data/README.md). Bootstrap the immutable Qwen snapshots with
`tdrl/scripts/bootstrap_models.py` or copy already hash-verified snapshots into `tdrl/models`.
Method-specific assets are documented in [`docs/ASSETS.md`](docs/ASSETS.md).

The untouched author repositories are pinned as submodules under
`baselines/<method>/upstream/`. If an existing checkout was cloned without submodules, initialize
them with:

```bash
git submodule update --init --recursive
```

The runtime adapters do not modify or import these upstream checkouts; they are immutable
provenance/reference material. Adapted implementations and execution glue live separately under
`tdrl/`. `scripts/fetch_upstream_sources.sh` remains available for a separate audit checkout.

Copy `.env.example` to `.env`, replace all paths, and source it:

```bash
cp .env.example .env
$EDITOR .env
source .env
```

## Validate, smoke, then run fixed-400

The repository must be clean and committed before a scientific job. The validator performs no
model inference:

```bash
python scripts/verify_checkout.py
python scripts/verify_checkout.py --require-assets
```

Submit the smallest real pipeline first, then the fixed-400 run only after its mini report passes:

```bash
bash scripts/submit_baseline.sh direct_llm_v2 mini
bash scripts/submit_baseline.sh direct_llm_v2 full400
```

Replace `direct_llm_v2` with another launcher name from the table. Logs are written to
`$BASELINE_STORAGE_ROOT/slurm`; all stage artifacts, progress records, checkpoints and reports are
under `$BASELINE_STORAGE_ROOT/runs/METHOD/`.

DPDP resume example:

```bash
export BASELINE_RUN_ROOT=/absolute/scratch/path/baseline_crs/runs/dpdp/EXISTING_RUN
export BASELINE_RESUME=1
bash scripts/submit_baseline.sh dpdp full400
```

Do not set resume for a different method or a fresh output root. Completed stages and dialogue
snapshots are provenance-checked before reuse.

## Evaluation contracts

All main-table methods share the pinned local Qwen generation model, embedding model where the
method needs it, BF16 precision, simulator, semantic outcome judge, decoding contract, maximum
turn count, fixed test IDs/order, root seed, and profile/resistant protocols. Intrinsic policy,
retrieval, transition, diffusion or reward models remain method-specific.

The code also contains the CARP-compatible legacy evaluator used for the later evaluation pass;
see [`tdrl/docs/carp-legacy-evaluator.md`](tdrl/docs/carp-legacy-evaluator.md). Do not merge numbers
from the strict shared evaluator and the CARP-compatible evaluator into one table without naming
the protocol.

## Repository map

- `baselines/<method>/upstream/`: untouched author code pinned as a Git submodule.
- `tdrl/src/tdrl/baselines/`: canonical import location used by the runnable package.
- `tdrl/scripts/<method>/`: canonical training/evaluation controllers used by the launchers.
- `tdrl/configs/`: model, source and experiment locks.
- `tdrl/src/tdrl/resources/`: explicit fixed test membership.
- `baselines/<method>/README.md`: method-specific fidelity and run notes.
- `scripts/submit_baseline.sh`: portable Slurm submission entrypoint.
- `scripts/slurm/baseline_e2e.sbatch`: shared allocation payload.
- `docs/ASSETS.md`: non-Git assets and verification requirements.
- `docs/UPSTREAM_SOURCES.md`: pinned upstream provenance.

DirectLLM-v2 is a repository-defined control and therefore has no external upstream submodule.
Execute experiments through the root launchers, which use the adapted `tdrl/` package.
`scripts/verify_checkout.py` verifies the pinned upstream gitlinks independently from the adapted
runtime source.

## Important limitations

- Evaluation is single-seed local interactive simulation, not human deployment evidence.
- EnPL-domain3 and SCOPE-CRS-lite are explicitly task-adapted variants, not exact reproductions on
  the authors' original tasks.
- TopDial paths for TRIP are weakly supervised because the dataset does not expose native TRIP
  intermediate paths.
- Model/data licenses apply independently. This repository does not redistribute restricted
  corpora or large weights.
- No project-wide software license has been selected yet; publication users should add the
  intended license before advertising unrestricted reuse.
