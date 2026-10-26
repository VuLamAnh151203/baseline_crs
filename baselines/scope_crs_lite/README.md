# SCOPE-CRS-lite

Untouched SCOPE author code: [`upstream/`](upstream/) at commit
`b35dbc64808afce25e8f8e5d6838ecf271ec4d0b`. CRS-lite adaptation remains under `tdrl/`.

Task-adapted SCOPE that keeps the released four-seed transition ensemble and online semantic-space
MCTS while replacing task-specific response generation with shared Qwen and fitting a
dataset-specific goal reward from train/dev only. This is not SCOPE-native.

```bash
export SCOPE_ASSET_ROOT=/absolute/path/scope/assets
bash scripts/submit_baseline.sh scope_crs_lite mini
bash scripts/submit_baseline.sh scope_crs_lite full400
```

The asset directory must contain the hash-audited transition ensemble and required Llama Guard
snapshot. See `tdrl/docs/SCOPE_CRS_LITE.md` and `tdrl/configs/scope-reference.lock.json`.
