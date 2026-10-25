# External assets

Git contains code, configurations, explicit scenario IDs and small test fixtures. It does not
contain large model snapshots, official dataset bodies, fitted checkpoints, request journals or
private dialogue traces.

## Shared assets

`tdrl/configs/models.lock.json` pins:

- `Qwen/Qwen3.5-9B` for local generation, simulation and semantic judgment;
- `Qwen/Qwen3-Embedding-8B` for normalized 4096-dimensional embeddings.

Both run in BF16. Scientific jobs force Hugging Face and Transformers offline mode. Bootstrap or
copy the snapshots before submitting a job, then verify the recorded hashes.

Official TopDial and DuRecDial2.0 releases must be obtained under their own licenses. Normalize
them to the layouts documented in `tdrl/data/README.md`; never commit official test bodies.

## Method-specific assets

| Method | Asset | Verification source |
| --- | --- | --- |
| DPDP | `FacebookAI/roberta-large` | `tdrl/configs/dpdp.models.lock.json` |
| T-EPL-v2 / EPL-native-Qwen | policy/retrieval backbones prepared by bootstrap | `tdrl/configs/epl.models.lock.json` |
| EnPL-domain3 | `ConvPlan.zip` | `tdrl/configs/enpl-reference.lock.json` |
| SCOPE-CRS-lite | released four-seed deterministic transition ensemble and Llama Guard snapshot | `tdrl/configs/scope-reference.lock.json` |
| DiffTOD | SEDD checkpoint and tokenizer | `tdrl/configs/difftod.models.lock.json` |
| TRIP | BERT encoder/tokenizer | `tdrl/configs/trip.models.lock.json` |

Never replace a pinned model silently. If an upstream snapshot is unavailable, stop and document
the blocked gate; a substitute model defines a different experimental variant.

## Storage layout

Recommended layout:

```text
$BASELINE_STORAGE_ROOT/
  data/raw/
  data/processed/
  models/
  runs/<method>/<run-id>/
  slurm/
  scope/assets/                 # when SCOPE is enabled
  convplan/ConvPlan.zip         # when EnPL is enabled
```

Use `tdrl/scripts/link_h100_storage.sh` to link only data and models. Run outputs remain outside
the Git checkout so a pull cannot overwrite evidence.
