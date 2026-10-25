# ProCoT

One-shot target-guided ProCoT using the shared local Qwen backbone. It preserves a single
thought-action-response planning call, current-topic to next-topic bridge planning and the pinned
upstream demonstration. It has no training stage under this evaluation.

```bash
bash scripts/submit_baseline.sh procot mini
bash scripts/submit_baseline.sh procot full400
```

Implementation: `tdrl/src/tdrl/baselines/procot/` and `tdrl/scripts/procot/`. Source prompt/code
hashes and adaptations are in `tdrl/configs/procot-reference.lock.json`.
