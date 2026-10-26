# T-EPL-v2

Untouched EPL author code: [`upstream/`](upstream/) at commit
`364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda`. T-EPL-v2 adaptations remain under `tdrl/`.

Task-adapted experience-based planning. The end-to-end controller builds training-only
self-simulated experience, fits the intrinsic policy, constructs its memory and evaluates frozen
artifacts. Official test outcomes are not used for fitting or selection.

```bash
bash scripts/submit_baseline.sh tepl_v2 mini
bash scripts/submit_baseline.sh tepl_v2 full400
```

Implementation: `tdrl/src/tdrl/baselines/epl/` and `tdrl/scripts/epl/`. The source audit and
departures are pinned in `tdrl/configs/epl-reference.lock.json`; see
`tdrl/docs/EPL_SOURCE_CORRECTED_V2.md`.
