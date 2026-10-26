# EPL-native-Qwen

Untouched EPL author code: [`upstream/`](upstream/) at commit
`364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda`. Qwen integration remains under `tdrl/`.

Source-faithful EPL policy/search/memory configuration with the original method's intrinsic
hyperparameters and shared Qwen only at the common language interfaces. This is separate from the
task-corrected T-EPL-v2 main-table variant.

```bash
bash scripts/submit_baseline.sh epl_native_qwen mini
export EPL_NATIVE_MINI_ROOT=/absolute/path/to/completed-mini-run
bash scripts/submit_baseline.sh epl_native_qwen full400
```

The full job requires a passed mini root because it verifies the exact native asset/config
contract before expensive evaluation. See `tdrl/docs/EPL_NATIVE_CORE_QWEN.md`.
