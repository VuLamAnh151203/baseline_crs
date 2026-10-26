# DirectLLM-v2

This is a repository-defined control, so there is no external author repository. Its adapted
implementation remains under `tdrl/src/tdrl/baselines/direct_llm/`.

Inference-only lower-bound/control: the shared Qwen generator responds directly from the same
observable profile, knowledge, target and dialogue history available to the planning baselines.
It uses the shared assistant prompt rather than a bespoke stronger baseline prompt. There is no
training checkpoint.

```bash
bash scripts/submit_baseline.sh direct_llm_v2 mini
bash scripts/submit_baseline.sh direct_llm_v2 full400
```

Implementation: `tdrl/src/tdrl/baselines/direct_llm/` and
`tdrl/scripts/direct_llm/run_e2e.py`. Design details: `tdrl/docs/DIRECT_LLM_V2.md`.
