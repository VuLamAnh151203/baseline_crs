# DirectLLM-v2: prompt-controlled no-planning baseline

DirectLLM-v2 is a separate baseline variant. It does not replace DirectLLM-v1 or
reinterpret any completed v1 result, including job 59949.

The method makes exactly one local Qwen assistant generation for every executed
dialogue turn. It has no training, checkpoint, retrieval, embedding, search,
rollout, or planner-produced guidance. Its final-response system prompt uses the
same byte-level scaffold as TDRL, with the guidance string set to empty.

Dataset-specific grounding follows the frozen TDRL candidates:

- TopDial includes the shared `target_fact_guidance` used by the selected v12
  candidate.
- DuRecDial2.0 does not include target-fact grounding, matching retained planner
  revision 1.

All other shared evaluation settings remain unchanged: Qwen3.5-9B, decoding,
simulator, semantic judge, ten-turn limit, natural budget, profile/resistant
protocols, root seed 17, and the immutable fixed-400 prefix (200 scenarios per
dataset). Prompt hashes are locked per dataset before official test evaluation.

Run the diagnostic mini pipeline first:

```bash
bash scripts/submit_direct_llm_v2_mini.sh
```

After the mini job passes, submit fixed-400 explicitly:

```bash
bash scripts/submit_direct_llm_v2_400.sh
```

Outputs are written to distinct roots under
`/mnt/data/u26anhvl/crs/direct_llm/direct-llm-v2-*`. The fixed-400 script is not
an extension of a v1 run root and cannot overwrite job 59949.
