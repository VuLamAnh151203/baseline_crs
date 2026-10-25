# EnPL-domain3

Reproduction of EnPL with domain-general exact entity overlap replacing the paper's movie-field
overlap. This change is necessary for TopDial POI and DuRecDial hobby/food goals and is reported as
an adapted variant, not as untouched EnPL. The plan bank is built only from the released ConvPlan
resource; there is no gradient-trained EnPL model.

Set the hash-verified archive before submission:

```bash
export ENPL_CONVPLAN_ZIP=/absolute/path/ConvPlan.zip
bash scripts/submit_baseline.sh enpl_domain3 mini
bash scripts/submit_baseline.sh enpl_domain3 full400
```

Implementation: `tdrl/src/tdrl/baselines/enpl/` and `tdrl/scripts/enpl/`. Full reproduction notes
and unresolved source ambiguities: `tdrl/docs/enpl-reproduction.md`.
