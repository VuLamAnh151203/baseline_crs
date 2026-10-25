#!/usr/bin/env python3
"""Generate paired development diagnostics directly from immutable run records."""

import json
import sys
from pathlib import Path

from tdrl.metrics import aggregate, paired_bootstrap
from tdrl.provenance import atomic_json, file_hash

output = Path(sys.argv[1])
mrn_split = len(sys.argv) > 2 and sys.argv[2] == "--mrn-split"
mrn_only = mrn_split or (len(sys.argv) > 2 and sys.argv[2] == "--mrn-only")
numeric = mrn_only or (len(sys.argv) > 2 and sys.argv[2] == "--numeric-baseline")
slots = [0] if mrn_only and not mrn_split else [0, 1]
if numeric:
    rows = [x for i in slots for x in json.loads((output / f"eval-{i}/records.json").read_text())]
    baseline_path = output.parent / "topdial-debug-v2-58159/eval-0/records.json"
    baseline_manifest = json.loads((baseline_path.parent / "manifest.json").read_text())
    if (
        baseline_manifest["status"] != "complete"
        or file_hash(baseline_path) != baseline_manifest["outputs"]["records.json"]
    ):
        raise ValueError("fixed PPDPP baseline incomplete or changed")
    baseline_provenance = baseline_manifest["provenance"]
    for slot in slots:
        directory = output / f"eval-{slot}"
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["status"] != "complete"
            or file_hash(directory / "records.json") != manifest["outputs"]["records.json"]
        ):
            raise ValueError("new numerical evaluation incomplete or changed")
        provenance = manifest["provenance"]
        for key in ["models", "dataset", "seeds", "prompts", "preprocessing"]:
            if provenance[key] != baseline_provenance[key]:
                raise ValueError(f"fixed PPDPP shared provenance changed: {key}")
        if provenance["checkpoint"]["PPDPP"] != baseline_provenance["checkpoint"]["PPDPP"]:
            raise ValueError("fixed PPDPP checkpoint changed")
        if provenance["scenarios"]["ids"] != baseline_provenance["scenarios"]["ids"]:
            raise ValueError("fixed PPDPP scenario order changed")
    rows.extend(x for x in json.loads(baseline_path.read_text()) if x["variant"] == "PPDPP")
    source_hashes = {
        f"eval-{i}/records.json": file_hash(output / f"eval-{i}/records.json") for i in slots
    }
    source_hashes["fixed_PPDPP"] = file_hash(baseline_path)
elif len(sys.argv) > 2 and sys.argv[2] != "--standard":
    iqe_output = Path(sys.argv[2])
    rows = json.loads((output / "eval-0/records.json").read_text())
    mrn_manifest = json.loads((output / "eval-1/manifest.json").read_text())
    for variant in ["repair-MRNFixed-R", "repair-MRNFixed-Full"]:
        for protocol in ["profile", "resistant"]:
            paths = sorted(
                (output / "eval-1/dialogues" / variant / protocol).glob("*/dialogue.json")
            )
            if len(paths) != 64:
                raise ValueError("incomplete retained MRN cell")
            for path in paths:
                relative = str(path.relative_to(output / "eval-1"))
                if file_hash(path) != mrn_manifest["outputs"][relative]:
                    raise ValueError("retained MRN dialogue checksum mismatch")
                rows.append(
                    {
                        "variant": variant,
                        "protocol": protocol,
                        "record": json.loads(path.read_text()),
                    }
                )
    rows.extend(json.loads((iqe_output / "records.json").read_text()))
    source_hashes = {
        "eval-0/records.json": file_hash(output / "eval-0/records.json"),
        "retained-MRN-manifest": file_hash(output / "eval-1/manifest.json"),
        "fresh-IQE-records": file_hash(iqe_output / "records.json"),
    }
else:
    rows = [
        x for slot in [0, 1] for x in json.loads((output / f"eval-{slot}/records.json").read_text())
    ]
    source_hashes = {
        f"eval-{i}/records.json": file_hash(output / f"eval-{i}/records.json") for i in [0, 1]
    }
if len(rows) != (384 if mrn_only else 640 if numeric else 896):
    raise ValueError("diagnostic count mismatch")
groups = {}
for x in rows:
    r = x["record"]
    if r["status"] != "complete" or r["infrastructure_errors"] or r["seed"] != 17:
        raise ValueError("incomplete/invalid diagnostic")
    key = x["variant"] + "/" + x["protocol"]
    by_id = groups.setdefault(key, {})
    if r["scenario_id"] in by_id:
        raise ValueError("duplicate scenario")
    by_id[r["scenario_id"]] = r
metrics = {k: aggregate(list(v.values())) for k, v in groups.items()}
if mrn_only and set(groups) != {
    variant + "/" + protocol
    for variant in ["PPDPP", "repair-MRNFixed-R", "repair-MRNFixed-Full"]
    for protocol in ["profile", "resistant"]
}:
    raise ValueError("MRN-only report requires exactly the frozen paired MRN/PPDPP cells")
comparisons = {}
for key, items in groups.items():
    baseline = groups["PPDPP/" + key.rsplit("/", 1)[1]]
    if items.keys() != baseline.keys() or len(items) != 64:
        raise ValueError("unpaired scenarios")
    ids = sorted(items)
    for scenario_id in ids:
        for field in [
            "judge_protocol_sha256",
            "acceptance_rule_version",
            "turn_budget",
            "protocol",
        ]:
            if items[scenario_id][field] != baseline[scenario_id][field]:
                raise ValueError(f"paired outcome/simulator contract mismatch: {field}")
    comparisons[key] = paired_bootstrap(
        [items[i]["accepted"] for i in ids], [baseline[i]["accepted"] for i in ids]
    )
full_vs_retrieval = {}
for key, items in groups.items():
    if not key.split("/")[0].endswith("-Full"):
        continue
    retrieval = groups[key.replace("-Full/", "-R/")]
    ids = sorted(items)
    full_vs_retrieval[key] = {
        **paired_bootstrap(
            [items[i]["accepted"] for i in ids], [retrieval[i]["accepted"] for i in ids]
        ),
        "won": sum(items[i]["accepted"] and not retrieval[i]["accepted"] for i in ids),
        "lost": sum(retrieval[i]["accepted"] and not items[i]["accepted"] for i in ids),
    }
record = {
    "schema_version": 1,
    "scope": "post-test development-only planner diagnostic; not independent test confirmation",
    "experimental_seed": 17,
    "dialogs": len(rows),
    "source_sha256": source_hashes,
    "results": metrics,
    "paired_vs_PPDPP": comparisons,
    "paired_Full_vs_R": full_vs_retrieval,
    "head_scope": "MRNFixed only" if mrn_only else "MRNFixed and IQE",
}
atomic_json(output / "report.json", record)
lines = [
    "# TopDial development planner diagnostic",
    "",
    record["scope"],
    "",
    "| Variant/protocol | SR | Difference vs PPDPP | Paired 95% CI | H100 s/dialogue |",
    "|---|---:|---:|---|---:|",
]
for key, m in metrics.items():
    p = comparisons[key]
    lines.append(
        f"| {key} | {m['sr']:.4f} | {p['difference']:.4f} | [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] | {m['h100_seconds_per_dialogue']:.2f} |"
    )
lines.extend(
    [
        "",
        "| Full/protocol vs R | SR difference | Paired 95% CI | Won / lost |",
        "|---|---:|---|---:|",
    ]
)
for key, p in full_vs_retrieval.items():
    lines.append(
        f"| {key} | {p['difference']:.4f} | [{p['ci95'][0]:.4f}, {p['ci95'][1]:.4f}] | {p['won']} / {p['lost']} |"
    )
lines.extend(
    [
        "",
        "Source record paths and SHA-256 hashes: `report.json`. All values generated from these records.",
    ]
)
(output / "report.md").write_text("\n".join(lines) + "\n")
