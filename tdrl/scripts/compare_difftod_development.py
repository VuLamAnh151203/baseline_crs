#!/usr/bin/env python3
"""Generate an audited paired MRN R/Full versus fixed fitted DiffTOD discovery report."""

import argparse
import json
from pathlib import Path

from tdrl.confirmation_lock import verify_candidate
from tdrl.paired_development import compare_records
from tdrl.provenance import atomic_json, file_hash, validate_manifest


def verified_records(directory, filename, stage):
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    if manifest["status"] != "complete" or manifest["stage"] != stage:
        raise ValueError("complete development stage required")
    path = directory / filename
    if file_hash(path) != manifest["outputs"].get(filename):
        raise ValueError("development record checksum mismatch")
    if manifest["provenance"]["dataset"]["split"] != "development":
        raise ValueError("comparison must use development only")
    return json.loads(path.read_text()), manifest


def shared_provenance(left, right):
    a, b = left["provenance"], right["provenance"]
    for field in ["models", "preprocessing"]:
        if a[field] != b[field]:
            raise ValueError("shared provenance mismatch: " + field)
    for field in ["dataset", "processed_sha256"]:
        if a["dataset"][field] != b["dataset"][field]:
            raise ValueError("development population mismatch")
    for field in ["assistant_sha256", "embedding_instruction_sha256"]:
        if a["prompts"][field] != b["prompts"][field]:
            raise ValueError("shared prompt mismatch")
    if a["scenarios"]["ids"] != b["scenarios"]["ids"]:
        raise ValueError("scenario order mismatch")


def verify_confirmation_source(manifest, payload, lock_sha256, membership):
    p = manifest["provenance"]
    if (
        p["code"]["commit"] != payload["code_commit"]
        or p["packages"]["lock_sha256"] != payload["package_lock_sha256"]
        or p["scenarios"].get("confirmation_lock_sha256") != lock_sha256
        or p["scenarios"]["ids"] != membership["scenario_ids"]
        or p["dataset"]["processed_sha256"] != membership["population_sha256"]
    ):
        raise ValueError("confirmation source does not match frozen candidate/membership")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--difftod", type=Path, required=True)
    p.add_argument("--mrn", type=Path, nargs=2, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--confirmation-lock", type=Path)
    args = p.parse_args()
    candidate = (
        verify_candidate(args.confirmation_lock, Path.cwd()) if args.confirmation_lock else None
    )
    membership = (
        json.loads((Path.cwd() / candidate["files"]["confirmation_membership"]["path"]).read_text())
        if candidate
        else None
    )
    lock_sha = file_hash(args.confirmation_lock) if candidate else None
    baseline, m = verified_records(
        args.difftod,
        "results.json",
        "DiffTOD-development-confirmation" if candidate else "DiffTOD-development-evaluate",
    )
    if candidate:
        verify_confirmation_source(m, candidate, lock_sha, membership)
        if m["provenance"]["checkpoint"]["checkpoint_sha256"] != candidate["difftod_policy_sha256"]:
            raise ValueError("confirmation DiffTOD policy differs from frozen baseline")
    sources = [
        {
            "path": str(args.difftod),
            "manifest_sha256": file_hash(args.difftod / "manifest.json"),
            "records_sha256": file_hash(args.difftod / "results.json"),
        }
    ]
    groups = {}
    for directory in args.mrn:
        rows, manifest = verified_records(
            directory,
            "records.json",
            "TopDial-planner-confirmation" if candidate else "TopDial-planner-evaluate",
        )
        shared_provenance(manifest, m)
        if candidate:
            verify_confirmation_source(manifest, candidate, lock_sha, membership)
            if manifest["provenance"]["checkpoint"]["MRNFixed"] != candidate["head_sha256"]:
                raise ValueError("confirmation MRN head differs from frozen candidate")
            thresholds_path = directory / "thresholds.json"
            if file_hash(thresholds_path) != manifest["outputs"].get("thresholds.json"):
                raise ValueError("confirmation threshold checksum mismatch")
            thresholds = json.loads(thresholds_path.read_text())
            if (
                thresholds["legacy"]["MRNFixed"]["H_edge"] != candidate["thresholds"]["H_edge"]
                or thresholds["repaired_H_cover"]["MRNFixed"] != candidate["thresholds"]["H_cover"]
                or thresholds.get("frozen_from") != lock_sha
                or thresholds["calibration_ids"]
            ):
                raise ValueError("confirmation must reuse frozen thresholds without recalibration")
        sources.append(
            {
                "path": str(directory),
                "manifest_sha256": file_hash(directory / "manifest.json"),
                "records_sha256": file_hash(directory / "records.json"),
            }
        )
        for row in rows:
            if row["variant"] not in {"repair-MRNFixed-R", "repair-MRNFixed-Full"}:
                raise ValueError("unexpected planner variant")
            groups.setdefault(row["variant"], []).append(row["record"])
    if set(groups) != {"repair-MRNFixed-R", "repair-MRNFixed-Full"}:
        raise ValueError("both MRN retrieval and Full required")
    comparisons = {key: compare_records(value, baseline) for key, value in groups.items()}
    if candidate:
        for result in comparisons.values():
            result["scope_limits"] = (
                "Single-seed disjoint simulated development confirmation; not official-test or human evidence."
            )
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output / "report.json",
        {
            "schema_version": 1,
            "scope": "Confirmation64 paired disjoint development; fixed fitted DiffTOD; not official-test evidence"
            if candidate
            else "Discovery64 paired development; fixed fitted DiffTOD; not independent confirmation or official-test evidence",
            **(
                {
                    "confirmation_lock_sha256": lock_sha,
                    "both_methods_beat_fitted_difftod_with_positive_ci": all(
                        r["candidate_macro_ci_excludes_zero_in_positive_direction"]
                        for r in comparisons.values()
                    ),
                }
                if candidate
                else {}
            ),
            "sources": sources,
            "comparisons": comparisons,
        },
    )
    lines = [
        "# TopDial paired development comparison",
        "",
        "Single-seed interactive simulation. Disjoint development confirmation."
        if candidate
        else "Single-seed interactive simulation. Discovery evidence only.",
        "",
        "| Method/protocol | SR | Difference vs DiffTOD | Paired 95% CI | H100 seconds/dialogue |",
        "|---|---:|---:|---|---:|",
    ]
    for method, result in comparisons.items():
        for protocol, metrics in result["candidate_metrics"].items():
            paired = result["paired_by_protocol"][protocol]
            lines.append(
                f"| {method}/{protocol} | {metrics['sr']:.4f} | {paired['difference']:.4f} | [{paired['ci95'][0]:.4f}, {paired['ci95'][1]:.4f}] | {metrics['h100_seconds_per_dialogue']:.2f} |"
            )
    lines.extend(
        [
            "",
            "Artifact paths and checksums: `report.json`. Scenario clusters preserve both protocols for the macro-average bootstrap.",
        ]
    )
    (args.output / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
