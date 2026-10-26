"""Auditable construction of a training-only EnPL plan bank from ConvPlan."""

import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from ...provenance import atomic_json, canonical, digest, file_hash
from .consistency import consistency
from .entity_extractor import DeterministicTextRankExtractor
from .normalize import conversation_hash, extract_domain_entities
from .schema import PlanBankEntry, RawRecord

BANK_SCHEMA_VERSION = 2
CONVPLAN_MEMBER = "ConvPlan.json"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _scenario_map(root):
    mapping = defaultdict(list)
    for split in ("train", "dev"):
        path = Path(root) / "data/processed/DuRecDial2.0" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            scenario = json.loads(line)
            key = conversation_hash([item["content"] for item in scenario["messages"]])
            mapping[key].append(
                {
                    "source_split": split,
                    "source_id": scenario["id"],
                    "target": scenario["target"],
                }
            )
    return mapping


def _immutable_jsonl(path, rows):
    path = Path(path)
    content = b"".join(canonical(row) + b"\n" for row in rows)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"immutable EnPL artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.part")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def build_bank(root, convplan_zip, output, max_records=None, extractor=None):
    """Build a deterministic bank; partial banks are diagnostics and cannot score tests."""
    root, convplan_zip, output = Path(root), Path(convplan_zip), Path(output)
    zip_sha256 = file_hash(convplan_zip)
    with zipfile.ZipFile(convplan_zip) as archive:
        if archive.namelist() != [CONVPLAN_MEMBER]:
            raise ValueError("unexpected ConvPlan archive topology")
        raw_bytes = archive.read(CONVPLAN_MEMBER)
    member_sha256 = _sha256_bytes(raw_bytes)
    values = json.loads(raw_bytes)
    if not isinstance(values, list) or not values:
        raise ValueError("ConvPlan member must be a nonempty JSON list")
    if max_records is not None:
        if max_records < 1:
            raise ValueError("max_records must be positive")
        values = values[:max_records]
    complete_source = max_records is None
    extractor = extractor or DeterministicTextRankExtractor()
    prepared_train = root / "data/processed/DuRecDial2.0/train.jsonl"
    prepared_dev = root / "data/processed/DuRecDial2.0/dev.jsonl"
    build_config = {
        "schema_version": BANK_SCHEMA_VERSION,
        "source_zip_sha256": zip_sha256,
        "source_member_sha256": member_sha256,
        "mapping": "exact_normalized_conversation_to_prepared_train_dev_v1",
        "prepared_train_sha256": file_hash(prepared_train),
        "prepared_dev_sha256": file_hash(prepared_dev),
        "builder_sha256": file_hash(Path(__file__)),
        "eligible_splits": ["train"],
        "expected_candidates_per_record": 10,
        "keep_per_conversation": 2,
        "extractor": extractor.config,
        "score": "one_minus_token_levenshtein_over_max_length",
        "tie_break": "candidate_index",
        "target_resolution": "canonical_from_exact_source_mapping",
        "retrieval": {
            "domain": "frozen_target_type_mapping_v1",
            "entities": "same_domain_profile_preferences_plus_target_v1",
            "score": "exact_set_intersection_cardinality",
        },
    }
    build_hash = digest(build_config)
    source_map = _scenario_map(root)
    all_candidates = []
    bank = []
    source_rows = []
    record_ids = set()
    mapping_counts = Counter()
    invalid_counts = Counter()
    excluded_selected = Counter()
    plan_total = 0
    duplicate_plan_total = 0
    selected_before_provenance = 0
    records_with_selection_shortage = 0
    empty_conversations = 0
    empty_goals = 0
    empty_profiles = 0
    retrieval_domain_counts = Counter()
    empty_retrieval_entities = 0
    for value in values:
        record = RawRecord.parse(value)
        if record.conver_id in record_ids:
            raise ValueError("duplicate ConvPlan conver_id")
        record_ids.add(record.conver_id)
        empty_conversations += int(not any(item.strip() for item in record.conversation))
        empty_goals += int(not record.goal.strip())
        empty_profiles += int(not record.user_profile)
        if len(record.plans) != 10:
            raise ValueError("ConvPlan candidate count differs from the audited release")
        plan_total += len(record.plans)
        duplicate_plan_total += len(record.plans) - len(set(record.plans))
        conv_hash = conversation_hash(record.conversation)
        matches = source_map.get(conv_hash, [])
        if len(matches) == 1:
            match = matches[0]
            source_split = match["source_split"]
            source_id = match["source_id"]
            target_items = tuple(match["target"])
            target_resolution = "canonical"
            mapping_method = "exact_normalized_conversation"
        elif matches:
            source_split, source_id, target_items = "unknown", record.conver_id, ()
            target_resolution = "ambiguous"
            mapping_method = "ambiguous_exact_normalized_conversation"
        else:
            source_split, source_id, target_items = "unknown", record.conver_id, ()
            target_resolution = "unresolved"
            mapping_method = "unmatched"
        mapping_counts[source_split] += 1
        source_rows.append(
            {
                "released_conver_id": record.conver_id,
                "source_dataset": "DuRecDial2.0",
                "source_id": source_id,
                "source_split": source_split,
                "duplicate_group_id": conv_hash,
                "target_items": list(target_items),
                "target_resolution": target_resolution,
                "mapping_method": mapping_method,
            }
        )
        conversation_text = "\n".join(record.conversation)
        conversation_entities = tuple(
            item.canonical_text for item in extractor.extract(conversation_text)
        )
        ranked = []
        for index, plan in enumerate(record.plans):
            entities = tuple(item.canonical_text for item in extractor.extract(plan))
            scored = consistency(entities, conversation_entities)
            reason = scored.reason
            if not plan.strip():
                reason = "empty_plan"
                scored = type(scored)(0.0, False, reason)
            candidate = {
                "released_conver_id": record.conver_id,
                "candidate_index": index,
                "entry_id": digest([member_sha256, record.conver_id, index]),
                "source_split": source_split,
                "source_id": source_id,
                "conversation_hash": conv_hash,
                "entity_sequence": list(entities),
                "conversation_entity_sequence": list(conversation_entities),
                "consistency_score": scored.score,
                "valid": scored.valid,
                "reason": reason,
                "selected": False,
            }
            all_candidates.append(candidate)
            if scored.valid:
                ranked.append((scored.score, index, plan, entities, candidate))
            else:
                invalid_counts[reason] += 1
        ranked.sort(key=lambda item: (-item[0], item[1]))
        records_with_selection_shortage += int(len(ranked) < 2)
        for score, index, plan, entities, candidate in ranked[:2]:
            candidate["selected"] = True
            selected_before_provenance += 1
            if source_split != "train" or target_resolution != "canonical":
                excluded_selected[
                    f"source_{source_split}_target_{target_resolution}"
                ] += 1
                continue
            retrieval_domain, retrieval_entities = extract_domain_entities(
                record.user_profile, target_items
            )
            retrieval_domain_counts[retrieval_domain] += 1
            empty_retrieval_entities += int(not retrieval_entities)
            entry = PlanBankEntry(
                entry_id=candidate["entry_id"],
                source_dataset="DuRecDial2.0",
                source_split=source_split,
                source_conversation_id=source_id,
                duplicate_group_id=conv_hash,
                candidate_index=index,
                plan_text=plan.strip(),
                user_profile=record.user_profile,
                goal_raw=record.goal,
                target_items=target_items,
                demonstration_target=target_items[1],
                target_resolution=target_resolution,
                retrieval_domain=retrieval_domain,
                retrieval_entities=retrieval_entities,
                conversation_hash=conv_hash,
                entity_sequence=entities,
                consistency_score=score,
                extraction_config_hash=extractor.config_hash,
                bank_build_config_hash=build_hash,
            )
            bank.append(entry.serialize())
    bank.sort(
        key=lambda item: (
            item["source_dataset"],
            item["source_conversation_id"],
            item["candidate_index"],
        )
    )
    source_rows.sort(key=lambda item: str(item["released_conver_id"]))
    all_candidates.sort(
        key=lambda item: (str(item["released_conver_id"]), item["candidate_index"])
    )
    _immutable_jsonl(output / "source-manifest.jsonl", source_rows)
    _immutable_jsonl(output / "candidate-scores.jsonl", all_candidates)
    _immutable_jsonl(output / "plan-bank.jsonl", bank)
    numeric_ids = [int(value) for value in record_ids if value.isdigit()]
    audit = {
        "schema_version": 1,
        "status": "complete" if complete_source else "diagnostic_partial",
        "source_zip_sha256": zip_sha256,
        "source_member_sha256": member_sha256,
        "records": len(values),
        "distinct_conver_ids": len(record_ids),
        "conver_id_range": [min(numeric_ids), max(numeric_ids)],
        "missing_ids_within_numeric_range": (
            max(numeric_ids) - min(numeric_ids) + 1 - len(set(numeric_ids))
        ),
        "records_with_empty_conversation": empty_conversations,
        "records_with_empty_goal": empty_goals,
        "records_with_empty_profile": empty_profiles,
        "candidate_plans": plan_total,
        "duplicate_candidate_strings_within_record": duplicate_plan_total,
        "mapping_counts": dict(sorted(mapping_counts.items())),
        "invalid_candidate_counts": dict(sorted(invalid_counts.items())),
        "records_with_fewer_than_two_valid_candidates": records_with_selection_shortage,
        "selected_candidates_before_provenance": selected_before_provenance,
        "excluded_selected_candidates": dict(sorted(excluded_selected.items())),
        "selected_verified_train_entries": len(bank),
        "retrieval_domain_counts": dict(sorted(retrieval_domain_counts.items())),
        "entries_with_empty_retrieval_entities": empty_retrieval_entities,
        "paper_reported_plan_count": 12147,
        "count_discrepancy_unresolved": len(bank) != 12147,
        "test_bodies_opened": False,
    }
    atomic_json(output / "audit.json", audit)
    manifest = {
        "schema_version": BANK_SCHEMA_VERSION,
        "status": "complete" if complete_source else "diagnostic_partial",
        "build_config": build_config,
        "build_config_sha256": build_hash,
        "source_zip_sha256": zip_sha256,
        "source_member_sha256": member_sha256,
        "bank_sha256": file_hash(output / "plan-bank.jsonl"),
        "source_manifest_sha256": file_hash(output / "source-manifest.jsonl"),
        "candidate_scores_sha256": file_hash(output / "candidate-scores.jsonl"),
        "audit_sha256": file_hash(output / "audit.json"),
        "entries": len(bank),
        "eligible_splits": ["train"],
        "prepared_train_sha256": build_config["prepared_train_sha256"],
        "prepared_dev_sha256": build_config["prepared_dev_sha256"],
        "builder_sha256": build_config["builder_sha256"],
        "test_bodies": "sealed",
    }
    atomic_json(output / "bank-manifest.json", manifest)
    return manifest


def load_bank(path, manifest_path, require_complete=True):
    path, manifest_path = Path(path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != BANK_SCHEMA_VERSION:
        raise ValueError("unknown EnPL bank manifest")
    if require_complete and manifest.get("status") != "complete":
        raise ValueError("scored EnPL evaluation requires the full verified bank")
    if manifest.get("bank_sha256") != file_hash(path):
        raise ValueError("EnPL bank hash mismatch")
    if manifest.get("eligible_splits") != ["train"] or manifest.get("test_bodies") != "sealed":
        raise ValueError("EnPL bank provenance is not training-only")
    entries = [
        PlanBankEntry.parse(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(entries) != manifest.get("entries") or not entries:
        raise ValueError("EnPL bank entry count mismatch")
    if any(entry.source_split != "train" for entry in entries):
        raise ValueError("EnPL bank contains a non-training entry")
    return entries, manifest
