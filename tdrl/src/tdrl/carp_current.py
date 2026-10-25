"""Bridges native EnPL/ProCoT planners to the pinned CARP evaluation harness.

This module contains no simulator, moderator, stopping, or metric logic.  Those
remain owned by the external CARP checkout.  It only translates CARP's typed
scenario/history objects to the native TDRL baseline planner contract.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .backend import LocalBackend
from .baselines.enpl.bank import load_bank
from .baselines.enpl.config import load_enpl_config
from .baselines.enpl.planner import ENPL_POLICY_VERSION, EnPLPlanner
from .baselines.enpl.retrieval import DomainEntityOverlapRetriever
from .baselines.procot.planner import ProCoTPlanner
from .baselines.procot.prompts import PROCOT_POLICY_VERSION
from .contracts import Decoding, Message as NativeMessage, ModelProvenance
from .provenance import Journal, digest, file_hash
from .state import DialogueState

SUPPORTED_METHODS = ("enpl_domain3", "procot")
METHOD_LABELS = {"enpl_domain3": "EnPL-domain3", "procot": "ProCoT"}


def scenario_to_native(scenario: Any) -> dict[str, Any]:
    """Translate only fields exposed by CARP at inference time."""
    return {
        "id": str(scenario.id),
        "target": [str(scenario.target_action), str(scenario.target_item)],
        "profile": dict(scenario.user_profile),
        "personality": dict(scenario.user_personality),
        "knowledge": [list(item) for item in scenario.knowledge],
        "messages": [],
        "dataset_name": str(scenario.dataset_name),
        "domain": str(scenario.domain),
    }


def history_to_native(history: list[Any]) -> DialogueState:
    return DialogueState(
        tuple(NativeMessage(str(message.role), str(message.text)) for message in history)
    )


class NativePlannerCarpAdapter:
    """Thread-local CARP adapter around one immutable native planner setup."""

    def __init__(
        self,
        method: str,
        generation: LocalBackend,
        planner_factory: Callable[[dict[str, Any]], Any],
        *,
        root_seed: int,
        request_namespace: str,
        method_provenance: dict[str, Any],
        opening_adapter: Any,
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported CARP baseline method: {method}")
        self.name = METHOD_LABELS[method]
        self.method = method
        self.generation = generation
        self.planner_factory = planner_factory
        self.root_seed = int(root_seed)
        self.request_namespace = str(request_namespace)
        self.method_provenance = dict(method_provenance)
        self.opening_adapter = opening_adapter
        self._scenario_id: str | None = None
        self._turn_traces: list[dict[str, Any]] = []

    def _reset_for(self, scenario: Any) -> None:
        scenario_id = str(scenario.id)
        if scenario_id != self._scenario_id:
            self._scenario_id = scenario_id
            self._turn_traces = []

    def generate_reply(self, scenario: Any, history: list[Any]) -> str:
        self._reset_for(scenario)
        if not history:
            response = self.opening_adapter.generate_reply(scenario, history)
            self._turn_traces.append(
                {
                    "turn_index": 0,
                    "request_id": None,
                    "response_sha256": digest(response),
                    "replay_ids": [],
                    "plan_compute": {
                        "strategy": "carp-current-shared-opening",
                        "method_core_invoked": False,
                        "generation_requests": 1,
                    },
                    "generation": {
                        "requests": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "latency_s": 0.0,
                        "token_counts_available": False,
                    },
                }
            )
            return response
        native_scenario = scenario_to_native(scenario)
        planner = self.planner_factory(native_scenario)
        turn_index = sum(1 for message in history if str(message.role) == "assistant")
        planner.seed_context = f"{scenario.id}/carp-current/{turn_index}"
        request_id = (
            f"carp-current/{self.request_namespace}/{self.method}/"
            f"{scenario.dataset_name}/{scenario.id}/turn/{turn_index}"
        )
        before = len(self.generation.consumed_requests)
        plan = planner.act(
            history_to_native(history),
            [],
            self.root_seed,
            request_id,
        )
        requests = self.generation.consumed_requests[before:]
        self._turn_traces.append(
            {
                "turn_index": turn_index,
                "request_id": request_id,
                "response_sha256": digest(plan.response),
                "replay_ids": list(plan.replay_ids),
                "plan_compute": dict(plan.compute),
                "generation": {
                    "requests": len(requests),
                    "input_tokens": sum(item.input_tokens for item in requests),
                    "output_tokens": sum(item.output_tokens for item in requests),
                    "total_tokens": sum(item.total_tokens for item in requests),
                    "latency_s": sum(item.latency_s for item in requests),
                    "token_counts_available": True,
                },
            }
        )
        return plan.response

    def get_assistant_meta(self) -> dict[str, Any]:
        generation = [trace["generation"] for trace in self._turn_traces]
        native_generation = [row for row in generation if row["token_counts_available"]]
        return {
            "adapter": "native_planner_to_carp_current_v1",
            "method": self.name,
            "method_key": self.method,
            "root_seed": self.root_seed,
            "request_namespace": self.request_namespace,
            "coverage_available": False,
            "coverage_note": "EnPL/ProCoT do not expose CARP retrieval-defined OOD labels.",
            "method_provenance": self.method_provenance,
            "compute": {
                "shared_carp_opening_requests": len(generation) - len(native_generation),
                "native_method_generation_requests": sum(
                    row["requests"] for row in native_generation
                ),
                "native_method_input_tokens": sum(row["input_tokens"] for row in native_generation),
                "native_method_output_tokens": sum(row["output_tokens"] for row in native_generation),
                "native_method_total_tokens": sum(row["total_tokens"] for row in native_generation),
                "native_method_generation_latency_s": sum(
                    row["latency_s"] for row in native_generation
                ),
                "shared_opening_token_counts_available": False,
            },
            "turn_traces": list(self._turn_traces),
        }


class AdapterFactory:
    """Creates isolated clients for CARP worker threads with shared read-only data."""

    def __init__(
        self,
        *,
        project_root: Path,
        output_root: Path,
        method: str,
        model_provenance: ModelProvenance,
        generation_url: str,
        enpl_artifact: Path,
        root_seed: int,
        contract: dict[str, Any],
    ) -> None:
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"unsupported CARP baseline method: {method}")
        self.project_root = Path(project_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.method = method
        self.model_provenance = model_provenance
        self.generation_url = generation_url
        self.root_seed = int(root_seed)
        self.contract = dict(contract)
        self.request_namespace = digest(self.contract)
        self.journal = Journal(
            self.output_root / "assistant-request-journal",
            {
                "adapter": "native_planner_to_carp_current_v1",
                "method": method,
                "model": asdict(model_provenance),
                "contract": self.contract,
            },
        )
        self.ledger = self.output_root / "assistant-request-ledger"
        self.assistant_decoding = self._assistant_decoding()
        self.retriever = None
        self.enpl_config = None
        self.method_provenance = {
            "policy_version": (
                ENPL_POLICY_VERSION if method == "enpl_domain3" else PROCOT_POLICY_VERSION
            ),
            "assistant_decoding": asdict(self.assistant_decoding),
        }
        if method == "enpl_domain3":
            self._load_enpl(Path(enpl_artifact).resolve())

    def _assistant_decoding(self) -> Decoding:
        filename = {
            "enpl_domain3": "comparison.enpl.json",
            "procot": "comparison.procot.json",
        }[self.method]
        import json

        payload = json.loads((self.project_root / "configs" / filename).read_text())
        return Decoding(**payload["assistant_decoding"])

    def _load_enpl(self, artifact: Path) -> None:
        config_path = self.project_root / "configs/enpl.tdrl-adapted.json"
        entries, manifest = load_bank(
            artifact / "bank/plan-bank.jsonl",
            artifact / "bank/bank-manifest.json",
            require_complete=True,
        )
        self.enpl_config = load_enpl_config(config_path)
        self.retriever = DomainEntityOverlapRetriever(
            entries,
            self.enpl_config["retrieval"]["k"],
            self.enpl_config["retrieval"]["distinct_source_conversations"],
        )
        self.method_provenance.update(
            {
                "bank_sha256": file_hash(artifact / "bank/plan-bank.jsonl"),
                "bank_manifest_sha256": file_hash(artifact / "bank/bank-manifest.json"),
                "bank_status": manifest["status"],
                "method_config_sha256": file_hash(config_path),
                "plan_decoding": self.enpl_config["plan_decoding"],
            }
        )

    def _planner_factory(self, generation: LocalBackend) -> Callable[[dict[str, Any]], Any]:
        if self.method == "procot":
            return lambda scenario: ProCoTPlanner(
                generation,
                scenario,
                self.assistant_decoding,
            )
        if self.enpl_config is None or self.retriever is None:
            raise RuntimeError("EnPL factory is not initialized")
        return lambda scenario: EnPLPlanner(
            generation,
            scenario,
            self.retriever,
            Decoding(**self.enpl_config["plan_decoding"]),
            self.assistant_decoding,
            self.enpl_config["context"]["max_input_characters"],
        )

    def build(self, *, opening_adapter: Any) -> NativePlannerCarpAdapter:
        generation = LocalBackend(
            self.generation_url,
            self.model_provenance,
            "generation",
            self.journal,
            self.ledger,
            request_concurrency=1,
        )
        return NativePlannerCarpAdapter(
            self.method,
            generation,
            self._planner_factory(generation),
            root_seed=self.root_seed,
            request_namespace=self.request_namespace,
            method_provenance=self.method_provenance,
            opening_adapter=opening_adapter,
        )
