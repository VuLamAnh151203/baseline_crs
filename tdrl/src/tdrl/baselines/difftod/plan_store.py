import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ...provenance import atomic_json, digest, file_hash


@dataclass(frozen=True)
class PlanArtifact:
    schema_version: int
    dataset: str
    split: str
    scenario_id: str
    seed: int
    budget: str
    checkpoint_sha256: str
    config_sha256: str
    guidance_sha256: str
    guidance: list[str]
    candidate_plans: list[str]
    self_bleu_scores: list[float]
    selected_candidate: int
    selected_plan: str
    diffusion_steps: int
    diffusion_model_calls: int
    diffusion_sequence_forwards: int
    wall_s: float
    guidance_wall_s: float
    peak_memory_bytes: int
    status: str = "complete"

    def __post_init__(self):
        if self.schema_version != 1 or self.status != "complete":
            raise ValueError("unknown/incomplete DiffTOD plan artifact")
        if self.dataset not in {"TopDial", "DuRecDial2.0"}:
            raise ValueError("unknown DiffTOD plan dataset")
        if self.split not in {"dev", "test"}:
            raise ValueError("unknown DiffTOD plan split")
        if not self.scenario_id or self.seed < 0:
            raise ValueError("invalid DiffTOD plan identity")
        if self.budget not in {"natural", "matched"}:
            raise ValueError("unknown DiffTOD plan budget")
        for value in (self.checkpoint_sha256, self.config_sha256, self.guidance_sha256):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("DiffTOD plan provenance requires SHA-256 hashes")
        count = len(self.candidate_plans)
        if (
            count < 1
            or len(self.guidance) != count
            or len(self.self_bleu_scores) != count
            or not 0 <= self.selected_candidate < count
            or self.selected_plan != self.candidate_plans[self.selected_candidate]
        ):
            raise ValueError("DiffTOD plan candidates/selection are inconsistent")
        if any(not isinstance(text, str) or not text.strip() for text in self.candidate_plans):
            raise ValueError("DiffTOD candidate plans must be nonempty text")
        if any(not math.isfinite(score) or score < 0 for score in self.self_bleu_scores):
            raise ValueError("DiffTOD MBR scores must be finite and nonnegative")
        if (
            self.diffusion_steps < 1
            or self.diffusion_model_calls not in {self.diffusion_steps, self.diffusion_steps + 1}
            or self.diffusion_sequence_forwards != self.diffusion_model_calls * count
        ):
            raise ValueError("DiffTOD diffusion accounting mismatch")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.wall_s, self.guidance_wall_s)
        ):
            raise ValueError("DiffTOD offline wall times must be nonnegative")
        if not isinstance(self.peak_memory_bytes, int) or self.peak_memory_bytes < 0:
            raise ValueError("DiffTOD peak memory must be a nonnegative integer")

    @property
    def sha256(self):
        return digest(asdict(self))


class PlanStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(dataset, split, scenario_id, seed, budget):
        return digest([dataset, split, scenario_id, seed, budget])

    def path(self, dataset, split, scenario_id, seed, budget):
        key = self.key(dataset, split, scenario_id, seed, budget)
        return self.root / dataset / split / budget / f"{key}.json"

    def write(self, artifact: PlanArtifact):
        path = self.path(
            artifact.dataset,
            artifact.split,
            artifact.scenario_id,
            artifact.seed,
            artifact.budget,
        )
        if path.exists():
            existing = self.load(
                artifact.dataset,
                artifact.split,
                artifact.scenario_id,
                artifact.seed,
                artifact.budget,
            )
            if existing != artifact:
                raise ValueError("attempt to overwrite a completed DiffTOD plan")
            return path
        atomic_json(path, asdict(artifact))
        return path

    def load(self, dataset, split, scenario_id, seed, budget):
        path = self.path(dataset, split, scenario_id, seed, budget)
        if not path.is_file():
            raise FileNotFoundError(f"missing DiffTOD plan: {dataset}/{scenario_id}/{seed}")
        artifact = PlanArtifact(**json.loads(path.read_text(encoding="utf-8")))
        if (
            artifact.dataset,
            artifact.split,
            artifact.scenario_id,
            artifact.seed,
            artifact.budget,
        ) != (
            dataset,
            split,
            scenario_id,
            seed,
            budget,
        ):
            raise ValueError("DiffTOD plan key does not match stored identity")
        return artifact

    def hash(self, dataset, split, scenario_id, seed, budget):
        return file_hash(self.path(dataset, split, scenario_id, seed, budget))
