import json
from pathlib import Path

from ...provenance import atomic_json, digest, file_hash
from .guidance import GuidanceArtifact


class GuidanceStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(dataset, split, scenario_id, seed):
        return digest([dataset, split, scenario_id, seed, "difftod-guidance-v1"])

    def path(self, dataset, split, scenario_id, seed):
        return self.root / dataset / split / f"{self.key(dataset, split, scenario_id, seed)}.json"

    def write(self, artifact: GuidanceArtifact):
        path = self.path(artifact.dataset, artifact.split, artifact.scenario_id, artifact.seed)
        if path.exists():
            if self.load(
                artifact.dataset, artifact.split, artifact.scenario_id, artifact.seed
            ) != artifact:
                raise ValueError("attempt to overwrite completed DiffTOD guidance")
            return path
        atomic_json(path, artifact.__dict__)
        return path

    def load(self, dataset, split, scenario_id, seed):
        path = self.path(dataset, split, scenario_id, seed)
        if not path.is_file():
            raise FileNotFoundError(f"missing DiffTOD guidance: {dataset}/{scenario_id}/{seed}")
        artifact = GuidanceArtifact(**json.loads(path.read_text(encoding="utf-8")))
        if (artifact.dataset, artifact.split, artifact.scenario_id, artifact.seed) != (
            dataset,
            split,
            scenario_id,
            seed,
        ):
            raise ValueError("DiffTOD guidance key does not match stored identity")
        return artifact

    def hash(self, dataset, split, scenario_id, seed):
        return file_hash(self.path(dataset, split, scenario_id, seed))
