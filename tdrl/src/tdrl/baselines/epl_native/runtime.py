import torch

from ...provenance import digest, file_hash
from ..epl.assets import verify_assets
from ..epl.checkpoint import load_checkpoint
from ..epl.contracts import EPLCheckpointMetadata
from ..epl.model import load_pretrained_policy
from ..epl.runtime import SOURCE_REVISION, load_tokenizer
from .config import EPLNativeConfig


def load_inference(
    config_path,
    vocabulary_path,
    checkpoint,
    encoder,
    retrieval_encoder,
    model_lock,
    device="cuda",
):
    config = EPLNativeConfig.from_paths(config_path, vocabulary_path)
    verify_assets(model_lock, encoder, retrieval_encoder)
    tokenizer = load_tokenizer(encoder)
    model = load_pretrained_policy(encoder, config, len(tokenizer), device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = EPLCheckpointMetadata(**payload.get("metadata", {}))
    if (
        metadata.schema_version != 1
        or metadata.dataset != config.dataset
        or metadata.config_sha256 != config.sha256
        or metadata.training_sha256 != config.training_sha256
        or metadata.action_taxonomy_sha256 != digest(config.actions)
        or metadata.model_lock_sha256 != file_hash(model_lock)
        or metadata.source_revision != SOURCE_REVISION
    ):
        raise ValueError("EPL native inference checkpoint provenance mismatch")
    load_checkpoint(checkpoint, model, metadata, map_location=device)
    model.eval()
    return model, tokenizer, config, metadata
