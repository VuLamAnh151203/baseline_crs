from pathlib import Path

import torch

from ...provenance import digest, file_hash
from .assets import verify_assets
from .checkpoint import load_checkpoint
from .config import EPLConfig
from .contracts import EPLCheckpointMetadata
from .model import load_pretrained_policy

SOURCE_REVISION = "364c6fd17f1bbdb2abcac9cd2381a29bf1bd5dda"
SPECIAL_TOKENS = (
    "[USER]",
    "[SYSTEM]",
    "[KNOW]",
    "[PATH]",
    "[SEP]",
    "[PROFILE]",
    "[CONTEXT]",
    "[GOAL]",
    "[TARGET]",
)


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS)})
    if any(
        token is None
        for token in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
    ):
        raise ValueError("EPL tokenizer lacks required special tokens")
    return tokenizer


def load_inference(
    config_path,
    checkpoint,
    encoder,
    tokenizer_path,
    retrieval_encoder,
    model_lock,
    device="cuda",
):
    config = EPLConfig.from_path(config_path)
    if Path(encoder).resolve() != Path(tokenizer_path).resolve():
        raise ValueError("EPL policy tokenizer must come from the locked BERT directory")
    assets = verify_assets(model_lock, encoder, retrieval_encoder)
    tokenizer = load_tokenizer(tokenizer_path)
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
        raise ValueError("EPL inference checkpoint provenance mismatch")
    load_checkpoint(checkpoint, model, metadata, map_location=device)
    model.eval()
    return model, tokenizer, config, assets, metadata
