import torch

from ...provenance import digest, file_hash
from .assets import verify_assets
from .checkpoint import load_checkpoint
from .config import DPDPConfig
from .contracts import DPDPCheckpointMetadata
from .model import load_pretrained_policy


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if (
        tokenizer.cls_token_id is None
        or tokenizer.sep_token_id is None
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("DPDP tokenizer lacks required special tokens")
    return tokenizer


def load_inference(config_path, checkpoint, encoder, tokenizer_path, model_lock, device="cuda"):
    config = DPDPConfig.from_path(config_path)
    assets = verify_assets(model_lock, encoder, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_pretrained_policy(
        encoder, config.actions, config.selfplay_dropout, device
    )
    # Scientific checkpoints retain exact data hashes. Read metadata first, then verify all
    # immutable fields except the per-run data hash which is not duplicated in evaluation args.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stored = payload.get("metadata", {})
    metadata = DPDPCheckpointMetadata(**stored)
    if (
        metadata.phase != "selfplay"
        or metadata.config_sha256 != config.fitting_sha256
        or metadata.action_taxonomy_sha256
        != digest(config.actions)
        or metadata.model_lock_sha256 != file_hash(model_lock)
        or metadata.source_revision != "4d1e22e35db452d51f0f39388456d310ec1c971b"
        or metadata.parent_checkpoint_sha256 is None
    ):
        raise ValueError("DPDP inference checkpoint provenance mismatch")
    load_checkpoint(checkpoint, model, metadata, map_location=device)
    model.eval()
    return model, tokenizer, config, assets, metadata
