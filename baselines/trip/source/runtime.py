import torch

from .assets import verify_assets
from .checkpoint import load_checkpoint
from .config import TripConfig
from .dataset import add_special_tokens
from .model import load_pretrained_model


def load_tokenizer(path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    token_ids = add_special_tokens(tokenizer)
    return tokenizer, token_ids


def load_inference(config_path, checkpoint, encoder, tokenizer_path, asset_lock, device="cuda"):
    config = TripConfig.from_path(config_path)
    assets = verify_assets(asset_lock, encoder, tokenizer_path)
    tokenizer, token_ids = load_tokenizer(tokenizer_path)
    model = load_pretrained_model(encoder, tokenizer, config.model, device)
    load_checkpoint(checkpoint, model, config.training_sha256, map_location=device)
    if torch.device(device).type == "cuda":
        model.to(dtype=torch.bfloat16)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer, token_ids, config, assets
