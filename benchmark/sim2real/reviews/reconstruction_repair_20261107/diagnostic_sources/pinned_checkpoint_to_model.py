from pathlib import Path
from typing import Any
import copy,torch

def checkpoint_to_model(checkpoint_path: Path, device: torch.device, model_cls, model_args_override: dict[str, Any] | None=None):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint:
        raise ValueError("checkpoint must contain the candidate model state under key 'model'")
    checkpoint_args = checkpoint.get('model_args')
    if checkpoint_args is None:
        if model_args_override is None:
            raise ValueError('checkpoint lacks model_args; pass the registered Stage-5 model config')
        checkpoint_args = model_args_override
    elif model_args_override is not None and checkpoint_args != model_args_override:
        raise ValueError('checkpoint model_args differ from the explicitly registered model config')
    model_args = copy.deepcopy(checkpoint_args)
    if not isinstance(model_args, dict) or model_args.get('num_bins') != 5:
        raise ValueError('E2VID model_args must be a dict with num_bins=5')
    model = model_cls(copy.deepcopy(model_args)).to(device)
    incompatible = model.load_state_dict(checkpoint['model'], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f'strict checkpoint restore mismatch: {incompatible}')
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return (model, checkpoint, model_args)
