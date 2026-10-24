#!/usr/bin/env python3
"""Exercise fixed STE-FlowNet entry/model/checkpoint contracts without a GPU.

The pinned STE-FlowNet forward hard-codes Tensor.cuda(). This isolated code
path check replaces only that method with identity and runs on CPU; it is not
evidence of CUDA-kernel or full-bank training correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    model_root = args.model_root.resolve()
    scripts = repo / 'benchmark/sim2real/scripts'
    sys.path.insert(0, str(scripts))
    from train_stage5_flow_selfsup_diagnostic import (
        initialize_model, registered_code_fingerprints, state_sha256,
    )

    fingerprints = registered_code_fingerprints(repo, model_root)
    expected_key = '@model/models/STEFlow.py'
    if expected_key not in fingerprints:
        raise RuntimeError(f'correctly cased STEFlow fingerprint absent: {fingerprints.keys()}')
    torch.set_num_threads(2)
    original_cuda = torch.Tensor.cuda
    original_cuda_seed = torch.cuda.manual_seed_all
    torch.Tensor.cuda = lambda self, *unused_args, **unused_kwargs: self
    torch.cuda.manual_seed_all = lambda *unused_args, **unused_kwargs: None
    try:
        device = torch.device('cpu')
        model, optimizer, _, _, upstream = initialize_model(model_root, 20310421, device)
        initial_hash = state_sha256(model)
        sample = torch.randn((1, 2, 32, 32, 5), dtype=torch.float32)
        model.train()
        outputs = model(sample, 32, use_dt4=False, flow_seq=False)
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise RuntimeError('training forward must return all four native STE-FlowNet scales')
        loss = sum(output.square().mean() for output in outputs)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError('model backward did not produce finite gradients')
        if not any(bool(torch.count_nonzero(gradient)) for gradient in gradients):
            raise RuntimeError('model backward gradients are all zero')
        optimizer.step()
        trained_hash = state_sha256(model)
        if trained_hash == initial_hash:
            raise RuntimeError('optimizer step left the complete model state unchanged')
        model.eval()
        with torch.inference_mode():
            prediction_before = model(sample, 32, use_dt4=False, flow_seq=False).clone()
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'optimizer_update': 1,
            'state_sha256': trained_hash,
            'source_sha256': fingerprints,
            'upstream_sha256': upstream['upstream_hashes'],
        }
        with tempfile.TemporaryDirectory(prefix='sim2real-steflow-contract-') as temp_dir:
            checkpoint_path = Path(temp_dir) / 'roundtrip.pt'
            torch.save(checkpoint, checkpoint_path)
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            restored, _, _, _, _ = initialize_model(model_root, 9917, device)
            restored_state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            restored.load_state_dict(restored_state['state_dict'], strict=True)
            restored.eval()
            with torch.inference_mode():
                prediction_after = restored(sample, 32, use_dt4=False, flow_seq=False)
        max_abs = float((prediction_before - prediction_after).abs().max())
        if max_abs != 0.0 or state_sha256(restored) != trained_hash:
            raise RuntimeError(f'ordinary eval output differs after fresh checkpoint restore: {max_abs}')
        result = {
            'schema': 'stage5_steflow_cpu_shim_entry_contract_v1',
            'status': 'pass_cpu_functional_path_only',
            'real_data_used': False,
            'optimizer_updates': 1,
            'device': 'cpu',
            'cpu_shim': 'Tensor.cuda returns the input tensor; torch.cuda.manual_seed_all is disabled for this process only',
            'cuda_validation': False,
            'full_synthetic_bank_training': False,
            'training_input_shape': list(sample.shape),
            'training_scale_shapes': [list(output.shape) for output in outputs],
            'training_loss': float(loss.detach()),
            'finite_gradient_tensors': len(gradients),
            'nonzero_gradient_tensors': sum(bool(torch.count_nonzero(gradient)) for gradient in gradients),
            'initial_state_sha256': initial_hash,
            'post_update_state_sha256': trained_hash,
            'ordinary_eval_shape': list(prediction_after.shape),
            'checkpoint_sha256': checkpoint_sha,
            'checkpoint_restore_max_abs_output_difference': max_abs,
            'code_fingerprints': fingerprints,
            'upstream_hashes': upstream['upstream_hashes'],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        torch.Tensor.cuda = original_cuda
        torch.cuda.manual_seed_all = original_cuda_seed


if __name__ == '__main__':
    main()
