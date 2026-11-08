"""CPU-only checks of the unchanged, pinned HQF metric code and local assets."""
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-recon-selected-evaluation-v1')
OLD = ROOT.parent / '20261108-hqf-repair-pair4800-v2/protocol/hqf_repair_pair_protocol.json'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    protocol = json.loads(OLD.read_text())
    bound = protocol['evaluation_sources']['evaluate_synthetic_reconstruction_fixture.py']
    helper = Path(bound['path'])
    assert sha(helper) == bound['sha256']
    sys.path.insert(0, str(helper.parent))
    spec = importlib.util.spec_from_file_location('hqf_bound_numeric_selftest', helper)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    torch.set_num_threads(2)
    target = np.full((64, 64), .4, dtype=np.float32)
    prediction = np.full_like(target, .3)
    metrics = module.classical_metrics(prediction, target)
    expected_ssim = (2 * .3 * .4 + .01**2) / (.3**2 + .4**2 + .01**2)
    assert abs(metrics['mse'] - .01) < 1e-8
    assert abs(metrics['psnr_db'] - 20) < 1e-5
    assert abs(metrics['ssim'] - expected_ssim) < 1e-5
    same = module.classical_metrics(target, target)
    assert same['mse'] == 0 and math.isinf(same['psnr_db']) and same['ssim'] == 1
    for wrong in (np.full_like(target, np.nan), np.full_like(target, 1.1)):
        try:
            module.classical_metrics(wrong, target)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite/out-of-range input was accepted')
    device = torch.device('cpu')
    model, source, weights, backbone = module.load_lpips_alex(Path(protocol['model_root']), device)
    with torch.inference_mode():
        x = module.lpips_input(target, device)
        identity = float(model(x, x).reshape(-1).mean())
    assert math.isfinite(identity) and abs(identity) < 1e-8
    result = dict(status='passed', synthetic_fixture_only=True, real_images_read=False,
                  GPU_used=False, metric_helper_sha256=sha(helper), classical=metrics,
                  expected_ssim=expected_ssim, identity_lpips_alex=identity,
                  metric_assets={str(p): sha(p) for p in (Path(source), Path(weights), Path(backbone))})
    with (ROOT / 'metric_runtime_selftest.json').open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print('PINNED_HQF_METRIC_NUMERICAL_SELFTEST_PASSED', flush=True)


if __name__ == '__main__':
    main()
