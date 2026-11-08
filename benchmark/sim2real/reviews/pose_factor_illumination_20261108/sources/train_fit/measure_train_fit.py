"""CPU zero-update train-fit readback using the locked 32-window cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic-sources', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not a.out.exists() and a.out.is_absolute() and a.out.parent.is_dir()
    root = a.diagnostic_sources
    assert sha(root / 'source_manifest.json') == '9e604e49ae861ab3831d641cd8c6d57d506017b060ebd30ee7a0b23f80e8771b'
    manifest = json.loads((root / 'source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert Path(name).name == name and sha(root / name) == digest
    sys.path.insert(0, str(root))
    import torch
    import export_pose_factors as exporter
    import probe_pose_training_losses as probe
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.monotonic()
    contract = json.loads((root / 'contract.json').read_text())
    ctx = exporter.load_context(contract, check_gpu=False)
    data, cohort = probe.prepare_train(ctx, contract)
    records = []
    for method in contract['methods']:
        for seed in contract['seeds']:
            for arm, canonical in contract['arm_mapping'].items():
                for branch in ('depth', 'center'):
                    assert time.monotonic() - start < 300
                    model, receipt, path = exporter.load_branch(ctx, method, seed, arm, branch)
                    before = exporter.state_sha(model.state_dict())
                    with torch.inference_mode():
                        for light, (x, target) in data[method].items():
                            _, uv, size, shape = model(x)
                            error = ((uv - target[:, :2]) * uv.new_tensor([346, 260])).norm(dim=1)
                            row = dict(method=method, seed=seed, arm=canonical, branch=branch,
                                       condition=light, windows=32, center_px=float(error.mean()),
                                       source_state_sha256=before, source_path=str(path))
                            if branch == 'depth':
                                z = (shape + target[:, 3] - size).exp()
                                truth = target[:, 4].exp()
                                row.update(depth_mae_m=float((z-truth).abs().mean()),
                                           depth_bias_m=float((z-truth).mean()),
                                           log_size_mae=float((size-target[:, 2]).abs().mean()),
                                           log_shape_mae=float((shape-(target[:, 4]-target[:, 3]+target[:, 2])).abs().mean()))
                            records.append(row)
                    assert before == exporter.state_sha(model.state_dict()) == receipt['final_state_sha256']
                    del model
    assert len(records) == 108
    for item in ctx.source_records:
        assert sha(item['path']) == item['sha256']
    a.out.mkdir(exist_ok=False)
    result = dict(status='complete', optimizer_updates=0, real_data_used=False, GPU_used=False,
                  records=records, cohort=cohort, worker_sha256=sha(__file__),
                  source_manifest_sha256=sha(root/'source_manifest.json'),
                  elapsed_seconds=time.monotonic()-start,
                  limitation='Frozen CPU forward, not claimed bitwise CUDA parity; selected32 TRAIN windows do not estimate population generalization.')
    (a.out/'completion.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('TRAIN_FIT_COMPLETE', len(records), flush=True)


if __name__ == '__main__':
    main()
