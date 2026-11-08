"""Bounded paired synthetic pose trial; reuse the frozen full native evaluator."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
CORE = '62da4f35ce3c5c96f04268bd1687a27657af9238'
GPU4 = 'GPU-0602549d-5c0b-133e-63ab-a01d1f6a49d3'
NATIVE_SHA = '85951e6d43e59140b81d7303370a6b252fec902dc065ff17bb30faf8e4fc6186'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): value.update(block)
    return value.hexdigest()


def load(path):
    spec = importlib.util.spec_from_file_location('pose_priority_bound_native', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign', 'prior-source-dir', 'radiance-train', 'radiance-validation', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--prior-source-manifest-sha256', required=True)
    parser.add_argument('--phase', choices=('smoke', 'study'), required=True)
    args = parser.parse_args()
    assert args.out.is_absolute() and not args.out.exists(), 'Preserve existing results'
    assert not args.out.is_symlink() and args.out.parent.is_dir()
    assert json.loads((args.campaign / 'sources/training_recipe.json').read_text())['frozen_simulator_core'] == CORE
    sources = args.prior_source_dir
    assert sources.resolve(strict=True) == sources
    assert sha(sources / 'source_manifest.json') == args.prior_source_manifest_sha256
    manifest = json.loads((sources / 'source_manifest.json').read_text())
    for name, digest in manifest.items():
        assert Path(name).name == name and sha(sources / name) == digest
    native_path = sources / 'pose_two_branch_coverage.py'
    assert sha(native_path) == NATIVE_SHA
    own = Path(__file__).resolve().parent
    own_manifest = json.loads((own / 'source_manifest.json').read_text())
    for name, digest in own_manifest.items():
        assert Path(name).name == name and sha(own / name) == digest
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (GPU4, '4'), 'Use the existing assigned GPU4 queue only'
    actual = subprocess.check_output(['nvidia-smi', '-i', '4', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=15).strip()
    assert actual == GPU4
    import torch
    assert torch.cuda.device_count() == 1
    torch.backends.cudnn.benchmark = False
    sys.path.insert(0, str(sources))
    native = load(native_path)
    from pose_priority_policy import BoundedFitter
    smoke = args.phase == 'smoke'
    fitter = BoundedFitter(native, args.out, smoke)
    native.fit, native.fit_center = fitter.fit, fitter.fit_center
    original_write = native.shared.write
    def write(path, value):
        if Path(path) == args.out / 'protocol.json':
            value = dict(value, source_sha256=sha(Path(__file__).resolve()),
                         native_source_sha256=NATIVE_SHA, native_main_and_evaluation_unchanged=True,
                         policy='half_priority_same_view_retention', auxiliary_half_probability=.75,
                         expected_supervision_fractions=dict(normal=.5, half=.375, eighth=.125),
                         paired_teacher_weight=2, paired_teacher_input='same half view; no paired teacher term on eighth',
                         training='unchanged cached physical events; normal replay plus half-prioritized physical auxiliary view',
                         real_evaluation='disabled; all original thirty illumination and eight confirmation checks retained')
        return original_write(path, value)
    native.shared.write = write
    command = [str(native_path), '--campaign', str(args.campaign), '--radiance-train', str(args.radiance_train),
               '--radiance-validation', str(args.radiance_validation), '--out', str(args.out)]
    if smoke: command.append('--smoke')
    original_argv = sys.argv
    try:
        sys.argv = command
        native.main()
        assert len(fitter.records) == (12 if smoke else 36)
        # Each same-seed method/arm/branch consumes the same labels and affine draws.
        for seed in ([0] if smoke else range(3)):
            assert len({r['labels_sha256'] for r in fitter.records if r['seed'] == seed}) == 1
        result = json.loads((args.out / 'completion.json').read_text())
        assert result['status'] == 'complete' and result['real_data_used'] is False
        if not smoke:
            assert len(result['gate']['checks']) == 30 and len(result['original_confirmation']['comparisons']) == 8
        original_write(args.out / 'priority_execution_receipt.json', dict(
            status='complete_synthetic_only', phase=args.phase, real_scores_computed=False,
            source_manifest_sha256=sha(own / 'source_manifest.json'), prior_source_manifest_sha256=args.prior_source_manifest_sha256,
            native_source_sha256=NATIVE_SHA, frozen_core=CORE,
            training_inputs='whole-frame events; synthetic targets used only in supervised losses',
            protected_conditions=['normal', 'half'], dark_teacher_transfer=False,
            records=fitter.records, thirty_gate=result.get('gate'), original_confirmation=result.get('original_confirmation')))
    finally:
        native.shared.write = original_write
        sys.argv = original_argv


if __name__ == '__main__': main()
