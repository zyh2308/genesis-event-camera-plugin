"""Invoke the unchanged archived 35-case strict geometry stress suite."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
        raise RuntimeError('One atomically reserved CUDA device is required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    stress_path = Path(__file__).resolve().parents[2]/'sampler_cost_20261105/source/stress_geometry.py'
    stress = load('archived_strict_stress', stress_path)
    reference = load('v1_geometry_reference', args.reference)
    candidate = load('isolated_geometry_candidate', args.candidate)
    result = stress.run_stress(reference, {'zbuf_reuse': candidate})
    result.update(reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                  stress_source_sha256=hashlib.sha256(stress_path.read_bytes()).hexdigest(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('STRICT_STRESS', result['case_count'], result['all_passed'], flush=True)
    if not result['all_passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
