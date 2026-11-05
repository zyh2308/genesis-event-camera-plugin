"""Strict projection-only regression for the homogeneous-storage candidate.

Run only on an authorized, atomically reserved device. This does not qualify
full events, sensor state, RNG, loaders, or default CUDA behavior.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

# Must precede torch import and CUDA initialization.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_bytes(value):
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_row(value):
    data = tensor_bytes(value)
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(data).hexdigest()}


def snapshot(value, prefix="input"):
    if torch.is_tensor(value):
        return {prefix: tensor_row(value)}
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            result.update(snapshot(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(snapshot(item, f"{prefix}.{index}"))
    return result


def pose(device, x=0., y=0., angle=0.):
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor([[cosine, -sine, 0., x],
                         [sine, cosine, 0., y],
                         [0., 0., 1., 0.],
                         [0., 0., 0., 1.]], dtype=torch.float64, device=device)


def fixture(device, mapping, invalid=False, noncontiguous=False):
    height, width = 9, 13
    # No random numbers are consumed by fixture construction.
    pixels = torch.arange(height * width, dtype=torch.float64, device=device).reshape(height, width)
    depth = (1. + pixels * .003).to(torch.float32)
    segmentation = torch.where((pixels.long() % 3) == 0, 1, 2).long()
    if invalid:
        depth[0, 0] = float("nan")
        depth[0, 1] = float("inf")
        depth[0, 2] = 0.
        depth[0, 3] = -.01
        segmentation[-1, -1] = -1
    if noncontiguous:
        storage = torch.empty((height, width * 2), dtype=depth.dtype, device=device)
        storage[:, ::2] = depth
        depth = storage[:, ::2]
        label_storage = torch.empty((height, width * 2), dtype=segmentation.dtype, device=device)
        label_storage[:, ::2] = segmentation
        segmentation = label_storage[:, ::2]
    intrinsics = torch.tensor([[37., 0., 6.], [0., 39., 4.], [0., 0., 1.]],
                              dtype=torch.float64, device=device)
    camera0, camera1 = pose(device), pose(device, .03, -.01, .04)
    alpha = torch.tensor([0., 1e-11, .1, .25, .5, .9, 1.-1e-11, 1.],
                         dtype=torch.float64, device=device)
    object0 = {1: pose(device, .01, .02, -.03), 2: pose(device, -.02, .01, .07)}
    object1 = {1: pose(device, .06, -.02, .06), 2: pose(device, .02, -.03, -.02)}
    deltas = {key: object1[key] @ torch.linalg.inv(value) for key, value in object0.items()}
    if mapping == "empty":
        object0, object1, deltas = {}, {}, {}
    elif mapping == "none":
        object0, object1, deltas = None, None, None
    elif mapping == "pose_two":
        deltas = None
    elif mapping == "delta_two":
        object0, object1 = None, None
    elif mapping == "absent_pose":
        object0, object1, deltas = {99: object0[1]}, {99: object1[1]}, None
    elif mapping == "absent_delta":
        object0, object1, deltas = None, None, {99: deltas[1]}
    elif mapping == "mixed":
        object0, object1 = {1: object0[1]}, {1: object1[1]}
    else:
        raise ValueError(mapping)
    return depth, segmentation, intrinsics, camera0, camera1, alpha, object0, object1, deltas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing regression receipt")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This regression must run on the reserved CUDA device")
    torch.use_deterministic_algorithms(True)
    reference = load_module("projection_storage_reference", args.reference.resolve())
    candidate = load_module("projection_storage_candidate", args.candidate.resolve())
    before_rng = tensor_bytes(torch.random.get_rng_state())
    before_cuda_rng = tensor_bytes(torch.cuda.get_rng_state(device))
    cases, failures = [], []
    with torch.inference_mode():
        for mapping in ("none", "empty", "pose_two", "delta_two", "absent_pose", "absent_delta", "mixed"):
            for invalid, strided in ((False, False), (True, False), (False, True)):
                data = fixture(device, mapping, invalid, strided)
                depth, segmentation, intrinsics, camera0, camera1, alpha, object0, object1, deltas = data
                for source_index in (0, 1):
                    source = camera0 if source_index == 0 else camera1
                    inputs = (depth, segmentation, intrinsics, source, camera0, camera1,
                              alpha, source_index, object0, object1, deltas)
                    initial = snapshot(inputs)
                    left = reference.project_endpoint_to_alpha_stack_torch(*inputs)
                    after_reference = snapshot(inputs)
                    right = candidate.project_endpoint_to_alpha_stack_torch(*inputs)
                    after_candidate = snapshot(inputs)
                    output_rows = []
                    for name, expected, actual in zip(("uv", "z", "valid"), left, right):
                        equal = expected.shape == actual.shape and expected.dtype == actual.dtype
                        equal = equal and tensor_bytes(expected) == tensor_bytes(actual)
                        output_rows.append({"name": name, "bitwise_equal": equal,
                                            "reference": tensor_row(expected), "candidate": tensor_row(actual)})
                    unchanged = initial == after_reference == after_candidate
                    passed = unchanged and all(row["bitwise_equal"] for row in output_rows)
                    row = {"mapping": mapping, "invalid_depth": invalid, "noncontiguous_inputs": strided,
                           "source_index": source_index, "inputs_byte_unchanged": unchanged,
                           "passed": passed, "outputs": output_rows}
                    cases.append(row)
                    if not passed:
                        failures.append(len(cases) - 1)
                    del left, right
    rng_equal = before_rng == tensor_bytes(torch.random.get_rng_state())
    cuda_rng_equal = before_cuda_rng == tensor_bytes(torch.cuda.get_rng_state(device))
    report = {
        "schema": "projection_storage_bitwise_regression_v1",
        "passed": not failures and rng_equal and cuda_rng_equal,
        "case_count": len(cases), "failed_case_indices": failures,
        "cpu_rng_unchanged": rng_equal, "cuda_rng_unchanged": cuda_rng_equal,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "device": str(device), "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
        "scope": "projection only; full plugin and default CUDA qualification not inferred",
        "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("passed", "case_count", "failed_case_indices",
                                                "cpu_rng_unchanged", "cuda_rng_unchanged")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
