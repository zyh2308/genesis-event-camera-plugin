"""Keep the pinned STE-FlowNet working on the device of its input tensor.

The upstream model contains legacy ``Tensor.cuda()`` calls for intermediate
buffers.  On CPU-only fallback runs those calls fail even though every tensor
is otherwise constructed from or placed on the CPU input.  This narrow
process-local shim makes no-argument ``.cuda()`` a no-op only when CUDA is
unavailable and the tensor is already on CPU.  On CUDA hosts it delegates to
the original PyTorch method unchanged.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def install_cpu_cuda_placement_compat(device: torch.device) -> dict:
    if device.type != "cpu":
        return {
            "status": "not_needed",
            "reason": "non-CPU execution preserves upstream Tensor.cuda behavior",
            "device": str(device),
        }
    if torch.cuda.is_available():
        raise RuntimeError("CPU placement shim must only be installed when CUDA is unavailable")

    current = torch.Tensor.cuda
    if getattr(current, "_steflow_cpu_placement_compat", False):
        patched = current
    else:
        original = current

        def cpu_compatible_cuda(self, device=None, non_blocking=False,
                                memory_format=torch.preserve_format):
            if device is None and self.device.type == "cpu" and not torch.cuda.is_available():
                return self
            return original(self, device=device, non_blocking=non_blocking,
                            memory_format=memory_format)

        cpu_compatible_cuda._steflow_cpu_placement_compat = True
        cpu_compatible_cuda._steflow_original_cuda = original
        torch.Tensor.cuda = cpu_compatible_cuda
        patched = cpu_compatible_cuda

    probe = torch.zeros(1, device="cpu")
    if probe.cuda().device.type != "cpu" or probe.cuda().data_ptr() != probe.data_ptr():
        raise RuntimeError("CPU fallback placement probe failed")
    source = Path(__file__).resolve()
    return {
        "status": "applied_cpu_device_placement_only",
        "device": str(device),
        "cuda_available": False,
        "scope": "process-local no-argument Tensor.cuda() on CPU tensors only",
        "model_math_or_parameters_changed": False,
        "cuda_execution_behavior_changed": False,
        "probe": "CPU tensor .cuda() preserves device and storage",
        "compatibility_source": str(source),
        "compatibility_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
