"""Pinned official kernel, restricted to the existing STE-FlowNet operator contract."""
import sys
from pathlib import Path
import torch
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root/'build'))
sys.path.insert(0, str(root/'upstream/Correlation_Module'))
from spatial_correlation_sampler import spatial_correlation_sample as _native
def spatial_correlation_sample(input1, input2, kernel_size=1, patch_size=9,
                               stride=1, padding=0, dilation=1, dilation_patch=1):
    if (kernel_size, patch_size, stride, padding, dilation, dilation_patch) != (1,9,1,0,1,1):
        raise ValueError('Unvalidated correlation parameters')
    if (not input1.is_cuda or not input2.is_cuda or input1.device != input2.device
            or input1.dtype != input2.dtype or input1.shape != input2.shape
            or input1.dtype not in (torch.float32, torch.float64)):
        raise ValueError('Expected equal CUDA feature shapes and dtypes')
    # The pinned upstream kernel launches on the default CUDA stream.
    # Reject unsupported concurrent-stream use rather than silently race.
    if torch.cuda.current_stream(input1.device) != torch.cuda.default_stream(input1.device):
        raise ValueError('Only the audited default CUDA stream is supported')
    return _native(input1.contiguous(), input2.contiguous(), kernel_size=1,
                   patch_size=9, stride=1, padding=0, dilation=1, dilation_patch=1)
