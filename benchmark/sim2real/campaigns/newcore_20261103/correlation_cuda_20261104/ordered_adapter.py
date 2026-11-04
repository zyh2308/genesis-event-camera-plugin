"""Reference forward with an exactly ordered, non-fused CUDA backward."""
import sys
from pathlib import Path
import torch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'build_ordered'))
sys.path.insert(0,str(root.parent/'sources/benchmark/sim2real/scripts'))
import ordered_correlation_backward as _cuda
from run_steflow_stage2_pilot import _MemoryEfficientSpatialCorrelation as _reference
class _Ordered(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):return _reference.forward(ctx,a,b)
    @staticmethod
    def backward(ctx,g):
        a,b=ctx.saved_tensors
        return tuple(_cuda.backward(a,b,g.contiguous()))
def spatial_correlation_sample(input1,input2,kernel_size=1,patch_size=9,stride=1,padding=0,dilation=1,dilation_patch=1):
    if (kernel_size,patch_size,stride,padding,dilation,dilation_patch)!=(1,9,1,0,1,1):raise ValueError('Unvalidated parameters')
    if (not input1.is_cuda or input1.device!=input2.device or input1.shape!=input2.shape or input1.dtype!=input2.dtype or input1.dtype not in (torch.float32,torch.float64)):raise ValueError('Expected equal CUDA FP32/FP64 tensors')
    return _Ordered.apply(input1.contiguous(),input2.contiguous())
