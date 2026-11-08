#!/usr/bin/env python3
"""CPU algebra-only native STE mask audit; no optimizer, model CP, SSH or CUDA."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

NATIVE_SHA = 'ff401317e49ce37ae0f8e9bfd6b82f133549b4a0271be35095f74bdc50a439c0'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).resolve().parents[2] / 'docs/sim2real-research-2026-10-16/sources/STE-FlowNet/multiscaleloss.py'
    parser.add_argument('--native-source', type=Path, default=default)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args()
    source = a.native_source.resolve(strict=True)
    before = source.read_bytes()
    if hashlib.sha256(before).hexdigest() != NATIVE_SHA:
        raise RuntimeError('Native source is not the frozen STE loss')
    if a.out.exists():
        raise FileExistsError('Refusing to overwrite an existing algebra receipt')
    import torch
    torch.set_num_threads(2)
    spec = importlib.util.spec_from_file_location('native_steflow_loss_algebra_only', source)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    original_cuda = torch.Tensor.cuda
    original_grid = torch.nn.functional.grid_sample
    calls = []
    def cpu_identity_cuda(self, *unused_args, **unused_kwargs):
        # This only permits executing the original hard-coded .cuda() calls on CPU.
        # No GPU op, precision, correctness or performance claim follows from this shim.
        return self
    def existing_align_corners_compat(x, grid, *args, **kwargs):
        kwargs.setdefault('align_corners', True)
        value = original_grid(x, grid, *args, **kwargs)
        calls.append(value)
        return value
    torch.Tensor.cuda = cpu_identity_cuda
    torch.nn.functional.grid_sample = existing_align_corners_compat
    results = []
    try:
        H = W = 11
        yy, xx = torch.meshgrid(torch.arange(H, dtype=torch.float32), torch.arange(W, dtype=torch.float32), indexing='ij')
        previous = ((xx + yy * .7) / 25 + .1)[None, None]
        following = ((xx * .4 + yy) / 25 + .2)[None, None]
        cases = dict(inbounds=torch.zeros(1, 2, H, W), all_outbounds=torch.ones(1, 2, H, W) * 30,
            mixed_outbounds=torch.stack((torch.linspace(-3, 3, W).repeat(H, 1), torch.zeros(H, W)), 0)[None],
            near_boundary=torch.stack((torch.full((H, W), 1e-4), torch.zeros(H, W)), 0)[None])
        for name, base in cases.items():
            for parameter_space in (False, True):
                if parameter_space:
                    model = torch.nn.Conv2d(1, 2, 1)
                    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
                    flow = base + model(previous) * .01; targets = tuple(model.parameters())
                else:
                    flow = base.clone().requires_grad_(True); targets = (flow,)
                calls.clear()
                old = native.compute_photometric_loss(previous[:, 0].numpy(), following[:, 0].numpy(), None, (flow,), weights=[1.])
                mask = calls[1]
                mask_gradient = torch.autograd.grad(mask.sum(), flow, retain_graph=True)[0]
                warped = native.warp(following, flow)
                masked_residual = native.charbonnier_loss((warped - previous) * mask)
                delta = warped - previous
                weighted_charbonnier = (torch.pow(torch.mul(delta, delta) + torch.mul(1e-3, 1e-3), .45) * mask).sum()
                original_grad = torch.autograd.grad(old, targets, retain_graph=True)
                corrected_grad = torch.autograd.grad(masked_residual, targets, retain_graph=True)
                weighted_grad = torch.autograd.grad(weighted_charbonnier, targets)
                results.append(dict(case=name, gradient_space='tiny_conv_parameters' if parameter_space else 'predicted_flow',
                    native_photo=float(old.detach()), masked_residual_photo=float(masked_residual.detach()),
                    weighted_charbonnier_photo=float(weighted_charbonnier.detach()),
                    valid_fraction=float(mask.detach().mean()), mask_requires_grad=mask.requires_grad,
                    mask_flow_gradient_maxabs=float(mask_gradient.abs().max()),
                    residual_gradient_maxabs_difference=max(float((x-y).abs().max()) for x,y in zip(original_grad,corrected_grad)),
                    weighted_gradient_maxabs_difference=max(float((x-y).abs().max()) for x,y in zip(original_grad,weighted_grad)),
                    native_gradient_l2=float(sum(v.square().sum() for v in original_grad).sqrt())))
        # Preserve the upstream nonsquare defect as a bounded, separate negative fixture.
        nonsquare = dict(shape_hw=[9,13], relevant_to_formal_square_training=False)
        try:
            native.compute_photometric_loss(torch.zeros(1,9,13).numpy(), torch.zeros(1,9,13).numpy(), None,
                                             (torch.zeros(1,2,9,13),), weights=[1.])
            nonsquare.update(status='unexpectedly_did_not_reject')
        except RuntimeError as exc:
            nonsquare.update(status='upstream_resize_width_height_reversed_observed', exception=str(exc))
    finally:
        torch.Tensor.cuda = original_cuda
        torch.nn.functional.grid_sample = original_grid
    if source.read_bytes() != before:
        raise RuntimeError('Original native source changed')
    passed = len(results) == 8 and all(r['residual_gradient_maxabs_difference'] == 0 and
             r['weighted_gradient_maxabs_difference'] == 0 and r['mask_flow_gradient_maxabs'] == 0 for r in results)
    if not passed:
        raise RuntimeError('Algebra expectation failed; do not interpret as an improvement')
    result = dict(schema='steflow_native_mask_algebra_cpu_v1', status='passed', source_path=str(source),
        source_sha256=NATIVE_SHA, source_unchanged=True, execution_device='CPU FP32', torch_version=torch.__version__,
        cuda_shim='Temporarily torch.Tensor.cuda(self,*args,**kwargs) returns self; restored in finally',
        coordinate_compatibility='Temporarily grid_sample default align_corners=True, matching existing explicit compatibility; restored in finally',
        case_count=8, all_gradient_differences_exactly_zero=True, results=results, nonsquare_negative_fixture=nonsquare,
        conclusion='SUM + hard binary warp mask: deleting invalid residual terms changes only loss constants; no flow or parameter gradient change in these cases. Mask-only is a negative control, not a training improvement.',
        limits=['CPU algebra probe, not a CUDA numerical/performance test or downstream quality result.',
                'masked residual retains Charbonnier epsilon baseline at invalid pixels; post-Charbonnier weighting removes that baseline too; both have identical gradients.',
                'Physical GT-invalid region includes geometry/occlusion and is not identical to predicted out-of-bounds.',
                'Native cv2.resize uses (height,width), reversed for nonsquare arrays; formal256x256 and352x352 paths are unaffected; not claimed as the training root cause.'],
        optimizer_updates=0, real_data_used=False, pretrained_weights_loaded=False, SSH_connected=False)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
