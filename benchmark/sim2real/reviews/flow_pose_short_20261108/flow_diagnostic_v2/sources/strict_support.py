"""Conservative GT-oracle support only; never a training/network input."""
import math
import torch
import torch.nn.functional as F

def strict_support(raw_valid,grid,in_crop,nearest_valid):
    bilinear=F.grid_sample(raw_valid.float(),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    strict=(bilinear>=.9999)&in_crop.bool()&(nearest_valid>.5)
    return strict,bilinear

def removal_summary(nn,strict,flow,photo_gradient,photo_map,support):
    nn=nn[:,0]>.5;strict=strict[:,0].bool();removed=nn&~strict
    energy=photo_gradient.detach().square().sum(dim=1);mag=flow.detach().square().sum(dim=1).sqrt()
    denominator=float(energy[nn].sum());total=float(energy.sum())
    n=int(nn.sum());count=int(removed.sum())
    rows=[]
    for lo,hi in zip((0,.1,.5,1,2,5),(.1,.5,1,2,5,math.inf)):
        mask=(mag>=lo)&(mag<hi)
        rows.append(dict(lower_px=lo,upper_px=hi if math.isfinite(hi) else None,NN_pixels=int((nn&mask).sum()),strict_pixels=int((strict&mask).sum()),removed_pixels=int((removed&mask).sum()),
            removed_photo_gradient_energy=float(energy[removed&mask].sum())))
    weights=support[:,0][removed]
    return dict(NN_valid_pixels=n,strict_valid_pixels=int(strict.sum()),removed_pixels=count,removed_fraction_of_NN_valid=count/n if n else None,
        removed_fraction_of_raster=count/nn.numel(),removed_photo_gradient_energy=float(energy[removed].sum()),
        NN_valid_photo_gradient_energy=denominator,all_photo_gradient_energy=total,
        removed_fraction_of_NN_valid_photo_gradient_energy=float(energy[removed].sum())/denominator if denominator else None,
        removed_fraction_of_all_photo_gradient_energy=float(energy[removed].sum())/total if total else None,
        removed_photo_loss_sum=float(photo_map[removed].sum()),NN_valid_photo_loss_sum=float(photo_map[nn].sum()),
        raw_valid_bilinear_weight_at_removed=dict(min=float(weights.min()),mean=float(weights.mean()),max=float(weights.max())) if count else None,
        motion_bins=rows,bin_label='original affine-interpolated GT magnitude in256-raster pixels; removed labels may be contaminated',
        interpretation='Removed means raw invalid/border interpolation support, not proof of physical occlusion or strict physical GT.')
