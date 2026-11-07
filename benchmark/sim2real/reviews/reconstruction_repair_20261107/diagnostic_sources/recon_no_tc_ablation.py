"""Causal ablation: the identical fixed-batch control with temporal weight zero."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
import recon_short_diagnostic as original

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    expected='583e4d6e538228f506d84fca4b5a586206a4275a51a75b41f9fa6ec1c232a0f4'
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()==expected
    torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.device_count()==1
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    args.out.mkdir(parents=True,exist_ok=False);context=original.setup(args.campaign);factory=context.shim_for
    def without_tc(model,arm):
        shim=factory(model,arm)
        for loss in shim.loss_ftns:
            if loss.__class__.__name__=='temporal_consistency_loss':loss.weight=0.
        return shim
    context.shim_for=without_tc
    original.save(args.out/'protocol.json',dict(real_data_used=False,optimizer_updates_per_arm=20,
        methods=['eventgenesis','v2e'],seeds=[2],fixed_train_batch=True,only_intervention='temporal weight 2 -> 0',
        original_source_sha256=expected,shared_initialization=True,shared_window_and_augmentation=True,
        purpose='causal diagnostic only; disabling temporal consistency is not an adopted repair'))
    results=[]
    for method in ['eventgenesis','v2e']:
        result=original.short_fit(context,args.campaign,args.out,method,2,'linear_no_tc',20,True,40)
        results.append(result);original.save(args.out/'partial_results.json',results)
    original.save(args.out/'completion.json',dict(status='complete',real_data_used=False,results=results))
if __name__=='__main__':main()
