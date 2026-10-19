from pathlib import Path
import importlib.util,json,hashlib
import torch,sys
import torch.nn.functional as F

source=Path(sys.argv[1]) if len(sys.argv)>1 else Path('/home/zhaoyuhan/Eventbased_WAM/external/sources/downstream/event_cnn_minimal/utils/loss.py')
spec=importlib.util.spec_from_file_location('pinned_e2vid_loss',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)
H=W=112
yy,xx=torch.meshgrid(torch.arange(H),torch.arange(W),indexing='ij')
texture=.25+.5*((xx+yy)%2).float()[None,None]
image=torch.zeros_like(texture)
flow=torch.zeros((1,2,H,W))
old_loss,old_maps=m.temporal_consistency_loss(image,image,texture,texture,flow,output_images=True)
orig=F.grid_sample
def aligned_grid_sample(input,grid,**kwargs):return orig(input,grid,align_corners=True,**kwargs)
try:
 m.F.grid_sample=aligned_grid_sample
 corrected_loss,corrected_maps=m.temporal_consistency_loss(image,image,texture,texture,flow,output_images=True)
finally:m.F.grid_sample=orig
out={'scope':'CPU synthetic zero-flow identity probe; no model, labels or training modified',
 'torch':torch.__version__,'source_path':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'size':[H,W],'flow':'exactly zero','previous_and_current_reconstruction':'identical checkerboard',
 'visibility_images':'both all zeros so visibility mask is exactly one',
 'original':{'warp_max_abs_error':float((old_maps['processed0_warped_to1']-texture).abs().max()),'warp_mean_abs_error':float((old_maps['processed0_warped_to1']-texture).abs().mean()),'temporal_loss':float(old_loss)},
 'temporary_in_memory_align_corners_true_control':{'warp_max_abs_error':float((corrected_maps['processed0_warped_to1']-texture).abs().max()),'temporal_loss':float(corrected_loss)}}
print(json.dumps(out,indent=2))
if len(sys.argv)>2:Path(sys.argv[2]).write_text(json.dumps(out,indent=2))
