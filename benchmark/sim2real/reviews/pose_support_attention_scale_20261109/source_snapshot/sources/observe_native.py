"""Read-only hooks observe the unchanged native CenterScale forward."""
import torch


@torch.inference_mode()
def nominal_center_target(grid,gt_uv):
    """Original supervision heatmap, for posthoc interpretation only."""
    gt=torch.as_tensor(gt_uv,dtype=grid.dtype,device=grid.device)
    if grid.shape!=(256,2) or gt.ndim!=2 or gt.shape[1]!=2:raise ValueError('native grid and normalized GT centers required')
    dist=(grid[None]-gt[:,None,:2]).square().sum(-1)
    attention=(-.5*dist*16**2).softmax(1)
    uv=attention@grid
    return dict(attention=attention,uv=uv,peak_uv=grid[attention.argmax(1)],
                outside_grid_range=((gt<grid.min(0).values)|(gt>grid.max(0).values)).any(1))


@torch.inference_mode()
def observe_native(model,images,scale,intrinsics):
    if model.training or len(images)!=len(intrinsics):raise ValueError('eval model and aligned cameras required')
    captures={}
    def features_hook(_module,_inputs,output):
        if 'features' in captures:raise ValueError('features called twice unexpectedly')
        captures['features']=output.detach().clone()
    def head_pre_hook(_module,inputs):
        if 'head_input' in captures:raise ValueError('size head called twice unexpectedly')
        captures['head_input']=inputs[0].detach().clone()
    h1=model.features.register_forward_hook(features_hook)
    h2=model.size.register_forward_pre_hook(head_pre_hook)
    before=images.clone()
    try:logits,uv,log_size,log_shape=model(images)
    finally:h1.remove();h2.remove()
    if not torch.equal(before,images):raise ValueError('native model modified input')
    features=captures['features'].flatten(2)
    if features.shape[1:]!=(64,256) or model.grid.shape!=(256,2):raise ValueError('native64/16grid inventory changed')
    attention=logits.softmax(1)
    local=(features*attention[:,None]).sum(2);context=features.mean(2)
    if not torch.equal(uv,attention@model.grid):raise ValueError('original decoder parity failed')
    if not torch.equal(torch.cat((local,context),1),captures['head_input']):raise ValueError('native local/context reconstruction failed')
    focal=torch.tensor([scale.focal(K) for K in intrinsics],device=images.device)
    z=(log_shape+focal-log_size).exp()
    peak=model.grid[attention.argmax(1)]
    out=dict(features_map=features,attention=attention,uv=uv,peak_uv=peak,local=local,context=context,
             log_size=log_size,log_shape=log_shape,z=z,size=log_size.exp(),focal=focal)
    if not all(torch.isfinite(t).all() for t in out.values()):raise ValueError('nonfinite native activations')
    return out


@torch.inference_mode()
def hybrid_scale(model,features,attention,focal):
    """Cross-light diagnostic only. No GT inputs or altered parameters."""
    if model.training or features.shape[1:]!=(64,256) or attention.shape!=(len(features),256):raise ValueError('native aligned feature/attention required')
    if len(focal)!=len(features):raise ValueError('focal alignment required')
    local=(features*attention[:,None]).sum(2)
    context=features.mean(2)
    log_size,log_shape=(model.size(torch.cat((local,context),1))-2.3).unbind(1)
    z=(log_shape+focal-log_size).exp()
    if not all(torch.isfinite(t).all() for t in (log_size,log_shape,z)):raise ValueError('nonfinite hybrid head output')
    return dict(log_size=log_size,log_shape=log_shape,z=z)
