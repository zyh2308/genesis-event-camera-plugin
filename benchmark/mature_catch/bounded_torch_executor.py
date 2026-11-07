"""Memory scheduling for the unchanged public temporal-batch warp kernel.

All original alphas, endpoint data, geometry, thresholds and sample counts
are retained. This adapter only groups independent temporal samples.
"""
def make_bounded(original,temporal_chunk=2):
    import torch
    def bounded(*args,**kwargs):
        count=args[2].shape[1]
        if count<=temporal_chunk:return original(*args,**kwargs)
        pieces=[]
        for start in range(0,count,temporal_chunk):
            end=min(start+temporal_chunk,count);part=list(args)
            for index in (2,3,4,7,8,9):part[index]=args[index][:,start:end]
            options=dict(kwargs)
            alpha=options.get('alpha')
            if alpha is not None and getattr(alpha,'ndim',0)>0:options['alpha']=alpha[start:end]
            pieces.append(original(*part,**options))
        output=[torch.cat([p[i] for p in pieces],dim=1) for i in range(2)]
        if len(pieces[0])==3:
            output.append({key:torch.cat([p[2][key] for p in pieces],dim=1) for key in pieces[0][2]})
        return tuple(output)
    return bounded

def install(temporal_chunk=2):
    from genesis_event_plugin import torch_physics
    original=torch_physics.depth_aware_bidirectional_warp_projected_batch_torch
    bounded=make_bounded(original,temporal_chunk)
    torch_physics.depth_aware_bidirectional_warp_projected_batch_torch=bounded
    # The interpolator imported this symbol when its module was loaded.
    from genesis_event_plugin import physics_interpolator
    physics_interpolator.depth_aware_bidirectional_warp_projected_batch_torch=bounded
    return original
