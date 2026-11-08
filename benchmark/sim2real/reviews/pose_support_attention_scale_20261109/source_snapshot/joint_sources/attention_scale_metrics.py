"""Pure NumPy summaries; GT regions are evaluation-only and never model input."""
import math
from collections import defaultdict
import numpy as np

PIXELS=np.array([346.,260.])


def attention_statistics(attention,region_fraction,uv,uv_gt,prediction_in_region,peak_in_region):
    a=np.asarray(attention,dtype=np.float64).reshape(-1)
    region=np.asarray(region_fraction,dtype=np.float64).reshape(-1)
    if a.shape!=region.shape or len(a)!=256 or not np.isfinite(a).all() or not np.isfinite(region).all():
        raise ValueError('finite aligned native 16x16 attention and region fractions required')
    if min(a)<0 or min(region)<0 or max(region)>1 or not np.isclose(a.sum(),1,atol=1e-6,rtol=0):
        raise ValueError('invalid probability/region fraction domain')
    area=float(region.mean());mass=float(a@region) if area>0 else None
    entropy=-float(np.sum(a*np.log(np.maximum(a,1e-300))))
    p,q=np.asarray(uv,dtype=float),np.asarray(uv_gt,dtype=float)
    if p.shape!=(2,) or q.shape!=(2,) or not np.isfinite([*p,*q]).all():raise ValueError('finite normalized center pairs required')
    return dict(region_attention_mass=mass,region_grid_area_fraction=area,
                region_mass_enrichment=mass/area if area>0 else None,
                region_available=bool(area>0),prediction_in_region=prediction_in_region if area>0 else None,
                peak_in_region=peak_in_region if area>0 else None,center_error_px=float(np.linalg.norm((p-q)*PIXELS)),
                attention_entropy=entropy,attention_normalized_entropy=entropy/math.log(len(a)),
                peak_probability=float(a.max()))


def feature_statistics(local,context):
    x,y=np.asarray(local,dtype=float),np.asarray(context,dtype=float)
    if x.shape!=(64,) or y.shape!=(64,) or not np.isfinite([*x,*y]).all():raise ValueError('two finite native 64D vectors required')
    nx,ny=float(np.linalg.norm(x)),float(np.linalg.norm(y))
    return dict(local_l2=nx,context_l2=ny,local_context_cosine=float(x@y/(nx*ny)) if nx>0 and ny>0 else None,
                local_feature=x.tolist(),context_feature=y.tolist())


def scale_statistics(log_size,log_shape,focal_log,gt_log_size,gt_z_m,pred_z_m):
    values=np.asarray([log_size,log_shape,focal_log,gt_log_size,gt_z_m,pred_z_m],float)
    if not np.isfinite(values).all() or gt_z_m<=0 or pred_z_m<=0:raise ValueError('finite physical scale/depth required')
    gt_shape=math.log(gt_z_m)-focal_log+gt_log_size
    e_size=float(log_size-gt_log_size);e_shape=float(log_shape-gt_shape)
    e_logz=float(log_shape+focal_log-log_size-math.log(gt_z_m))
    if not np.isclose(e_logz,e_shape-e_size,atol=2e-14,rtol=0):raise ValueError('log-depth component identity failed')
    return dict(pred_log_size=float(log_size),pred_log_shape=float(log_shape),focal_log=float(focal_log),
                gt_log_size=float(gt_log_size),gt_log_shape=float(gt_shape),gt_z_m=float(gt_z_m),pred_z_m=float(pred_z_m),
                signed_log_size_error=e_size,signed_log_shape_error=e_shape,signed_log_depth_error=e_logz,
                signed_depth_error_m=float(pred_z_m-gt_z_m),size_shape_error_product=e_size*e_shape,
                log_depth_error_square=e_logz*e_logz,
                size_shape_identity_residual=e_logz-(e_shape-e_size))


def _feature_delta(first,second,key):
    x,y=np.asarray(first[key],float),np.asarray(second[key],float)
    nx,ny=float(np.linalg.norm(x)),float(np.linalg.norm(y))
    return dict(l2_delta=float(np.linalg.norm(y-x)),norm_ratio=ny/nx if nx>0 else None,
                cosine=float(x@y/(nx*ny)) if nx>0 and ny>0 else None)


def paired_light_deltas(rows):
    """Same-model, same-physical-window normal->half/eighth. No pooled matching."""
    groups=defaultdict(dict)
    keys=('method','seed','arm','branch','input_view','rid','sample_id')
    for row in rows:
        key=tuple(row[k] for k in keys)
        if row['condition'] in groups[key]:raise ValueError('duplicate cross-light record')
        groups[key][row['condition']]=row
    output=[]
    for key,lights in sorted(groups.items()):
        if set(lights)!={'normal','half','eighth'}:raise ValueError('all three matched illuminations required')
        baseline=lights['normal']
        for condition in ('half','eighth'):
            changed=lights[condition]
            for label in ('gt_log_size','gt_z_m','focal_log'):
                if baseline['labels'][label]!=changed['labels'][label]:raise ValueError('physical geometry changed across illumination')
            mass0=baseline['attention']['endpoint']['region_attention_mass'];mass1=changed['attention']['endpoint']['region_attention_mass']
            out=dict(zip(keys,key),condition=condition,reference_condition='normal',
                     delta_center_error_px=changed['attention']['endpoint']['center_error_px']-baseline['attention']['endpoint']['center_error_px'],
                     delta_region_attention_mass=mass1-mass0 if mass0 is not None and mass1 is not None else None,
                     feature_local=_feature_delta(baseline['features'],changed['features'],'local_feature'),
                     feature_context=_feature_delta(baseline['features'],changed['features'],'context_feature'))
            if baseline['branch']=='depth':
                base_s,cur_s=baseline['scale'],changed['scale']
                ds=cur_s['pred_log_size']-base_s['pred_log_size'];dh=cur_s['pred_log_shape']-base_s['pred_log_shape']
                out.update(delta_log_size=ds,delta_log_shape=dh,delta_log_depth=dh-ds,
                           delta_signed_depth_error_m=cur_s['signed_depth_error_m']-base_s['signed_depth_error_m'])
            output.append(out)
    return output


def scalar_rollout_summaries(rows):
    """All descriptive moments retain rollout/seed identities and zero windows."""
    groups=defaultdict(lambda:defaultdict(list))
    keys=('method','seed','arm','branch','input_view','condition')
    for row in rows:groups[tuple(row[k] for k in keys)][row['rid']].append(row)
    out=[]
    for key,rollouts in sorted(groups.items()):
        by={}
        for rid,items in sorted(rollouts.items()):
            vals=({name:np.asarray([r['scale'][name] for r in items],float) for name in
                  ('signed_log_size_error','signed_log_shape_error','signed_log_depth_error','signed_depth_error_m')}
                  if key[3]=='depth' else {})
            stats={}
            for name,x in vals.items():
                stats[name]=dict(bias=float(x.mean()),mae=float(np.abs(x).mean()),centered_scatter=float(x.std(ddof=0)),rmse=float(np.sqrt((x*x).mean())))
            if key[3]=='depth':
                stats['size_shape_error_product_mean']=float(np.mean([r['scale']['size_shape_error_product'] for r in items]))
                stats['log_depth_error_square_mean']=float(np.mean([r['scale']['log_depth_error_square'] for r in items]))
            for name,field in (('endpoint_attention_mass_mean','region_attention_mass'),('endpoint_peak_hit_rate','peak_in_region'),('endpoint_uv_hit_rate','prediction_in_region')):
                values=[r['attention']['endpoint'][field] for r in items if r['attention']['endpoint'][field] is not None]
                stats[name]=float(np.mean(values)) if values else None
                stats[name+'_available_count']=len(values)
            stats['center_error_px']=float(np.mean([r['attention']['endpoint']['center_error_px'] for r in items]))
            by[rid]=dict(windows=len(items),metrics=stats)
        def flatten(value,prefix=''):
            d={}
            for k,v in value.items():
                if isinstance(v,dict):d.update(flatten(v,prefix+k+'/'))
                else:d[prefix+k]=v
            return d
        flat=[flatten(v['metrics']) for v in by.values()]
        summary={name:float(np.mean([v[name] for v in flat])) if all(v[name] is not None for v in flat) else None for name in flat[0]}
        out.append(dict(zip(keys,key),rollout_count=len(by),window_count=sum(v['windows'] for v in by.values()),
                        equal_rollout_summary=summary,by_rollout=by,
                        scatter_note='population descriptive scatter within each rollout, then equal rollout mean; not an independent-window confidence interval'))
    return out


def hybrid_summaries(rows):
    """Four RAW feature/attention path cells, rollout-equal then seed-equal."""
    groups=defaultdict(lambda:defaultdict(list));seen=set()
    keys=('method','seed','arm','paired_light','feature_source','attention_source')
    for row in rows:
        key=tuple(row[k] for k in keys);identity=key+(row['rid'],row['sample_id'])
        if identity in seen:raise ValueError('duplicate hybrid physical window')
        seen.add(identity);groups[key][row['rid']].append(row)
    perseed=[]
    for key,rollouts in sorted(groups.items()):
        by={}
        for rid,items in sorted(rollouts.items()):
            z=np.asarray([r['z_m']-r['gt_z_m'] for r in items],float)
            size=np.asarray([r['log_size']-r['gt_log_size'] for r in items],float)
            shape=np.asarray([r['log_shape']-r['gt_log_shape'] for r in items],float)
            by[rid]=dict(depth_mae_m=float(np.abs(z).mean()),depth_bias_m=float(z.mean()),
                         log_size_mae=float(np.abs(size).mean()),log_size_bias=float(size.mean()),
                         log_shape_mae=float(np.abs(shape).mean()),log_shape_bias=float(shape.mean()),window_count=len(items))
        names=set(next(iter(by.values())))-{'window_count'}
        summary={name:float(np.mean([v[name] for v in by.values()])) for name in sorted(names)}
        perseed.append(dict(zip(keys,key),summary=summary,by_rollout=by,rollout_count=len(by),window_count=sum(len(v) for v in rollouts.values())))
    grouped=defaultdict(list)
    for row in perseed:grouped[tuple(row[k] for k in keys if k!='seed')].append(row)
    means=[]
    for key,cells in sorted(grouped.items()):
        seeds=sorted(r['seed'] for r in cells)
        if len(seeds)!=len(set(seeds)):raise ValueError('duplicate hybrid seed')
        mean={name:float(np.mean([r['summary'][name] for r in cells])) for name in cells[0]['summary']}
        means.append(dict(zip((k for k in keys if k!='seed'),key),seeds=seeds,metrics=mean,diagnostic_only=True,deployable=False))
    return dict(per_seed=perseed,three_seed_means=means)
