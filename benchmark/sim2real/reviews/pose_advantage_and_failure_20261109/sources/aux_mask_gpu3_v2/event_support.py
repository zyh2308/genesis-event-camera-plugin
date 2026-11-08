"""Synthetic-only visible-region/support diagnostics; no model/filter inputs.

Masks classify spatial REGIONS, not the physical cause of individual events.
An exact endpoint mask is not exact visibility throughout a 10-ms event window.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import numpy as np


def need(ok,message):
    if not ok:raise ValueError(message)


def file_record(path):
    p=Path(path);h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return dict(path=str(p),sha256=h.hexdigest(),size_bytes=p.stat().st_size)


def _array_digest(named_arrays):
    h=hashlib.sha256()
    for name,value in named_arrays:
        a=np.ascontiguousarray(value);h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


class RegionMask:
    __slots__=('_mask','_area','metadata')
    def __init__(self,mask,metadata):
        self.metadata=metadata
        if mask is None:self._mask=None;self._area=None
        else:
            value=np.array(mask,dtype=bool,copy=True);need(value.ndim==2,'Region must be2D')
            value.setflags(write=False);self._mask=value;self._area=int(value.sum())

    @property
    def mask(self):return self._mask

    @property
    def available(self):return self.mask is not None

    @property
    def area(self):return None if self.mask is None else self._area


def instance_region(seg,depth,target_id,expected_count,*,frame_index,timestamp_s,target_slot):
    seg=np.asarray(seg);depth=np.asarray(depth)
    need(seg.ndim==2 and depth.shape==seg.shape and np.issubdtype(seg.dtype,np.integer),'Bad physical segmentation/depth shape')
    mask=seg==target_id;count=int(mask.sum())
    need(count==int(expected_count),'Instance count differs from target_visible_pixels')
    need(np.isfinite(depth[mask]).all() and (depth[mask]>0).all(),'Invalid depth at target-visible pixel')
    return RegionMask(mask,dict(kind='physical_instance_endpoint_exact',visibility_exact_at_frame=True,
        event_time_visibility_exact=False,frame_index=int(frame_index),timestamp_s=float(timestamp_s),
        target_object_slot=int(target_slot),target_segmentation_id=int(target_id),
        target_visible_pixels=count,depth_validated=True,status='empty_mask' if count==0 else 'available'))


def endpoint_union(start,end):
    if not(start.available and end.available):return RegionMask(None,dict(kind='endpoint_union_approx',status='unavailable',visibility_exact_at_frame=False,event_time_visibility_exact=False))
    need(start.mask.shape==end.mask.shape,'Endpoint shapes differ')
    return RegionMask(start.mask|end.mask,dict(kind='endpoint_union_approx',visibility_exact_at_frame=False,event_time_visibility_exact=False,
        start_timestamp_s=start.metadata['timestamp_s'],end_timestamp_s=end.metadata['timestamp_s'],
        limitation='union of two exact endpoint masks; may miss intermediate motion/rotation; not a continuous swept mask',status='available'))


def resolve_episode(g):
    """Double proof: unique world-pose slot, then visible-pixel counts at ALL frames."""
    target=np.asarray(g['target_pose']);objects=np.asarray(g['object_to_world']);ids=np.asarray(g['object_segmentation_link_ids'])
    need(target.ndim==3 and target.shape[1:]==(4,4) and objects.shape==(len(target),len(ids),4,4),'Bad target/object poses')
    matches=[i for i in range(len(ids)) if np.array_equal(target,objects[:,i])]
    need(len(matches)==1 and len(np.unique(ids))==len(ids),'Target world pose must identify exactly one object/segmentation ID')
    slot=matches[0];target_id=int(ids[slot]);seg=np.asarray(g['segmentation_link_idx']);depth=np.asarray(g['depth_m']);visible=np.asarray(g['target_visible_pixels'])
    times=np.asarray(g['frame_timestamps_s']);need(seg.shape==depth.shape==(21,260,346) and times.shape==(21,),'Wrong physical dimensions/cadence')
    need(np.isfinite(times).all() and np.allclose(np.diff(times),.01,rtol=0,atol=1e-7),'Physical frame cadence changed')
    masks=[instance_region(seg[i],depth[i],target_id,visible[i],frame_index=i,timestamp_s=times[i],target_slot=slot) for i in range(21)]
    return masks,times,slot,target_id


def regions_for_rows(bank,rows,*,strict=False):
    """No window omission. Invalid episodes become explicit unavailable records.

    Set strict=True to fail the whole admission instead of retaining error rows.
    """
    import h5py
    result={};by_rid={}
    for row in rows:by_rid.setdefault(row['rid'],[]).append(row)
    for rid,items in by_rid.items():
        path=Path(bank)/'physical'/(rid+'.h5');source=dict(path=str(path),available=False)
        try:
            source=file_record(path)
            with h5py.File(path,'r') as f:
                g=f['episodes'][rid];masks,times,slot,target_id=resolve_episode(g)
                corners=np.asarray(g['projected_corners_uv']);anchors=np.asarray(g['projected_centroid_uv'])
                K=np.asarray(g['camera_intrinsics_K_px'])
                physical_sha=_array_digest((name,np.asarray(g[name])) for name in ('camera_intrinsics_K_px','camera_to_world_cv','cad_keypoints',
                    'frame_timestamps_s','target_pose','object_to_world','object_segmentation_link_ids','segmentation_link_idx','target_visible_pixels'))
                need(corners.shape==(21,8,2) and anchors.shape==(21,2) and np.isfinite(corners).all() and np.isfinite(anchors).all(),'Invalid projected geometry')
                for row in items:
                    need(row['id'] not in result,'Duplicate physical sample ID');k=int(row['id'].rsplit('/',1)[1]);need(4<=k<20 and row['id']==f'{rid}/{k:03d}','Window outside original physical ID inventory')
                    need(np.array_equal(row['K'],K),'Window K differs from physical input')
                    np.testing.assert_allclose(np.asarray(row['uv'][0]),anchors[k+1]/[346.,260.],rtol=0,atol=1e-7)
                    endpoint=masks[k+1];start=masks[k]
                    lo=corners[k+1].min(0);hi=corners[k+1].max(0);wh=hi-lo
                    result[row['id']]=dict(status='available',rid=rid,sample_id=row['id'],start_s=float(times[k]),end_s=float(times[k+1]),
                        startpoint=start,endpoint=endpoint,endpoint_union_approx=endpoint_union(start,endpoint),source_record=source,
                        physical_contract_sha256=physical_sha,
                        motion=dict(max_projected_corner_displacement_px=float(np.linalg.norm(corners[k+1]-corners[k],axis=1).max()),
                                    projected_anchor_displacement_px=float(np.linalg.norm(anchors[k+1]-anchors[k]))),
                        projected_bbox=dict(unclipped_xyxy=[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])],
                            geometric_size_px=float(np.sqrt(np.prod(wh))),geometric_size_normalized=float(np.sqrt(np.prod(wh/[346.,260.]))),
                            outside_corner_count=int(((corners[k+1]<0)|(corners[k+1]>=[346.,260.])).any(1).sum()),
                            interpretation='projection of eight3D bbox corners; not an exact amodal silhouette/occlusion fraction'),
                        geometric_sanity=geometry_sanity(endpoint,corners[k+1],anchors[k+1]),
                        target_identity=dict(object_slot=slot,segmentation_id=target_id,all21_pose_and_count_checks_passed=True))
        except Exception as error:
            if strict:raise
            for row in items:
                missing=RegionMask(None,dict(kind='unavailable',status='failed',reason=repr(error),visibility_exact_at_frame=False,event_time_visibility_exact=False))
                result[row['id']]=dict(status='failed',rid=rid,sample_id=row['id'],reason=repr(error),endpoint=missing,startpoint=missing,
                    endpoint_union_approx=missing,motion=None,source_record=source)
    need(len(result)==len(rows),'Region inventory changed');return result


def verify_matched_regions(reference,current):
    need(reference.keys()==current.keys(),'Cross-light physical sample IDs differ');checks=[]
    for sample_id,a in reference.items():
        b=current[sample_id];need(a['status']==b['status']=='available','Unavailable region prevents cross-light geometry admission')
        need(isinstance(a['physical_contract_sha256'],str) and len(a['physical_contract_sha256'])==64 and a['physical_contract_sha256']==b['physical_contract_sha256'],'Cross-light K/poses/keypoints/times/segmentation contract changed')
        need(a['start_s']==b['start_s'] and a['end_s']==b['end_s'],'Cross-light event window times differ')
        need(all(np.array_equal(a[k].mask,b[k].mask) for k in ('startpoint','endpoint')),'Cross-light visible target masks differ')
        checks.append(dict(sample_id=sample_id,passed=True))
    return dict(passed=True,checks=checks,scope='full21 physical geometry/timestamps/segmentation plus selected endpoint mask equality')


def region_geometry(region):
    if not region.available:return dict(available=False,area_px=None,area_fraction_native=None,bbox_xyxy=None,empty_mask=None,
        visible_mask_touches_border=None,visible_bbox_geom_size_px=None,visible_bbox_geom_size_normalized=None,visible_bbox_log_size_normalized=None,
        mask_kind=region.metadata['kind'],visibility_exact_at_frame=False,event_time_visibility_exact=False)
    mask=region.mask;h,w=mask.shape;ys,xs=np.nonzero(mask);empty=len(xs)==0
    bbox=None if empty else [int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    return dict(available=True,area_px=int(mask.sum()),area_fraction_native=float(mask.mean()),empty_mask=empty,bbox_xyxy=bbox,
        bbox_coordinate_convention='integer pixel cells; max-exclusive right/bottom',
        visible_mask_touches_border=False if empty else bool(mask[0].any() or mask[-1].any() or mask[:,0].any() or mask[:,-1].any()),
        visible_bbox_geom_size_px=None if empty else math.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])),
        visible_bbox_geom_size_normalized=None if empty else math.sqrt((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(w*h)),
        visible_bbox_log_size_normalized=None if empty else .5*math.log((bbox[2]-bbox[0])*(bbox[3]-bbox[1])/(w*h)),
        mask_kind=region.metadata['kind'],visibility_exact_at_frame=region.metadata['visibility_exact_at_frame'],event_time_visibility_exact=False)


def region_pyramid(region,*,encoder_size=416,input_size=128,attention_shape=(16,16)):
    """Diagnostic fractional occupancy using actual image-resize operations.

    This is a spatial weighting proxy, not a feature receptive-field mask.
    """
    import cv2,torch
    from torch.nn import functional as F
    if not region.available:return dict(available=False,region_fraction_native=None,region_fraction_attention=None,attention_fraction=None,attention16=None)
    native=region.mask.astype(np.float32);encoded=cv2.resize(native,(encoder_size,encoder_size),interpolation=cv2.INTER_LINEAR)
    x=torch.from_numpy(encoded)[None,None];inputs=F.interpolate(x,(input_size,input_size),mode='area')
    attention=F.interpolate(inputs,tuple(attention_shape),mode='area')[0,0].numpy()
    return dict(available=True,native_mask=region.mask,encoder_fraction=encoded,input_fraction=inputs[0,0].numpy(),attention_fraction=attention,attention16=attention,
        region_fraction_native=float(native.mean()),region_fraction_attention=float(attention.mean()),
        mapping='native bool -> OpenCV LINEAR416 -> Torch CPU AREA128 -> AREA attention grid; diagnostic proxy, not RF support')


def point_membership(region,normalized_uv):
    need(len(normalized_uv)==2 and np.isfinite(normalized_uv).all(),'Invalid normalized UV')
    if not region.available or region.area==0:return dict(available=False,in_frame=None,inside=None,reason='unavailable_or_empty_region')
    h,w=region.mask.shape;x,y=np.asarray(normalized_uv)*[w,h];inside=0<=x<w and 0<=y<h
    return dict(available=True,in_frame=bool(inside),inside=bool(region.mask[int(np.floor(y)),int(np.floor(x))]) if inside else False,
        sampling='floor(normalized_uv*[native_width,native_height]); no receptive-field interpretation')


def geometry_sanity(region,projected_corners_uv,anchor_uv_px):
    """Only reports; projected bbox hull NEVER replaces the instance mask."""
    import cv2
    if not region.available:return dict(available=False)
    corners=np.asarray(projected_corners_uv,dtype=np.float64);anchor=np.asarray(anchor_uv_px,dtype=np.float64)
    need(corners.shape==(8,2) and anchor.shape==(2,) and np.isfinite(corners).all() and np.isfinite(anchor).all(),'Bad geometry sanity input')
    hull=cv2.convexHull(corners.astype(np.float32));raster=np.zeros(region.mask.shape,np.uint8)
    need(np.abs(hull).max()*256<np.iinfo(np.int32).max,'Projected hull exceeds safe raster range')
    cv2.fillConvexPoly(raster,np.rint(hull*256).astype(np.int32),1,lineType=cv2.LINE_8,shift=8)
    hmask=raster.astype(bool);visible=region.mask;area=int(visible.sum());ha=int(hmask.sum());intersection=int((visible&hmask).sum())
    dil_h=cv2.dilate(raster,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    dil_v=cv2.dilate(visible.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    ys,xs=np.nonzero(visible);centroid=None if not area else [float(xs.mean()),float(ys.mean())]
    delta=None if centroid is None else (anchor-np.asarray(centroid))
    height,width=visible.shape
    return dict(available=True,projection_kind='projected_bbox_hull_approx',rasterization='OpenCV convex hull;1/256pixel fixed-point vertices;LINE_8;onepixel Chebyshev tolerance reported separately',
        visible_area_px=area,projected_bbox_hull_area_px=ha,intersection_px=intersection,
        visible_covered_by_hull_fraction=intersection/area if area else None,
        hull_covered_by_visible_fraction=intersection/ha if ha else None,visible_to_hull_area_ratio=area/ha if ha else None,
        visible_outside_hull_px=int((visible&~hmask).sum()),visible_outside_hull_dilated1px=int((visible&~dil_h).sum()),
        hull_outside_visible_px=int((hmask&~visible).sum()),hull_outside_visible_dilated1px=int((hmask&~dil_v).sum()),
        GT_anchor_uv_px=anchor.tolist(),GT_anchor_visible_membership=point_membership(region,anchor/[width,height]),
        GT_anchor_inside_projected_bbox_hull=bool(cv2.pointPolygonTest(hull,tuple(map(float,anchor)),False)>=0),
        visible_mask_centroid_pixel_index_xy=centroid,GT_anchor_minus_visible_centroid_px=None if delta is None else delta.tolist(),
        GT_anchor_to_visible_centroid_distance_px=None if delta is None else float(np.linalg.norm(delta)),
        interpretation='bbox-corner hull is not exact CAD silhouette; area ratios are NOT occlusion rates; inside-object attention does not imply correct anchor center')


def summarize_attention(region,attention16,pred_uv,peak_uv=None,*,pyramid=None):
    a=np.asarray(attention16,dtype=np.float64);need(a.ndim==2 and np.isfinite(a).all() and (a>=0).all() and abs(float(a.sum())-1)<1e-5,'Original normalized attention required')
    h,w=a.shape
    if peak_uv is None:
        i=int(a.argmax());peak_uv=[(i%w+.5)/w,(i//w+.5)/h]
    if pyramid is None:pyramid=region_pyramid(region,attention_shape=a.shape)
    else:
        need(pyramid['available']==region.available,'Cached pyramid availability differs')
        if region.available:
            need(pyramid['attention16'].shape==a.shape and (pyramid['native_mask'] is region.mask or np.array_equal(pyramid['native_mask'],region.mask)),'Cached pyramid belongs to another region/grid')
    area=pyramid['region_fraction_attention']
    available=region.available and region.area>0 and area is not None and area>0
    mass=float((a*pyramid['attention_fraction']).sum()) if available else None
    pred=point_membership(region,pred_uv);peak=point_membership(region,peak_uv)
    return dict(available=available,region_attention_mass=mass,region_fraction_native=pyramid['region_fraction_native'],region_grid_area_fraction=area,
        region_mass_enrichment=mass/area if available else None,prediction_in_region=pred['inside'],peak_in_region=peak['inside'],
        predicted_center_membership=pred,peak_membership=peak,
        interpretation='spatial attention weights vs visible-region proxy, not feature RF containment or center accuracy')


def causal_features(timestamp_s,x,y,width=346,height=260):
    """Past RAW inputs; rejected inputs also support future inputs. No masks/GT."""
    t=np.asarray(timestamp_s,dtype=np.float64)*1e6;x=np.asarray(x);y=np.asarray(y)
    need(t.ndim==1 and x.shape==y.shape==t.shape and np.isfinite(t).all() and (t>=0).all() and (np.diff(t)>=0).all(),'Invalid chronological event arrays')
    need(np.issubdtype(x.dtype,np.integer) and np.issubdtype(y.dtype,np.integer) and ((x>=0)&(x<width)&(y>=0)&(y<height)).all(),'Invalid event coordinates')
    history=[-math.inf]*(width*height);ages=np.full(len(t),np.inf);same=np.full(len(t),np.inf);keep=np.zeros(len(t),bool);startup=np.zeros(len(t),bool)
    for i,(now,xx,yy) in enumerate(zip(t,x,y)):
        xx,yy=int(xx),int(yy);p=yy*width+xx;last=-math.inf;unseen=False
        for by in range(max(0,yy-1),min(height,yy+2)):
            for bx in range(max(0,xx-1),min(width,xx+2)):
                if bx==xx and by==yy:continue
                v=history[by*width+bx];unseen|=v==-math.inf;last=max(last,v)
        ages[i]=now-last;same[i]=now-history[p];real=ages[i]<2000.;startup[i]=not real and unseen and now<2000.;keep[i]=real or startup[i];history[p]=float(now)
    return dict(neighbour8_age_us=ages,same_pixel_interval_us=same,baf_keep=keep,startup_only_keep=startup,
        definition='closest prior INPUT at8 neighbours; no polarity, strict2ms, all inputs update; float64 microseconds without rounding')


def _quantiles(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return None if len(values)==0 else dict(zip(('p50','p90','p99'),map(float,np.quantile(values,[.5,.9,.99]))))


def _partition_stats(mask,selected,kept,x,y,temporal):
    area=int(mask.sum());raw_idx=selected&mask[y,x];keep_idx=raw_idx&kept;n=int(raw_idx.sum());k=int(keep_idx.sum())
    raw_map=np.zeros(mask.shape,np.int64);kept_map=np.zeros_like(raw_map)
    np.add.at(raw_map,(y[raw_idx],x[raw_idx]),1);np.add.at(kept_map,(y[keep_idx],x[keep_idx]),1)
    occupied=raw_map>0;padded=np.pad(occupied.astype(np.int8),1);neighbours=np.zeros_like(raw_map)
    full_map=np.zeros_like(raw_map);np.add.at(full_map,(y[selected],x[selected]),1);full_pad=np.pad((full_map>0).astype(np.int8),1);all_neighbours=np.zeros_like(raw_map)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            if dx or dy:
                neighbours+=padded[1+dy:1+dy+mask.shape[0],1+dx:1+dx+mask.shape[1]]
                all_neighbours+=full_pad[1+dy:1+dy+mask.shape[0],1+dx:1+dx+mask.shape[1]]
    values=all_neighbours[y[raw_idx],x[raw_idx]];same_region=neighbours[y[raw_idx],x[raw_idx]]
    return dict(region_pixels=area,raw_events=n,baf_events=k,event_retention_ratio=k/n if n else None,
        raw_occupied_pixels=int(occupied.sum()),baf_occupied_pixels=int((kept_map>0).sum()),
        raw_occupancy_ratio=float(occupied.sum()/area) if area else None,baf_occupancy_ratio=float((kept_map>0).sum()/area) if area else None,
        raw_events_per_region_pixel=n/area if area else None,baf_events_per_region_pixel=k/area if area else None,
        raw_same_window_spatially_isolated_event_fraction=float((values==0).mean()) if n else None,
        raw_same_window_occupied_neighbour8_mean=float(values.mean()) if n else None,
        raw_same_window_same_region_occupied_neighbour8_mean=float(same_region.mean()) if n else None,
        raw_prior_neighbour8_within2ms_fraction=float((temporal['neighbour8_age_us'][raw_idx]<2000).mean()) if n else None,
        raw_prior_neighbour8_no_history_fraction=float(np.isinf(temporal['neighbour8_age_us'][raw_idx]).mean()) if n else None,
        raw_same_pixel_no_history_fraction=float(np.isinf(temporal['same_pixel_interval_us'][raw_idx]).mean()) if n else None,
        raw_prior_neighbour8_age_us=_quantiles(temporal['neighbour8_age_us'][raw_idx]),
        raw_same_pixel_interval_us=_quantiles(temporal['same_pixel_interval_us'][raw_idx]),startup_only_retained_events=int(temporal['startup_only_keep'][raw_idx].sum()))


def window_support(t,x,y,keep,start_s,end_s,region,temporal):
    selected=(t>start_s)&(t<=end_s);n=int(selected.sum())
    if not region.available:return dict(available=False,total_raw_events=n,total_baf_events=int((selected&keep).sum()),empty_window=n==0,foreground=None,background_region=None)
    return dict(available=True,total_raw_events=n,total_baf_events=int((selected&keep).sum()),empty_window=n==0,
        mask_kind=region.metadata['kind'],event_membership='event XY inside this spatial endpoint/union region; not per-event physical-source GT',
        foreground=_partition_stats(region.mask,selected,keep,x,y,temporal),background_region=_partition_stats(~region.mask,selected,keep,x,y,temporal))


def window_support_for_rows(bank,rows,method,baf_module,regions=None):
    """One independent read-only episode pass, shared across every model/seed."""
    bank=Path(bank);regions=regions_for_rows(bank,rows) if regions is None else regions;manifest_path=bank/'events'/method/'manifest.json'
    manifest=json.loads(manifest_path.read_text());items={r['rollout_id']:r for r in manifest['rollouts']};out={};sources=[file_record(manifest_path)]
    for rid in sorted({r['rid'] for r in rows}):
        path=bank/'events'/method/(rid+'.npy');r=file_record(path);need(r['sha256']==items[rid]['events_sha256'],'Raw event source hash differs');sources.append(r)
        raw=np.load(path,allow_pickle=False);t=raw[:,0];x=raw[:,1].astype(int);y=raw[:,2].astype(int);p=(raw[:,3]>0).astype(np.uint8)
        temporal=causal_features(t,x,y);keep=baf_module.BackgroundActivityFilter(346,260).process(t,x,y,p)
        need(np.array_equal(keep,temporal['baf_keep']),'Independent temporal support does not reproduce frozen BAF')
        for row in (r for r in rows if r['rid']==rid):
            rr=regions[row['id']]
            if rr['status']!='available':
                out[row['id']]=dict(status='region_unavailable',sample_id=row['id'],rid=rid,reason=rr['reason']);continue
            out[row['id']]=dict(status='complete',sample_id=row['id'],rid=rid,start_s=rr['start_s'],end_s=rr['end_s'],
                regions={name:window_support(t,x,y,keep,rr['start_s'],rr['end_s'],rr[name],temporal) for name in ('startpoint','endpoint','endpoint_union_approx')})
    need(len(out)==len(rows),'Event-support inventory changed');return out,sources
