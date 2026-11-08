"""Posthoc soft-region centroid algebra; no forward, training or corrected score."""
import argparse,csv,hashlib,json,math,os
from collections import defaultdict
from pathlib import Path
import numpy as np
KEYS=('method','seed','arm','branch','input_view','condition','rid','sample_id')
VERIFIER_SOURCE='2ed9d3da4482798d1f6537e242bd5f58949a0d3ebadd2b3b940391422a1d963d'
def need(x,m):
    if not x:raise ValueError(m)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,allow_nan=False,indent=2)
def artifact(c,name):
    r=next(r for r in c['artifacts'] if Path(r['path']).name==name);p=Path(r['path']);need(p.stat().st_size==r['size_bytes'] and sha(p)==r['sha256'],'Artifact changed: '+name);return p
def fraction(mask):
    import cv2,torch
    x=torch.from_numpy(cv2.resize(mask.astype(np.float32),(416,416),interpolation=cv2.INTER_LINEAR))[None,None]
    return torch.nn.functional.interpolate(torch.nn.functional.interpolate(x,(128,128),mode='area'),(16,16),mode='area')[0,0].numpy()
def decompose(attention,region,gt):
    p=np.asarray(attention,np.float64).reshape(-1);f=np.asarray(region,np.float64).reshape(-1);g=np.asarray(gt,np.float64)
    need(p.shape==f.shape==(256,) and g.shape==(2,) and np.isfinite([*p,*f,*g]).all() and (p>=0).all() and ((f>=0)&(f<=1)).all() and p.sum()>0,'Invalid fixed map/region/GT')
    total=float(p.sum());p=p/total;yy,xx=np.meshgrid(np.arange(16),np.arange(16),indexing='ij');grid=np.stack(((xx.ravel()+.5)/16*346,(yy.ravel()+.5)/16*260),1)
    m=float(p@f);b=float(p@(1-f));u=p@grid;e=u-g;length=float(np.linalg.norm(e));direction=e/length if length>0 else None
    uf=(p*f)@grid/m if m>0 else None;ub=(p*(1-f))@grid/b if b>0 else None;ef=uf-g if uf is not None else None;tail=b*(ub-uf) if uf is not None and ub is not None else None
    residual=float(np.max(np.abs(e-ef-tail))) if tail is not None else None
    if residual is not None:need(residual<=1e-10,'Native-pixel vector decomposition identity failed')
    proj=lambda v:float(v@direction) if v is not None and direction is not None else None
    pf,pt=proj(ef),proj(tail)
    if pf is not None and pt is not None:need(abs(pf+pt-length)<=1e-10,'Signed projection identity failed')
    return dict(original_attention_sum=total,probability_normalization='FP64 p/sum(p) for algebra only',mass_fg=m,mass_bg=b,map_centroid_native_px=u.tolist(),GT_anchor_native_px=g.tolist(),e_total_native_px=e.tolist(),e_fg_native_px=None if ef is None else ef.tolist(),outside_tail_native_px=None if tail is None else tail.tolist(),signed_foreground_projection_px=pf,signed_outside_tail_projection_px=pt,total_error_length_px=length,identity_residual_px=residual,diagnostic_only=True,deployable=False)
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for x in ('verifier-completion','full-completion','out'):p.add_argument('--'+x,type=Path,required=True)
    for x in ('verifier-sha256','full-sha256'):p.add_argument('--'+x,required=True)
    a=p.parse_args();need(a.out.is_absolute() and a.out.parent.is_dir() and not a.out.exists(),'Unique absolute output required');a.out.mkdir();os.environ['CUDA_VISIBLE_DEVICES']=''
    import cv2,torch
    cv2.setNumThreads(1);torch.set_num_threads(2)
    need(sha(a.verifier_completion)==a.verifier_sha256 and sha(a.full_completion)==a.full_sha256,'Input completion SHA mismatch');v,c=read(a.verifier_completion),read(a.full_completion)
    need(v['schema']=='independent_attention_k12_verifier_v1' and v['status']=='passed' and v['source_manifest_sha256']==VERIFIER_SOURCE and v['all_model_and_source_unchanged'] is True and v['verified_attention_maps']==v['attention_maps']==576 and v['sentinel_k']==12 and v['failed_cases']==0 and v['optimizer_updates']==0 and v['real_data_used'] is False and v['full_completion_sha256']==a.full_sha256,'Actual fixed independent576 verifier required')
    need(c['status']=='complete' and c['mode']=='full' and c['diagnostic_rows']==9216 and c['optimizer_updates']==0 and c['real_data_used'] is False,'Actual complete frozen full diagnostic required')
    mp=artifact(v,'attention_maps_k12.npz');ip=artifact(v,'attention_map_index.json');rp=artifact(v,'native_region_masks_deduplicated.npz');ri=artifact(v,'native_region_mask_index.json');fp=artifact(c,'per_window.jsonl')
    inputs=[a.verifier_completion,a.full_completion,mp,ip,rp,ri,fp];before={str(x):sha(x) for x in inputs}
    maps=np.load(mp,allow_pickle=False)['attention'];index=read(ip);masks=np.load(rp,allow_pickle=False);regions={}
    for r in read(ri):
        if r['kind'] not in ('endpoint','endpoint_union_approx'):continue
        key=(r['method'],r['condition'],r['sample_id'],r['kind']);need(key not in regions,'Duplicate mask reference')
        mask=masks['m_'+r['sha256']];need(mask.shape==(260,346) and mask.dtype==np.bool_ and hashlib.sha256(mask.tobytes()).hexdigest()==r['sha256'],'Native mask SHA changed');regions[key]=fraction(mask)
    refs={}
    for line in fp.open():
        r=json.loads(line)
        if int(r['sample_id'].rsplit('/',1)[1])==12:
            key=tuple(r[k] for k in KEYS);need(key not in refs,'Duplicate full sentinel record');refs[key]=r
    need(maps.shape==(576,16,16) and len(index)==len(refs)==576 and {tuple(r[k] for k in KEYS) for r in index}==set(refs),'Sentinel cohort changed');rows=[]
    for i,r in enumerate(index):
        need(r['index']==i,'Map array/index order changed');old=refs[tuple(r[k] for k in KEYS)];g=old['geometric_sanity']['GT_anchor_uv_px'];parts={}
        for region in ('endpoint','endpoint_union_approx'):
            d=decompose(maps[i],regions[r['method'],r['condition'],r['sample_id'],region],g);need(np.max(np.abs(np.asarray(d['map_centroid_native_px'])/[346,260]-old['pred_uv']))<=2e-6,'Map centroid differs from full UV beyond original tolerance')
            d.update(peak_in_region=old['attention'][region]['peak_in_region'],native_UV_in_region=old['attention'][region]['prediction_in_region']);parts[region]=d
        rows.append({k:r[k] for k in KEYS}|dict(regions=parts))
    groups=defaultdict(list)
    for r in rows:groups[tuple(r[k] for k in KEYS if k!='seed')].append(r)
    table=[]
    for key,items in sorted(groups.items()):
        need(sorted(r['seed'] for r in items)==[0,1,2],'All three seeds required');row=dict(zip((k for k in KEYS if k!='seed'),key));row['seeds']='0|1|2'
        for region in ('endpoint','endpoint_union_approx'):
            ds=[r['regions'][region] for r in items]
            for metric in ('mass_fg','mass_bg','signed_foreground_projection_px','signed_outside_tail_projection_px'):
                values=[d[metric] for d in ds if d[metric] is not None];prefix=region+'_'+metric;row[prefix+'_mean']=float(np.mean(values)) if values else None;row[prefix+'_available_seeds']=len(values)
            row[region+'_peak_available_seeds']=sum(d['peak_in_region'] is not None for d in ds);row[region+'_peak_fg_cases']=sum(d['peak_in_region'] is True for d in ds);row[region+'_peak_fg_positive_tail_cases']=sum(d['peak_in_region'] is True and d['signed_outside_tail_projection_px'] is not None and d['signed_outside_tail_projection_px']>0 for d in ds)
        table.append(row)
    need(len(rows)==576 and len(table)==192,'Fixed per-case/three-seed-per-rid inventory changed')
    with (a.out/'per_case_decomposition.jsonl').open('x') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    write(a.out/'three_seed_per_rid.json',table)
    with (a.out/'three_seed_per_rid.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    need(before[str(a.verifier_completion)]==a.verifier_sha256 and before[str(a.full_completion)]==a.full_sha256,'Completion changed during decomposition')
    need(all(sha(x)==h for x,h in before.items()),'Input bytes changed');write(a.out/'completion.json',dict(schema='posthoc_attention_tail_decomposition_v1',status='complete',cases=576,three_seed_per_rid_rows=192,regions=['endpoint','endpoint_union_approx'],optimizer_updates=0,model_forward_passes=0,GPU_used=False,real_data_used=False,GT_regions_posthoc_only=True,decomposition_is_deployable=False,GT_definition='physical endpoint projected anchor native pixels; not a new GT-corrected model score',original_raw_maps_sha256=sha(mp),input_sha256=before,source_script_sha256=sha(__file__)))
if __name__=='__main__':main()
