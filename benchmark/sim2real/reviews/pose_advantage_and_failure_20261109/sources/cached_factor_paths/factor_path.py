"""Frozen synthetic GT-factor diagnostic: cached rotations only, NO forward/training.

Every GT substitution is nondeployable. Center changes can alter both the
anchor ray and the previously exported rotation crop; predicted crop size stays
fixed. This is a conditional path diagnostic, not a unique causal partition.
"""
from __future__ import annotations
import argparse, copy, csv, hashlib, importlib.util, itertools, json, sys, traceback
from pathlib import Path
import numpy as np

PINS = {
 'helper':'ba4c13e5dc8ee3f8de3b7f5cbeb970fad4e31bdea2a49f2f15d9b612d01f9344',
 'predictions':'2405d6c59edefb70bb4c3e5a3025ccf19d7f6d9ed3cfb8838fafb9975e24ab02',
 'vertices':'8e0d1e9b27d63f03ecb3f960aa3140ebb9ad420c6aecb58b219f520bb5c23340',
 'keypoints':'094a15aa6b4cdc89388fbee3c4c80b69ed0e5631259bbe57c27ae455a91a6043',
 'native':'371f352669816ba2cf89c7360a7363337cded23eb74cd8a5d5103fa14b7908bb',
 'export_completion':'fea5095c363b2d4a02602c4fe2dec6144058564be877c55caf0ef1dc24631093',
 'factor_macro':'c6c42fcfebf4f896c9e2dcc6eae09cce1c4c660486eae1c82c156e24b07a57a7',
 'factor_completion':'15bad7e64b59dfd92e08ca6d18ebf2475607cc92a678810740e8054a6a90c534',
 'crop_macro':'2c7e98dfc49cf7f61e2d4fbeb2e862d4c5e7529a05653bbcdd4dace098960a1f',
 'crop_completion':'c64d6a43bf691c722a36aa8f10b7af01bd4c467768ed50fe8c50b448369ad662'}
PATHS=('held_R','center_crop_R')
METRICS=('success','auc','center_px','depth_mae_m','rotation_error_deg','translation_error_m')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()

def load_helper(p):
 if sha(p)!=PINS['helper']:raise ValueError('frozen helper SHA mismatch')
 s=importlib.util.spec_from_file_location('frozen_factor_path_geometry',p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m

def key(r):return (r['method'],int(r['seed']),r['condition'])
def rowkey(r):return key(r)+(r['rid'],r['sample_id'])
def variant(path,u,z):return f'{path}/u{u}z{z}'
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def new_crop_center_score(H,record,raw,assets):
 """Only new cell: GT center, predicted z/size, saved GT-center crop R."""
 p=dict(record['predictions']['C1']);p['uv']=H._gt(record)['uv'];p['R']=H._rotation(raw['R_crop_gt_center'])
 return H.score_factors(record,p,assets)

def assemble(H,flat,assets,native,pg,pcr,strict=True):
 records=H.group_flat_predictions(flat);raw={rowkey(r):r for r in flat if r['arm']=='C1'}
 cells=set(map(key,records));expected=set(itertools.product(('eventgenesis','v2e'),range(3),('normal','half','eighth')))
 if strict and (cells!=expected or len(flat)!=3456 or len(records)!=1152):raise ValueError('full 3456/1152 inventory required')
 for cell in cells:
  group=[r for r in records if key(r)==cell]
  if strict and (len(group)!=64 or sorted(__import__('collections').Counter(r['rid'] for r in group).values())!=[16]*4):raise ValueError('64 windows/four complete rollouts required')
 new=[]
 for r in records:
  new.append(dict({n:r[n] for n in ('method','seed','condition','rid','sample_id')},variant=variant('center_crop_R',1,0),metrics=new_crop_center_score(H,r,raw[rowkey(r)],assets)))
 result=H.aggregate_rollout_equal(new)
 def get(cache,name):
  found=[r for r in cache if r['variant']==name]
  if len(found)!=len(cells) or set(map(key,found))!=cells:raise ValueError('cached factor inventory mismatch: '+name)
  return found
 for path,u,z in itertools.product(PATHS,(0,1),(0,1)):
  if (path,u,z)==('center_crop_R',1,0):continue
  source=get(pcr,'rotation_crop/GTtranslation/GTcenter_crop_R') if (path,u,z)==('center_crop_R',1,1) else get(pg,f'oracle/C1/u{u}z{z}r0')
  result.extend(dict(copy.deepcopy(r),variant=variant(path,u,z)) for r in source)
 parity={}
 for path in PATHS:
  base=[dict(r,variant='actual/C1') for r in result if r['variant']==variant(path,0,0)]
  parity[path]=H.native_parity_report(base,[r for r in native if r['arm']=='C1'])
  if parity[path]['status']!='passed':raise ValueError('cached native control parity failed')
 for r in result:
  r.update(diagnostic_only=True,deployable=False,uses_privileged_gt=('u1' in r['variant'] or 'z1' in r['variant']))
  if strict and (r['sample_count']!=64 or r['rollout_count']!=4 or sorted(x['sample_count'] for x in r['by_rollout'].values())!=[16]*4):raise ValueError('cached cohort mismatch')
 return sorted(result,key=lambda r:key(r)+(r['variant'],)),new,parity

def tables(macro):
 conditions=[];rollouts=[]
 for method,light in sorted({(r['method'],r['condition']) for r in macro}):
  out=dict(method=method,condition=light,diagnostic_only=True,deployable=False)
  rows=[r for r in macro if (r['method'],r['condition'])==(method,light)]
  for v in sorted({r['variant'] for r in rows}):
   ss=[r for r in rows if r['variant']==v]
   if sorted(r['seed'] for r in ss)!=[0,1,2]:raise ValueError('all three seeds required')
   for m in METRICS:
    values=np.array([r['summary'][m] for r in ss],float)
    out[v+'/'+m+'_mean']=float(values.mean());out[v+'/'+m+'_sd']=float(values.std(ddof=1))
  conditions.append(out)
 for cell in sorted(set(map(key,macro))):
  ss=[r for r in macro if key(r)==cell];rids=set(ss[0]['by_rollout'])
  if any(set(r['by_rollout'])!=rids for r in ss):raise ValueError('cached rollout IDs mismatch')
  for rid in sorted(rids):
   out=dict(zip(('method','seed','condition'),cell),rid=rid,diagnostic_only=True,deployable=False)
   for r in ss:
    for m in METRICS:out[r['variant']+'/'+m]=r['by_rollout'][rid][m]
   rollouts.append(out)
 return conditions,rollouts

def csvwrite(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for n in ('export-root','geometry-root','geometry-sources','crop-root','out'):a.add_argument('--'+n,type=Path,required=True)
 x=a.parse_args()
 if x.out.exists():raise FileExistsError('fresh output directory required')
 x.out.mkdir(parents=True)
 try:
  paths={'helper':x.geometry_sources/'factor_geometry.py','predictions':x.export_root/'predictions.jsonl','vertices':x.export_root/'assets/vertices.npy','keypoints':x.export_root/'assets/keypoints.npy','native':x.export_root/'native_scores.json','export_completion':x.export_root/'completion.json','factor_macro':x.geometry_root/'macro_scores.json','factor_completion':x.geometry_root/'completion.json','crop_macro':x.crop_root/'macro_scores.json','crop_completion':x.crop_root/'completion.json'}
  receipts={n:dict(path=str(p.resolve()),sha256=sha(p)) for n,p in paths.items()}
  if any(receipts[n]['sha256']!=PINS[n] for n in PINS):raise ValueError('frozen input SHA mismatch')
  H=load_helper(paths['helper']);pgc=json.loads(paths['factor_completion'].read_text());pcrc=json.loads(paths['crop_completion'].read_text())
  for c in (pgc,pcrc):
   if c['status']!='complete_diagnostic_only' or c['real_data_used'] or c['optimizer_updates']!=0:raise ValueError('source scope mismatch')
   for n in ('predictions','vertices','keypoints','native_scores'):
    if c['inputs'][n]['sha256']!=PINS[{'native_scores':'native'}.get(n,n)]:raise ValueError('source input binding mismatch')
  if pcrc['inputs']['factor_macro_json']['sha256']!=PINS['factor_macro'] or pcrc['helper_sha256']!=PINS['helper']:raise ValueError('crop source chain mismatch')
  flat=[json.loads(s) for s in paths['predictions'].read_text().splitlines() if s.strip()]
  assets=H.FactorAssets(np.load(paths['vertices'],allow_pickle=False),np.load(paths['keypoints'],allow_pickle=False),pgc['diameter_m'])
  macro,new,parity=assemble(H,flat,assets,json.loads(paths['native'].read_text()),json.loads(paths['factor_macro'].read_text()),json.loads(paths['crop_macro'].read_text()))
  conditions,rids=tables(macro)
  if len(macro)!=144 or len(new)!=1152 or len(conditions)!=6 or len(rids)!=72:raise ValueError('output inventory mismatch')
  write(x.out/'macro_scores.json',macro);write(x.out/'six_condition_summary.json',conditions);csvwrite(x.out/'six_condition_summary.csv',conditions);csvwrite(x.out/'seventy_two_rollouts.csv',rids);write(x.out/'native_parity.json',parity)
  with (x.out/'new_GTcenter_predZ_cropR.jsonl').open('w') as f:
   for row in new:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
  if any(sha(p)!=receipts[n]['sha256'] for n,p in paths.items()):raise ValueError('source changed during analysis')
  artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in x.out.iterdir() if p.is_file()}
  completion=dict(schema='pose_center_rotation_path_diagnostic_v1',status='complete_diagnostic_only',inputs=receipts,driver_sha256=sha(Path(__file__)),artifacts=artifacts,physical_predictions_C1=1152,macro_cells=144,condition_rows=6,rollout_rows=72,cached_macro_cells=126,new_geometric_window_scores=1152,native_control_cells=36,native_parity='passed',model_forward_calls=0,optimizer_updates=0,real_data_used=False,GT_oracles_deployable=False,interpretation='GT substitutions are diagnostic only. Held-R versus saved GT-center-crop R isolates a conditional crop-mediated path at fixed predicted crop size; it is not a uniquely identified causal effect. All figures use matched64, never nativeclean384.')
  write(x.out/'completion.json',completion);print(json.dumps(completion))
 except BaseException:
  write(x.out/'failure.json',dict(status='failed',error=traceback.format_exc(),diagnostic_only=True));raise

if __name__=='__main__':main()
