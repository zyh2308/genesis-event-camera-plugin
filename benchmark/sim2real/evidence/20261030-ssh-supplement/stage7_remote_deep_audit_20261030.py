"""Read existing real-test artifacts; no model inference, training or new scoring."""
from pathlib import Path
import json,hashlib,datetime,math,statistics,subprocess,collections
BASE=Path('/home/zhaoyuhan/Eventbased_WAM')
RUN=BASE/'runs/sim2real/20261025-hqf14-e2vid-seed0-actual-r2'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
sp=RUN/'paired_score.json';score=json.loads(sp.read_text());mp=Path(score['run_manifest']['path']);manifest=json.loads(mp.read_text())
out={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':subprocess.check_output(['hostname'],text=True).strip(),'operation':'existing artifacts audit only; no new inference or LPIPS scoring','score_path':str(sp),'score_bytes':sp.stat().st_size,'score_sha256_actual':sha(sp),'run_manifest_path':str(mp),'run_manifest_sha256_actual':sha(mp),'run_manifest_sha256_expected':score['run_manifest']['sha256'],'metric_provenance':score['metric_provenance'],'protocol':score['protocol'],'data_manifest':score['data_manifest'],'manifest_keys':list(manifest),'manifest_header':{k:v for k,v in manifest.items() if k!='arms'},'per_sequence':[],'aggregate':{},'preview_records':[],'checkpoints':{}}
assert out['score_sha256_actual']=='4fb0679ddcde1f4af3b2cbccf1af99fc0e193c9c51e809b171947b2dd36a3692'
out['run_manifest_matches_embedded_score_hash']=out['run_manifest_sha256_actual']==out['run_manifest_sha256_expected']
for key in ('data_manifest','protocol'):
 p=Path(score[key]['path'])
 if p.is_file():out[key]['actual_sha256']=sha(p);out[key]['actual_matches']=out[key]['actual_sha256']==score[key]['sha256']
seqs=sorted(score['per_arm']['eventgenesis']['per_sequence']);groups=collections.defaultdict(list)
for row in score['per_frame']:groups[(row['arm'],row['sequence'])].append(row)
metrics=('lpips_alex','ssim','psnr_db','mse')
for arm in ('eventgenesis','v2e'):
 per=[]
 binding=score['checkpoint_bindings'][arm];p=Path(binding['path']);actual=sha(p)
 out['checkpoints'][arm]={**binding,'actual_sha256':actual,'matches':actual==binding['sha256'],'bytes':p.stat().st_size}
 for seq in seqs:
  rows=groups[(arm,seq)];saved=score['per_arm'][arm]['per_sequence'][seq]
  assert [r['sample_index'] for r in rows]==list(range(1,len(rows)+1))
  values={m:statistics.mean(r['metrics'][m] for r in rows) for m in metrics}
  old=saved['metrics'];expected={'lpips_alex':old['lpips_alex_mean'],'ssim':old['ssim_mean'],'mse':old['mse_mean'],'psnr_db':old['psnr']['mean_db']}
  errs={m:abs(values[m]-expected[m]) for m in metrics};assert max(errs.values())<1e-12
  entry={'arm':arm,'sequence':seq,'frames':len(rows),**values,'max_difference_from_stored_summary':max(errs.values())};out['per_sequence'].append(entry);per.append(entry)
  previews=[r for r in rows if r['is_preregistered_preview']]
  for row in previews:out['preview_records'].append(row)
  # Find the source input path via the registered per-sequence manifest.
  arm_seq=manifest['arms'][arm]['sequences'][seq]
  entry['inference_run_record']=arm_seq
 agg={m:statistics.mean(x[m] for x in per) for m in metrics}
 old=score['per_arm'][arm]['equal_sequence_mean'];expect={'lpips_alex':old['lpips_alex_mean'],'ssim':old['ssim_mean'],'mse':old['mse_mean'],'psnr_db':old['psnr']['mean_db']}
 assert max(abs(agg[m]-expect[m]) for m in metrics)<1e-12
 out['aggregate'][arm]={'sequence_count':len(per),'frames':sum(x['frames'] for x in per),**agg,'matches_stored_aggregate':True}
out['paired_differences']=[]
for seq in seqs:
 a=next(r for r in out['per_sequence'] if r['arm']=='eventgenesis' and r['sequence']==seq)
 b=next(r for r in out['per_sequence'] if r['arm']=='v2e' and r['sequence']==seq)
 out['paired_differences'].append({'sequence':seq,**{m:a[m]-b[m] for m in metrics}})
out['eventgenesis_win_counts']={m:sum((r[m]<0 if m in ('lpips_alex','mse') else r[m]>0) for r in out['paired_differences']) for m in metrics}
out['gpu_snapshot']=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=20).stdout
flow=BASE/'runs/sim2real/20261029-stage5-hightexture-fixed-gtfit-g3-v2';out['flow_tail_evidence']={}
for p in sorted(flow.iterdir()):
 if p.is_file() and p.name in ('run_state.txt','exit_code.txt','finished_utc.txt'):out['flow_tail_evidence'][p.name]=p.read_text()
 if p.is_dir() and any(s in p.name for s in ('restore','checkpoint')):
  out['flow_tail_evidence'][p.name]={'files':[x.name for x in p.iterdir() if x.is_file()]}
  for q in p.iterdir():
   if q.is_file() and q.suffix in ('.json','.txt','.log') and q.stat().st_size<40000:out['flow_tail_evidence'][p.name][q.name]=q.read_text(errors='replace')[-5000:]
pose=BASE/'runs/sim2real/20261029-epose-second-sample-audit';out['epose_audit_files']=[{'name':p.name,'bytes':p.stat().st_size} for p in pose.iterdir() if p.is_file()]
print(json.dumps(out,ensure_ascii=False))
