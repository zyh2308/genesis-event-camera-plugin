from pathlib import Path
import json,statistics,hashlib
B=Path('//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real')
D=B/'evidence/S5/20261019-parallel-diagnostics'
out={}
d=json.loads((D/'flow_input_perturbation_results.json').read_text())
out['perturbation']={'models_keys':{a:list(v) for a,v in d['models'].items()},'all_windows_start_s':[x['event_start_s'] for x in d['frozen_masks_and_inputs']],
 'active':sum(x['frozen_active_union_pixels'] for x in d['frozen_masks_and_inputs']),
 'valid':sum(x['valid_pixels'] for x in d['frozen_masks_and_inputs']),
 'counts':{a:sum(x['event_counts'][a] for x in d['frozen_masks_and_inputs']) for a in ['eventgenesis','v2e']}}
p=D/'gt_supervised_pilot/gt_supervised_8window_80update_v1_7dfaa80'
d=json.loads((p/'posthoc_active_eval/summary.json').read_text())
out['active_pilot']={}
for a,arm in d['arms'].items():
 out['active_pilot'][a]={}
 for split in ['train','validation']:
  rows=arm[split]
  out['active_pilot'][a][split]={
   'epe':statistics.mean(r['common_active_epe_px_per_30ms'] for r in rows),
   'zero':statistics.mean(r['common_active_zero_flow_epe_px_per_30ms'] for r in rows),
   'active':sum(r['common_event_active_pixels'] for r in rows),'valid':sum(r['valid_pixels'] for r in rows),
   'matches_summary':abs(statistics.mean(r['common_active_epe_px_per_30ms'] for r in rows)-arm[split+'_active_macro_epe'])<1e-10}
d=json.loads((p/'scale_target_audit/summary.json').read_text())
out['scale_audit']={a:{'n':len(v['samples']),'max_train_eval_abs':max(x['scale0_train_to_eval_max_abs_difference_px'] for x in v['samples'])} for a,v in d['arms'].items()}
remote=json.loads((Path(__file__).parent/'stage5_review_remote.json').read_text())
out['tc_source_hash_match']={rel:hashlib.sha256((B/rel).read_bytes()).hexdigest()==digest for rel,digest in remote['code_hashes'].items()}
out['resume']={a:{'oracle_keys':list(v.get('resume_oracle.json',{})),'loader_keys':list(v.get('loader_preflight.json',{}))} for a,v in remote['arms'].items()}
(Path(__file__).parent/'stage5_reaggregate.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
