from pathlib import Path
import json,hashlib
L=Path(__file__).parent
R=Path(r'\\wsl.localhost\Ubuntu-22.04\home\科研\Eventbased_WAM\repositories\gec-sim2real')
E=R/'benchmark/sim2real/evidence/S4/20261019-final-best-motion-diagnostic'
d=json.loads((E/'motion_diagnostic.json').read_text())
rows=d['dataset_and_label_checks']['checks']['samples']
out={'sample_count':len(rows),'regions':{}}
for arm in ['eventgenesis','v2e']:
 out['regions'][arm]={}
 for region in ['full_valid','shared_event_active_union']:
  rs=[x['regions'][arm][region] for x in rows];n=sum(x['pixels'] for x in rs)
  epe=sum(x['pixels']*x.get('epe_px_per_30ms',0) for x in rs)/n
  zero=sum(x['pixels']*x.get('zero_flow_epe_px_per_30ms',0) for x in rs)/n
  out['regions'][arm][region]={'pixels':n,'recomputed_epe':epe,'zero':zero,'improvement_fraction':1-epe/zero,'recorded_error':abs(epe-d['metrics']['all_strata_pixel_pooled'][arm][region]['epe_px_per_30ms'])}
out['source_checkpoint_records']=d['paths']['checkpoints']
(L/'final_best_aggregation_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
