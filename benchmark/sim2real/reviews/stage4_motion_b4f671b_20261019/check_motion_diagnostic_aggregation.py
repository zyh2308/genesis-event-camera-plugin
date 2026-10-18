from pathlib import Path
import json,hashlib
R=Path(r'\\wsl.localhost\Ubuntu-22.04\home\科研\Eventbased_WAM\repositories\gec-sim2real')
E=R/'benchmark/sim2real/evidence/S4/20261018-motion-diagnostic'
d=json.loads((E/'motion_diagnostic.json').read_text(encoding='utf-8'))
rows=d['dataset_and_label_checks']['checks']['samples']
out={'samples':len(rows),'unique_ids':len({r['sample_id'] for r in rows}), 'aggregation':{},'hashes':{}}
for arm in ['eventgenesis','v2e']:
    out['aggregation'][arm]={}
    for region in ['full_valid','shared_event_active_union','gt_motion_ge_0_5','gt_motion_ge_1_0']:
        rr=[r['regions'][arm][region] for r in rows]
        n=sum(r['pixels'] for r in rr)
        epe=sum(r['pixels']*r.get('epe_px_per_30ms',0) for r in rr)/n
        zero=sum(r['pixels']*r.get('zero_flow_epe_px_per_30ms',0) for r in rr)/n
        near=sum(r['pixels']*r.get('near_zero_prediction_rate_lt_0_1_px',0) for r in rr)/n
        recorded=d['metrics']['all_strata_pixel_pooled'][arm][region]
        out['aggregation'][arm][region]={'pixels':n,'recomputed_epe':epe,'epe_abs_error':abs(epe-recorded['epe_px_per_30ms']),
          'zero_abs_error':abs(zero-recorded['zero_flow_epe_px_per_30ms']),'near_zero_abs_error':abs(near-recorded['near_zero_prediction_rate']['0.1']),
          'coverage':n/sum(r['valid_pixels'] for r in rows),'improvement_fraction':1-epe/zero}
for p in E.iterdir():
    if p.is_file():out['hashes'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((E/'run_receipt.json').read_text(encoding='utf-8'))
out['receipt_result_hash_matches']=receipt['diagnostic_result']['sha256']==out['hashes']['motion_diagnostic.json']
out['source_hash_matches_receipt']=hashlib.sha256((R/'benchmark/sim2real/scripts/stage4_motion_learning_diagnostic.py').read_bytes()).hexdigest()==receipt['diagnostic_code']['sha256']
Path(__file__).with_name('motion_aggregation_verification.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
