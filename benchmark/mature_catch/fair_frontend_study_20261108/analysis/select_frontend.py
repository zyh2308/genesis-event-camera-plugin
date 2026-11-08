"""Lock one frontend per source using complete, actual development episodes only."""
from pathlib import Path
from fractions import Fraction
import argparse, datetime, hashlib, json, math
from collect_fair_stage import PRIMARY, FRONTEND_KEYS, frontend_id

CANONICAL = {'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':'10',
             'CATCHIT_EVENT_FLIGHT_FIT':'1',
             'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':'50'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(data):
 assert data['status']=='PASS' and data['complete']
 assert data['metadata']['stage']=='validation','Blind outcomes cannot select frontends'
 rows=[r for r in data['rows'] if r['method'] in PRIMARY]
 candidates={m:{r['frontend_id'] for r in rows if r['method']==m} for m in PRIMARY}
 assert candidates['ours']==candidates['v2e'] and 1<=len(candidates['ours'])<=5,'Both sources must receive the same registered candidate budget'
 assert frontend_id(CANONICAL) in candidates['ours'],'Canonical frontend was not validated'
 expected={(r['physical_case_id'],r['seed'],float(r['endpoint_hz'])) for r in rows}
 assert {x[2] for x in expected}=={25.,12.5},'Development selection is pre-registered across 25 and 12.5 Hz'
 assert len({x[:2] for x in expected})==4,'Selection requires all four registered development seed blocks'
 assert all(sum(x[:2]==b for x in expected)==2 for b in {x[:2] for x in expected})
 ranking={};locked={}
 for method in PRIMARY:
  scores=[]
  for fid in sorted(candidates[method]):
   values=[r for r in rows if r['method']==method and r['frontend_id']==fid]
   actual=[(r['physical_case_id'],r['seed'],float(r['endpoint_hz'])) for r in values]
   assert len(actual)==len(set(actual)) and set(actual)==expected,'Missing or repeated candidate episodes: '+method+'/'+fid
   env=values[0]['frontend_overrides']
   assert set(env)==set(FRONTEND_KEYS) and all(r['frontend_overrides']==env for r in values)
   assert frontend_id(env)==fid
   byrate=[]
   for hz in sorted({x[2] for x in expected},reverse=True):
    group=[r for r in values if float(r['endpoint_hz'])==hz];n=len(group);k=sum(r['verified_catch'] for r in group)
    perception=[r['perception'] for r in group]
    coverage=sum(r['control_state_coverage'] for r in perception)/n
    error=sum(r['normalized_equal_weight_loss'] for r in perception)/n
    assert all(math.isfinite(x) for x in (coverage,error))
    byrate.append(dict(endpoint_hz=hz,n_physical_blocks=n,verified_catches=k,success_rate=k/n,
     state_availability=coverage,normalized_error=error,
     penalized_position_error_m=sum(r['penalized_position_error_mean_m'] for r in perception)/n,
     penalized_velocity_error_m_s=sum(r['penalized_velocity_error_mean_m_s'] for r in perception)/n))
   success=sum((Fraction(g['verified_catches'],g['n_physical_blocks']) for g in byrate),Fraction())/len(byrate)
   coverage=sum(g['state_availability'] for g in byrate)/len(byrate)
   error=sum(g['normalized_error'] for g in byrate)/len(byrate)
   deviations=sum(str(env[k])!=v for k,v in CANONICAL.items())
   numeric=tuple(int(env[k]) for k in ('CATCHIT_EVENT_OBSERVATION_WINDOW_MS','CATCHIT_EVENT_FLIGHT_FIT','CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS'))
   scores.append(dict(frontend_id=fid,frontend_overrides=env,macro_strict_success=float(success),
    exact_macro_strict_success=str(success),macro_state_availability=coverage,macro_normalized_error=error,
    deviations_from_canonical=deviations,numeric_lex_order=list(numeric),by_rate=byrate,
    actual_episodes=len(values),physical_seed_blocks=4,
    _sort=(-success,-coverage,error,deviations,*numeric)))
  scores.sort(key=lambda r:r['_sort']);winner=scores[0]
  catch_ties=[r['frontend_id'] for r in scores if r['exact_macro_strict_success']==winner['exact_macro_strict_success']]
  for rank,row in enumerate(scores,1):row.pop('_sort');row['rank']=rank
  ranking[method]=dict(selected_frontend=winner['frontend_id'],all_top_success_ties=catch_ties,candidates=scores)
  locked[method]=winner['frontend_overrides']
 return dict(status='LOCKED',frontend_by_method=locked,rankings=ranking,
  actual_candidate_budget_per_source=len(candidates['ours']),actual_primary_episodes=len(rows),
  physical_seed_blocks=4,endpoint_rates_hz=[25,12.5],single_global_configuration_per_source=True,
  rule=['Highest exact macro strict catch rate, equal weight for the two endpoint rates',
        'Highest macro actual control-state availability',
        'Lowest macro normalized equal-weight position/velocity error, including missing-state penalties',
        'Fewest parameter deviations from canonical w10_fit1_gap50',
        'Numeric lexicographic order: smaller window, flight fit off, smaller confirmation gap'],
  error_definition=dict(position_cap_m=.5,velocity_cap_m_s=2.,missing_or_zero_query_loss=1.,
   combined='.5*(min(position_error,.5)/.5)+.5*(min(velocity_error,2)/2)',
   scope='Equal episode means within rate, then equal rate means; all actual post-release/pre-first-real-contact query states, including static/false-grasp propagation. No missing-state exclusion.'),
  comparison='Unrounded stored numeric metrics, no hidden tie epsilon; strict-success macro uses exact fractions',
  no_blind_outcome_used=True,repeated_frontends_and_rates_are_not_independent_seed_blocks=True)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
 root=args.root;path=root/'FAIR_STAGE_RESULTS.json'
 completion=json.loads((root/'FAIR_COLLECTION_COMPLETE.json').read_text())
 assert completion['status']=='PASS' and completion['results_sha256']==digest(path)
 result=select(json.loads(path.read_text()))
 result.update(locked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),validation_root=str(root),validation_results_sha256=digest(path),
  validation_completion_sha256=digest(root/'FAIR_COLLECTION_COMPLETE.json'),
  registered_protocol_sha256=digest(root/'EVALUATION_PROTOCOL.json'),selection_tool_sha256=digest(__file__))
 output=root/'FRONTEND_SELECTION.json'
 assert not output.exists(),'Refuse to overwrite an already sealed frontend selection'
 output.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(path=str(output),sha256=digest(output),frontend_by_method=result['frontend_by_method'])))


if __name__=='__main__':main()
