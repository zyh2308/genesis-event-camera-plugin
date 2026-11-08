"""Audit result denominators, terminal scoring, shared baselines and local media links."""
from pathlib import Path
import csv,hashlib,json,re
r=Path(__file__).resolve().parent;out=r/'delivery_local';receipts=[]
for name,count,seeds in [('CADENCE_FINAL.json',44,4),('PAIRED_FINAL_CORRECTED.json',72,6)]:
 d=json.loads((r/name).read_text(encoding='utf-8'));assert d['complete'] and not d['violations'];records=d['records'];assert len(records)==count and len({x['seed'] for x in records})==seeds;assert len({(x['case_id'],x['mode']) for x in records})==count
 for row in records:
  raw=row['result'];expected=bool(raw['success'] and not raw.get('terminated',False) and raw.get('hand_contact_steps',0)>0 and row.get('final_hand_contact',False));assert row['verified_catch']==expected
  if 'initial' in raw:assert row['matched_initial_state']
  else:assert raw.get('steps')==0 and not raw['success'] and row.get('initialization',{}).get('valid') is False
  assert row['protocol']['checkpoint_sha256']=='81585772adf9b368e299875a3dce29247e23c641d8294295bec07dfb6450aaf6'
  if row['mode']!='state_reference':assert row['dense_reads']==0
 receipts.append({'dataset':name,'unique_valid_episodes':count,'seeds':seeds,'raw_terminal_flag_conflicts':sum(x['terminal_flag_conflict'] for x in records),'verified_catches':sum(x['verified_catch'] for x in records)})
new=json.loads((out/'CADENCE_SUMMARY.json').read_text(encoding='utf-8'));assert new['actual_episodes']==44
baseline=list(csv.DictReader((out/'对照表_含明确复用的基线.csv').open(encoding='utf-8-sig')));assert len(baseline)==64;assert len({(x['actual_source_case'],x['method']) for x in baseline})==44
for prefix in ['cadence50','cadence25','difference25']:
 for method in ['ours','v2e']:
  path=r/'received_media'/(prefix+'_'+method+'_event_causal_audit.csv');rows=list(csv.DictReader(path.open(encoding='utf-8')))
  for row in rows:
   if row['maximum_displayed_event_s']:
    assert float(row['maximum_displayed_event_s'])<=float(row['latest_arrived_packet_end'])+1e-6
    assert float(row['latest_arrived_packet_end'])+.01<=float(row['control_t_before'])+1e-8
html=(out/'打开演示.html').read_text(encoding='utf-8');links=re.findall(r'(?:src|href)="([^"]+)"',html);missing=[x for x in links if not (out/x).exists()];assert not missing,missing
for folder in [out,out/'压力测试72回合']:
 audit=json.loads((folder/'FINAL_AUDIT.json').read_text(encoding='utf-8'));assert audit['status']=='PASS'
 result=folder/'SERVER_SHA256SUMS';assert result.exists() and result.stat().st_size>1000
(out/'DELIVERY_QA.json').write_text(json.dumps({'status':'PASS','datasets':receipts,'shared_baseline_cells':64,'unique_actual_new_episodes':44,'all_html_media_links_present':len(links),'demo_events_causal':True,'source_policy_and_core_audited_on_server':True},indent=2),encoding='utf-8');print('PASS:116 unique valid episodes, honest terminal scoring, baseline reuse, causal demos and all local links')
