"""Write Chinese result notes only from complete, locked, audited blind artifacts."""
from pathlib import Path
import argparse, csv, json, math, sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parent/'figures_tool'))
import make_fair_figures as f

RATES=(100.,50.,25.,12.5)
LABELS={'ours':'Ours','v2e':'V2E','init_only':'仅首帧RGB-D、停止事件更新','state_reference':'原任务含噪状态参考'}
FAILURES={'none':'','initialization_failure':'首帧初始化失败','no_contact':'未接触',
 'contact_then_termination':'接触后失败终止','contact_not_retained_at_task_end':'任务结束时未保持接触'}


def resolve_selection(stage,protocol):
 choices=[stage/'FRONTEND_SELECTION.json',Path(protocol['frontend_selection_file'])]
 selected=next((p for p in choices if p.is_file()),None)
 f.require(selected is not None,'Missing actual FRONTEND_SELECTION.json; download the locked file, do not reconstruct it')
 f.require(f.sha(selected)==protocol['frontend_selection_sha256'],'Locked frontend file hash differs from blind protocol')
 lock=f.read_json(selected)
 f.require(lock.get('status')=='LOCKED' and lock.get('locked_utc') and lock.get('no_blind_outcome_used') is True
  and lock.get('single_global_configuration_per_source') is True,'Missing complete pre-blind frontend lock')
 f.require(lock.get('actual_primary_episodes')==48 and lock.get('physical_seed_blocks')==4,'Unexpected development selection budget')
 return selected,lock


def verified_statistics(figures,stage,blind):
 path=figures/'blind_paired_by_rate.csv'
 fields=('endpoint_hz','paired_blocks','both_success','ours_only','v2e_only','both_fail',
  'ours_catches','v2e_catches','success_rate_difference','mcnemar_exact_two_sided','mcnemar_holm_across_rates')
 rows=f.csv_rows(path,fields);byrate={f.number(r['endpoint_hz'],'rate'):r for r in rows}
 f.require(len(rows)==4 and set(byrate)==set(RATES),'Figures must report all four registered rate comparisons')
 manifestpath=figures/'blind_success_vs_rate_MANIFEST.json';manifest=f.read_json(manifestpath)
 for name in ('FAIR_STAGE_RESULTS.json','FAIR_COLLECTION_COMPLETE.json'):
  f.require(any(Path(p['path']).name==name and p['sha256']==f.sha(stage/name) for p in manifest['inputs']),
   'Figures are not bound to this actual collector stage: '+name)
 f.require(any(Path(p['path']).name==path.name and p['sha256']==f.sha(path) for p in manifest['outputs']),
  'Paired statistics CSV differs from its figure manifest')
 for expected in blind['paired']:
  actual=byrate[expected['endpoint_hz']]
  for key,value in expected.items():
   f.require(math.isclose(f.number(actual[key],key),float(value),rel_tol=1e-12,abs_tol=1e-12),
    'Figure statistics differ from actual paired outcomes: '+key)
 return byrate,[path,manifestpath]


def build(stage,figures,cost=None):
 stage=Path(stage).resolve();figures=Path(figures).resolve()
 loaded=f.load_stage(stage,'blind');f.require(len(loaded['rows'])==320,'Summary requires all320 planned blind actors')
 protocol=f.read_json(stage/'EVALUATION_PROTOCOL.json');selection,lock=resolve_selection(stage,protocol)
 f.require(sum(len(c['method_order']) for c in protocol['cases'])==320,'Protocol actor budget differs from320')
 planned={(c['physical_case_id'],int(c['seed']),float(c['endpoint_hz']) if method in f.METHODS else None,method)
  for c in protocol['cases'] for method in c['method_order']}
 actual={(r['physical_case_id'],r['seed'],r['endpoint_hz'],r['method']) for r in loaded['rows']}
 f.require(len(planned)==320 and planned==actual,'Actual blind outcomes differ from the planned physical cases')
 blind=f.validate_blind(loaded,selection,32,f.BLIND_RATES)
 reported,statsinputs=verified_statistics(figures,stage,blind)
 per_rate=[]
 for hz in RATES:
  row=reported[hz];ours=next(x for x in blind['summaries'] if x['method']=='ours' and x['endpoint_hz']==hz)
  other=next(x for x in blind['summaries'] if x['method']=='v2e' and x['endpoint_hz']==hz)
  paired=next(x for x in blind['paired'] if x['endpoint_hz']==hz)
  direction='none'
  if f.number(row['mcnemar_holm_across_rates'],'Holm')<.05:
   direction='ours' if paired['ours_only']>paired['v2e_only'] else 'v2e' if paired['v2e_only']>paired['ours_only'] else 'none'
  per_rate.append(dict(endpoint_hz=hz,ours_successes=ours['successes'],v2e_successes=other['successes'],denominator=32,
   paired=paired,reported_mcnemar_p=row['mcnemar_exact_two_sided'],reported_holm_p=row['mcnemar_holm_across_rates'],
   statistically_supported_higher_success=direction))
 cost_result=dict(status='NOT_AVAILABLE',message='尚无有效成本曲线。')
 inputs=loaded['inputs']+[stage/'EVALUATION_PROTOCOL.json',selection]+statsinputs
 if cost is not None:
  cost=Path(cost).resolve();measured=f.validate_cost(cost,blind,minimum_repeats=3)
  cost_result=dict(status='VALIDATED',message='已有通过完整性检查的固定轨迹成本数据；成本与抓取成功率分别统计。',
   summaries=measured['summaries'],hardware=measured['hardware'],equivalence_kind=measured['equivalence_kind'],
   actual_primary_measurements=len(measured['accepted']),excluded_rows=len(measured['excluded']))
  inputs.append(cost)
 original=f.read_json(stage/'FAIR_STAGE_RESULTS.json')['rows'];raw={f.key(f.episode(r)):r for r in original}
 chinese=[]
 for row in sorted(loaded['rows'],key=lambda r:(r['physical_case_id'],r['seed'],r['method'],r['endpoint_hz'] or 0)):
  source=raw[f.key(row)];failure=source.get('failure_class','')
  chinese.append({'记录ID':source.get('case_id',''),'来球条件':row['physical_case_id'],'随机种子':row['seed'],
   '输入图像频率_Hz':f'{row["endpoint_hz"]:g}' if row['endpoint_hz'] is not None else '共享参考（每条件仅一次）',
   '方法':LABELS[row['method']],'感知前端':row['frontend_id'],'接取成功':'是' if row['verified_catch'] else '否',
   '失败原因':FAILURES.get(failure,failure),'首帧初始化成功':'不适用' if row['initialization_valid'] is None else '是' if row['initialization_valid'] else '否',
   '发生手接触的控制步数':row['hand_contact_steps'],'末步发生接触':'是' if row['final_contact'] else '否',
   '模拟时长_秒':row['sim_time_s'],'单回合进程耗时_秒（非吞吐指标）':row['wall_s'],'原始输出目录':row['output']})
 summary=dict(status='PASS',actual_actors=320,independent_throwing_blocks=32,source_rates_hz=list(RATES),
  locked_frontends=blind['selected'],locked_utc=lock['locked_utc'],per_rate=per_rate,references=blind['references'],
  all_successes_and_failures_retained=True,references_reused_without_denominator_inflation=True,
  experiment='固定原策略的事件感知闭环比较；首帧实际RGB-D初始化，随后事件与机器人自身状态。没有训练新策略，不是训练数据迁移实验。',
  fixed_setup='原Catch It!权重、7°机位、冻结事件仿真器；每种来源使用开发阶段锁定的一套前端。',
  catch_definition='完成原任务时限，未失败终止，曾发生真实手接触，且最后40ms控制步内发生接触；不代表连续保持0.5秒。',
  statistics_scope='同seed来球条件配对；四档比较一并报告Holm校正，不挑选帧率。跨帧率重复不增加独立种子数。',cost=cost_result,
  inputs=[dict(path=str(p.resolve()),sha256=f.sha(p)) for p in dict.fromkeys(Path(p) for p in inputs)],
  summary_tool_sha256=f.sha(__file__),figures_tool_sha256=f.sha(f.__file__))
 return summary,chinese


def write_summary(stage,figures,output,cost=None):
 output=Path(output);f.require(not output.exists(),'Use a fresh summary output directory')
 summary,rows=build(stage,figures,cost);output.mkdir(parents=True,exist_ok=False)
 with (output/'逐条结果.csv').open('w',encoding='utf-8-sig',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 (output/'汇总.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 lines=['已完成32组新来球条件、320个正式回合。四档输入图像频率中，Ours和V2E各测试32次；两个共享参考各测试32次。所有成功与失败均保留。',
  '',summary['experiment'],summary['fixed_setup'],'',
  '| 输入图像频率 | Ours成功/32 | V2E成功/32 | 仅Ours成功 | 仅V2E成功 | 同时成功 | 同时失败 | McNemar p | 四档Holm校正p |',
  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for row in summary['per_rate']:
  p=row['paired'];lines.append(f'| {row["endpoint_hz"]:g} Hz | {row["ours_successes"]}/32 | {row["v2e_successes"]}/32 | {p["ours_only"]} | {p["v2e_only"]} | {p["both_success"]} | {p["both_fail"]} | {row["reported_mcnemar_p"]} | {row["reported_holm_p"]} |')
 lines+=['','；'.join(f'{LABELS[r["method"]]}：{r["successes"]}/32' for r in summary['references'])+'。参考跨帧率共用，每个条件仅算一次。','',summary['catch_definition'],'',summary['statistics_scope']]
 significant=[r for r in summary['per_rate'] if r['statistically_supported_higher_success']!='none']
 if significant:lines+=['','；'.join(f'{r["endpoint_hz"]:g} Hz档{LABELS[r["statistically_supported_higher_success"]]}成功率较高，并通过四档Holm校正' for r in significant)+'。结论限于本任务和已测条件。']
 else:lines+=['','本批没有确立任何一方在四档比较中的统计显著优势；数值差异仍完整列出。']
 lines+=['',summary['cost']['message'],'','逐回合明细见[逐条结果.csv](逐条结果.csv)，输入来源及完整统计见[汇总.json](汇总.json)。',
  '图表来自：'+str(Path(figures).resolve())+'。原样报告图表统计，不使用单回合结束耗时推算成本。']
 (output/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 return summary


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',type=Path,required=True)
 parser.add_argument('--figures',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cost',type=Path)
 args=parser.parse_args()
 try:
  result=write_summary(args.stage,args.figures,args.output,args.cost)
  print(json.dumps(dict(status='PASS',actors=result['actual_actors'],blocks=result['independent_throwing_blocks'],output=str(args.output.resolve()))))
 except (f.DataError,KeyError,TypeError,OSError,json.JSONDecodeError) as exc:parser.exit(2,'NOT_READY: '+str(exc)+'\nNo partial or placeholder summary is written.\n')


if __name__=='__main__':main()
