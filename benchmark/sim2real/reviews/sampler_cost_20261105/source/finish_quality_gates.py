"""Pairwise physical gate and actual three-task loader gate; CPU only."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;W=E.parent/'simulator_formal_acceptance_20261105d';R=E.parent/'sim2real/20261103-stage9-direct-v1';P=E.parent.parent
summary={'status':'running','pairs':[],'all_exact':False}
for folder,case in [('gate',x) for x in ['s0_00','s1_00','s2_00','s3_00','train_000']]+[('gate_long','s3_00')]:
    out=E/'gate_comparison'/(folder+'_'+case+'_physical.json')
    result=subprocess.run([sys.executable,str(E/'compare_plugin_gates_v2.py'),'--left',str(E/folder/'frozen'/case),'--right',str(E/folder/'candidate'/case),'--out',str(out),'--require-exact'],check=True)
    data=json.loads(out.read_text());summary['pairs'].append({'folder':folder,'case':case,'report':str(out),'passed':data['all_physical_observables_bitwise_equal'],'events':data['raw_events']['shape_left'][0]})
summary['all_exact']=all(x['passed'] for x in summary['pairs'])
(E/'full_quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert summary['all_exact']
folder=E/'formal_events';folder.mkdir(exist_ok=False)
for side in ['frozen','candidate']:
    raw=np.load(E/'gate_long'/side/'s3_00/events.npy')
    assert raw.dtype==np.float64
    raw[:,0]+=1.0 # Restore the exact physical time origin; never round or quantize.
    np.save(folder/(side+'.npy'),raw,allow_pickle=False)
manifest=json.loads((W/'cases_deterministic.json').read_text())
case=manifest['cases'][0];case.update(id='sampler_indexreuse_exact',reference=str(folder/'frozen.npy'),candidate=str(folder/'candidate.npy'))
(E/'formal_cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
script=W/'check_formal_loaders_v2.py'
argv=[sys.executable,str(script),'--cases-json',str(E/'formal_cases.json'),'--scripts-root',str(R/'sources/benchmark/sim2real/scripts'),'--flow-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_steflow_stage9.py'),'--e2vid-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py'),'--e2vid-model-root',str(P/'external/sources/downstream/event_cnn_minimal'),'--output-root',str(E/'formal_loader_gate')]
print('FORMAL_LOADER_SOURCE_SHA256',hashlib.sha256(script.read_bytes()).hexdigest(),flush=True)
subprocess.run(argv,check=True)
summary.update(status='complete',actual_formal_loader_report=str(E/'formal_loader_gate/formal_loader_results.json'),actual_formal_loaders_passed=json.loads((E/'formal_loader_gate/formal_loader_results.json').read_text())['passed'])
(E/'full_quality_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert summary['actual_formal_loaders_passed']
print('ALL_QUALITY_GATES_COMPLETE',flush=True)
