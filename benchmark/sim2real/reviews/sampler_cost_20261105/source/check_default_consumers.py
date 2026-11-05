"""Actual normal-mode inputs and fixed-checkpoint predictions, plus old/self control."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
E=Path(__file__).resolve().parent;W=E.parent/'simulator_formal_acceptance_20261105d';R=E.parent/'sim2real/20261103-stage9-direct-v1';P=E.parent.parent
rawdir=E/'normal_formal_events';rawdir.mkdir(exist_ok=False)
for name,method,trial in [('old0','frozen_gpu',0),('new0','candidate_gpu',0),('old1','frozen_gpu',1)]:
    x=np.load(E/'bench/s3_00'/method/('trial'+str(trial)+'_events.npy'));assert x.dtype==np.float64
    x[:,0]+=1.0;np.save(rawdir/(name+'.npy'),x,allow_pickle=False)
manifest=json.loads((W/'cases_deterministic.json').read_text());template=manifest['cases'][0];cases=[]
for label,other in [('old_vs_new','new0'),('old_vs_old','old1')]:
    case=dict(template,id=label,reference=str(rawdir/'old0.npy'),candidate=str(rawdir/(other+'.npy')),start_s=1.0,end_s=1.2);cases.append(case)
manifest['cases']=cases;(E/'normal_formal_cases.json').write_text(json.dumps(manifest,indent=2)+'\n')
argv=[sys.executable,str(W/'check_formal_loaders_v2.py'),'--cases-json',str(E/'normal_formal_cases.json'),'--scripts-root',str(R/'sources/benchmark/sim2real/scripts'),'--flow-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_steflow_stage9.py'),'--e2vid-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py'),'--e2vid-model-root',str(P/'external/sources/downstream/event_cnn_minimal'),'--output-root',str(E/'normal_formal_gate')]
code=subprocess.run(argv).returncode
if code not in [0,2]:raise RuntimeError('Formal input check did not complete')
report=json.loads((E/'normal_formal_gate/formal_loader_results.json').read_text());assert report['status']=='complete'
previous=json.loads((W/'03_reconstruction_gpu_probe.done.json').read_text())['job']['argv']
argv=list(previous);argv[0]=sys.executable
overrides={'--reference-h5':E/'normal_formal_gate/old_vs_new/reference.h5','--candidate-h5':E/'normal_formal_gate/old_vs_new/candidate.h5','--reference-repeat-h5':E/'normal_formal_gate/old_vs_old/candidate.h5','--start-s':'1','--end-s':'1.2','--expected-windows':'20','--output-root':E/'normal_reconstruction_predictions_gpu'}
for flag,value in overrides.items():argv[argv.index(flag)+1]=str(value)
# Read the registered full digest, avoiding transcription of checkpoint hashes.
registered=json.loads((R/'training/reconstruction/seed0/eventgenesis/completion.json').read_text())['selected_checkpoint_sha256']
argv[argv.index('--expected-checkpoint-sha256')+1]=registered
subprocess.run(argv,check=True)
probe=json.loads((E/'normal_reconstruction_predictions_gpu/reconstruction_prediction_probe.json').read_text());assert probe['status']=='complete'
summary={'status':'complete','normal_loader_check_exit_code':code,'normal_loader_inputs_all_exact':report['tested_model_inputs_bitwise_equal'],'prediction_pairs':{k:{m:v for m,v in d.items() if m!='rows'} for k,d in probe['pairs'].items()},'no_tolerance_acceptance':True,'no_training_or_real_test':True}
(E/'normal_consumer_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
