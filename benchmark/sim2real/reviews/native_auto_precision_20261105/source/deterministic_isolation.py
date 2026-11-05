"""Causal numerical isolation: identical deterministic ops, no sensor changes."""
import gc, hashlib, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import torch
W=Path(__file__).resolve().parent;V=W.parent/'simulator_efficiency_comparison_20261105c'
spec=importlib.util.spec_from_file_location('static_isolation',V/'compare_static.py')
c=importlib.util.module_from_spec(spec);sys.modules[spec.name]=c;spec.loader.exec_module(c)
c.ROOT=W/'deterministic';c.ROOT.mkdir(exist_ok=False)
torch.use_deterministic_algorithms(True)
rows=[]
for case,intervals in [('s0_00',2),('s1_00',2),('s2_00',2),('train_000',2),('s3_00',100)]:
    f=c.Fixture(case,intervals);results=[];times=[]
    for variant in ['original_reference','ours_resident']:
        record,ev=c.run_ours(f,variant)
        assert torch.are_deterministic_algorithms_enabled()
        np.save(c.ROOT/(case+'_'+variant+'.npy'),ev,allow_pickle=False)
        results.append(ev);times.append(record)
        print('DETERMINISTIC_PART',case,variant,len(ev),flush=True)
    a,b=results
    shape_equal=a.shape==b.shape
    row={'case':case,'intervals':intervals,'simulated_s':float(f.ts[-1]),'events_reference':len(a),'events_candidate':len(b),
         'raw_bitwise_equal':bool(shape_equal and a.dtype==b.dtype and a.tobytes()==b.tobytes()),
         'xy_p_order_equal':bool(shape_equal and np.array_equal(a[:,1:],b[:,1:])),
         'max_abs_timestamp_s':float(np.abs(a[:,0]-b[:,0]).max(initial=0)) if shape_equal else None,
         'actual_fast_path_asserted_by_runner':True,'runs':times,'input':f.description}
    rows.append(row)
    (c.ROOT/'results.json').write_text(json.dumps({'status':'running','rows':rows},indent=2)+'\n')
    print('DETERMINISTIC_CASE',json.dumps(row),flush=True)
    gc.collect();torch.cuda.empty_cache()
report={'status':'complete','rows':rows,'all_raw_bitwise_equal':all(x['raw_bitwise_equal'] for x in rows),
        'scope':'isolated deterministic-algorithm conformance, not a claim that default nondeterministic production arrays are bitwise equal',
        'sensor_noise_precision_visibility_unchanged':True,'training_modified':False,'real_data_used':False}
(c.ROOT/'results.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if report['all_raw_bitwise_equal'] else 2)
