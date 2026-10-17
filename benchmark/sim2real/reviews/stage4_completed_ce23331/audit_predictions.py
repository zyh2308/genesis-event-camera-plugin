from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=0','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_prediction_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=240)
(O/'prediction_diagnostic.json').write_text(r.stdout);(O/'prediction_stderr.txt').write_text(r.stderr)
if r.returncode:print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'tail':r.stdout[-1000:]}))
else:
    import numpy as np
    d=json.loads(r.stdout);summary={}
    for gen,a in d['arms'].items():
        summary[gen]={'windows':len(a['windows']),'prediction_magnitude_mean':float(np.mean([r['prediction_magnitude_mean'] for r in a['windows']])),'masked_mean_EPE':{k:{name:float(np.mean([r['masked_metrics'][k][name] for r in a['windows'] if r['masked_metrics'][k][name] is not None])) for name in ['model_EPE','zero_EPE']} for k in a['windows'][0]['masked_metrics']}}
    (O/'prediction_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
