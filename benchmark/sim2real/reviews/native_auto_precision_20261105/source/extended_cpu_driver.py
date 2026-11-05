import datetime, json, os, subprocess, sys, time
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent/'sim2real/20261103-stage9-direct-v1';P=W.parents[1]
start=time.monotonic()
while True:
    try:
        if json.loads((W/'native_auto/summary.json').read_text())['status']=='complete':break
    except (FileNotFoundError,json.JSONDecodeError):pass
    if time.monotonic()-start>1500:raise TimeoutError('Native suite did not finish')
    time.sleep(5)
subprocess.run([sys.executable,str(W/'prepare_extended_cases.py')],check=True)
args=[sys.executable,'-u',str(W/'check_formal_loaders_v2.py'),'--cases-json',str(W/'cases_extended.json'),
      '--scripts-root',str(R/'sources/benchmark/sim2real/scripts'),
      '--flow-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_steflow_stage9.py'),
      '--e2vid-trainer-source',str(R/'sources/benchmark/sim2real/scripts/train_e2vid_stage9.py'),
      '--e2vid-model-root',str(P/'external/sources/downstream/event_cnn_minimal'),
      '--output-root',str(W/'formal_extended')]
result=subprocess.run(args,timeout=600)
(W/'formal_extended.done.json').write_text(json.dumps({'exit_code':result.returncode,'argv':args,
    'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2)+'\n')
raise SystemExit(result.returncode)
