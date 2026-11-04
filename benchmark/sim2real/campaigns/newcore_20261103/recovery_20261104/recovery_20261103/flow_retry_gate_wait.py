import subprocess,sys,time
from pathlib import Path
subprocess.run(sys.argv[1:],check=True)
gate=Path.cwd()/'preflight/tasks/flow/paired_gate.json'
for _ in range(60):
    if gate.is_file():
        print('Flow paired gate published',flush=True)
        break
    time.sleep(2)
else:
    raise SystemExit('Flow retry passed, but pair gate was not published within 120 seconds')
