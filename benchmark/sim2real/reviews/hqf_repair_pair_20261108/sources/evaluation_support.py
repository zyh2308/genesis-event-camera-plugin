import datetime,json,os,signal,subprocess
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);t=path.with_name(path.name+'.partial');t.write_text(json.dumps(value,indent=2)+'\n');t.replace(path)
def stop_child(child):
    if child.poll() is not None:return
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
