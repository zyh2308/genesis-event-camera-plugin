"""Consume finite, pre-registered shards; exit rather than hold an idle GPU."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, signal, socket, subprocess, time

child = None


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def stop(signum, frame):
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    raise SystemExit(128 + signum)


def main():
    global child
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--worker-id', required=True)
    a = p.parse_args()
    spec = json.loads(a.protocol.read_text())
    root = Path(spec['root'])
    queue = root / 'workqueue'
    claims = queue / 'claims'
    claims.mkdir(parents=True, exist_ok=True)
    plan = json.loads((queue / 'PLAN.json').read_text())
    protocol_sha = hashlib.sha256(a.protocol.read_bytes()).hexdigest()
    assert plan['protocol_sha256'] == protocol_sha
    done = []
    for shard in plan['ordered_shards']:
        path = claims / f'shard{shard:03d}.json'
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        claim = dict(shard=shard, worker_id=a.worker_id, host=socket.gethostname(),
                     pid=os.getpid(), started_utc=stamp(), status='running',
                     protocol_sha256=protocol_sha)
        with os.fdopen(fd, 'w') as f:
            json.dump(claim, f, indent=2)
        start = time.monotonic()
        command = ['/usr/bin/python3', '-u', str(Path(spec['source']) / 'run_cadence_comparison.py'),
                   '--protocol', str(a.protocol), '--shard', str(shard)]
        with (queue / f'shard{shard:03d}.log').open('x') as log:
            child = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait()
        child = None
        claim.update(exit_code=code, finished_utc=stamp(), wall_s=time.monotonic()-start,
                     status='completed' if code == 0 else 'infrastructure_failure')
        path.write_text(json.dumps(claim, indent=2))
        done.append(claim)
        (queue / (a.worker_id + '.progress.json')).write_text(json.dumps(done, indent=2))
        print(json.dumps(claim), flush=True)
        if code:
            raise SystemExit(code)
    (queue / (a.worker_id + '.finished.json')).write_text(json.dumps(
        dict(status='PASS', shards=len(done), finished_utc=stamp(),
             note='No unclaimed registered shards remain; worker exits and releases its reservation.'), indent=2))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    main()
