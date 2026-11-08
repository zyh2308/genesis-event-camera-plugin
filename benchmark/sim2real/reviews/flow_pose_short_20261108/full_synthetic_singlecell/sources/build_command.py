#!/usr/bin/env python3
"""Build one sealed stdin command after root has explicitly published CPU-ready."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import lease_single_cell as r


def rec(path):
    p=path.resolve(strict=True)
    return dict(path=str(p),size_bytes=p.stat().st_size,sha256=r.sha(p))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--gpu-index',type=int,choices=[1],required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();root=a.task_root.resolve(strict=True)
    r.need(root.parent==r.RUNS and root.name.startswith('20261108-flow-full-synthetic-readonly-'),'Wrong task root')
    path=r.within(a.out,root)
    r.need(not path.exists(),'Refusing existing sealed command')
    entry=root/'sources/evaluate_single_cell.py';wc=root/'sources/contract.json'
    shard='rife_seed1';out=root/shard
    seeds=['1']
    parent=Path(r.__file__).resolve()
    c=dict(schema='flow_full_synthetic_cell_command_v1',task='flow_full_synthetic_readonly_singlecell',task_root=str(root),
        gpu_index=a.gpu_index,gpu_uuid=r.GPUS[a.gpu_index],optimizer_updates=0,real_data_used=False,
        real_scores_computed=False,model_weights_read_only=True,timeout_seconds=1200,min_free_mib=22000,
        parent_source=rec(parent),source_manifest=rec(root/'sources/source_manifest.json'),worker_contract=rec(wc),
        entrypoint=rec(entry),ready=rec(root/'DEPLOYMENT_READY.json'),cwd=str(root/'sources'),out=str(out),
        argv=[str(Path(sys.executable).resolve()),'-B','-u',str(entry),'--contract',str(wc),'--seeds',*seeds,'--methods','rife_esim','--out',str(out)])
    r.verify_contract(c,root,a.gpu_index,parent)
    payload=(json.dumps(c,sort_keys=True,separators=(',',':'))+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(payload)
    print(json.dumps(dict(path=str(path),size_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                         GPU_queried=False,GPU_leased=False,worker_started=False)))


if __name__=='__main__':main()
