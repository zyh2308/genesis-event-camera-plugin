"""Resume official outdoor HDF5 inputs in an isolated staging directory."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any
import urllib.request
import zipfile

import h5py
import numpy as np

FILES={
    'outdoor_day_calib.zip':('1Y0sPP0ebX_cEKUCVVhJLej9TZgCxS3ME',1290038,'calibration'),
    'outdoor_day1_gt.hdf5':('1wzUmTBxQ5wtSpB0KBogliB2IGTrCtJ7e',6267453264,'gt'),
    'outdoor_day1_data.hdf5':('1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0',12136423279,'data'),
}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def load_validators(path):
    tree=ast.parse(path.read_text())
    definitions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('validate_hdf5','validate_archive')]
    assert len(definitions)==2
    scope=dict(Path=Path,Any=Any,h5py=h5py,np=np,zipfile=zipfile)
    exec(compile(ast.Module(body=definitions,type_ignores=[]),str(path),'exec'),scope)
    return scope


def probe(name,identifier,total,proxy):
    url='https://drive.usercontent.google.com/download?id='+identifier+'&export=download&confirm=t'
    handler=urllib.request.ProxyHandler({'https':proxy}) if proxy else urllib.request.ProxyHandler()
    opener=urllib.request.build_opener(handler)
    request=urllib.request.Request(url,headers={'Range':'bytes=0-4095','User-Agent':'Mozilla/5.0'})
    with opener.open(request,timeout=30) as response:
        prefix=response.read(4096)
        assert response.status==206 and response.headers.get('Content-Range')=='bytes 0-4095/'+str(total)
        assert 'html' not in response.headers.get('Content-Type','').lower()
        expected=b'PK' if name.endswith('.zip') else b'\x89HDF\r\n\x1a\n'
        assert prefix.startswith(expected),'Official endpoint returned a non-dataset response'
        disposition=response.headers.get('Content-Disposition','')
        assert name in disposition,'Official endpoint filename differs'
        return dict(url=url,etag=response.headers.get('ETag'),expected_bytes=total,
                    prefix_sha256=hashlib.sha256(prefix).hexdigest(),disposition=disposition,
                    content_range=response.headers.get('Content-Range'),checked_utc=now())


def fetch(name,args,validators):
    identifier,total,kind=FILES[name]
    folder=args.out/'outdoor_day';folder.mkdir(exist_ok=True)
    final=folder/name;partial=folder/(name+'.part');record=args.out/(name+'.json')
    assert not final.is_symlink() and not partial.is_symlink()
    if final.exists():
        old=json.loads(record.read_text())
        assert old['status']=='ready_unscored' and old['drive_file_id']==identifier
        assert final.stat().st_size==total and sha(final)==old['sha256']
        return old
    initial=probe(name,identifier,total,args.proxy)
    if partial.exists() and partial.stat().st_size:
        old=json.loads(record.read_text())
        assert old['status']=='partial' and old['drive_file_id']==identifier
        assert partial.stat().st_size==old['size_bytes'] and sha(partial)==old['sha256']
        assert old['probe']['prefix_sha256']==initial['prefix_sha256']
        if old['probe'].get('etag') and initial.get('etag'):assert old['probe']['etag']==initial['etag']
        with partial.open('rb') as f:
            magic=b'PK' if name.endswith('.zip') else b'\x89HDF\r\n\x1a\n'
            assert f.read(len(magic))==magic,'Refuse to resume a saved HTML/error response'
    command=['curl','--location','--fail','--proto','=https','--proto-redir','=https','--connect-timeout','30','--max-time','10800',
        '--speed-limit','1024','--speed-time','180','--continue-at','-',
        '--limit-rate',args.limit_rate,'--output',str(partial),
        '--user-agent','Mozilla/5.0','--dump-header',str(args.out/(name+'.headers.log'))]
    if args.proxy:command+=['--proxy',args.proxy]
    etag=initial.get('etag')
    if etag and re.fullmatch(r'(W/)?"[A-Za-z0-9_.:/+=-]+"',etag):command+=['--header','If-Match: '+etag]
    command.append(initial['url'])
    start=now()
    with (args.out/(name+'.curl.log')).open('ab',buffering=0) as log:
        result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    info=dict(name=name,drive_file_id=identifier,kind=kind,source='official MVSEC HDF5 folder',
        probe=initial,started_utc=start,finished_utc=now(),curl_exit_code=result.returncode,
        path=str(partial),size_bytes=partial.stat().st_size if partial.exists() else 0,
        sha256=sha(partial) if partial.exists() else None,status='partial',real_scores_computed=False)
    save(record,info)
    assert result.returncode==0 and info['size_bytes']==total,'Incomplete transfer retained with prefix SHA'
    checked=validators['validate_archive'](partial) if kind=='calibration' else validators['validate_hdf5'](partial,kind)
    assert not final.exists()
    # An atomic non-overwriting link publishes the verified input. Only its own
    # temporary link is removed; the complete dataset bytes remain at final.
    os.link(partial,final)
    assert os.path.samefile(partial,final)
    partial.unlink()
    info.update(status='ready_unscored',path=str(final),validation=checked,
        official_reference_checksum_available=False,checksum_note='Full local SHA recorded; no independent official checksum was provided')
    save(record,info)
    print('OFFICIAL_MVSEC_INPUT_READY',name,info['size_bytes'],info['sha256'],flush=True)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--validator-source',type=Path,required=True)
    p.add_argument('--proxy')
    p.add_argument('--limit-rate',default='16M')
    args=p.parse_args();args.out=args.out.absolute()
    assert shutil.which('curl')
    args.out.mkdir(parents=True,exist_ok=True)
    assert not args.out.is_symlink()
    lock=(args.out/'.download.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    marker=args.out/'owner.json'
    identity=dict(schema='official_mvsec_outdoor_staging_v1',files=FILES)
    if marker.exists():assert json.loads(marker.read_text())==json.loads(json.dumps(identity))
    else:
        assert not list(args.out.glob('outdoor_day/*'))
        save(marker,identity)
    remaining=sum(max(0,total-next((p.stat().st_size for p in
        [args.out/'outdoor_day'/name,args.out/'outdoor_day'/(name+'.part')] if p.exists()),0))
        for name,(_,total,_) in FILES.items())
    assert shutil.disk_usage(args.out).free>remaining+1024**3
    validators=load_validators(args.validator_source)
    save(args.out/'launch.json',dict(pid=os.getpid(),started_utc=now(),source_sha256=sha(__file__),
        validator_source=str(args.validator_source),validator_sha256=sha(args.validator_source),
        real_scores_computed=False,parallel_downloads=2))
    completed={};errors={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(fetch,name,args,validators):name for name in FILES}
        for future in as_completed(futures):
            name=futures[future]
            try:completed[name]=future.result()
            except Exception as exc:errors[name]=dict(type=type(exc).__name__,message=str(exc))
            save(args.out/'status.json',dict(utc=now(),complete=completed,errors=errors,
                real_scores_computed=False,status='running' if len(completed)+len(errors)<3 else 'finished'))
    save(args.out/'completion.json',dict(status='complete' if len(completed)==3 else 'partial',
        finished_utc=now(),files=completed,errors=errors,real_scores_computed=False))
    return 0 if len(completed)==3 else 1


if __name__=='__main__':raise SystemExit(main())
