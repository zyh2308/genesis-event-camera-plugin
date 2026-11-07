"""Wait for verified native inputs, preserve the entire left sensor, then resume upload."""
import argparse,base64,datetime,hashlib,importlib.util,json,os,shlex,shutil,subprocess,sys,time,zipfile
from pathlib import Path
import h5py,numpy as np

TARGET='/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-EINexus-left-20261107-v1/outdoor_day'
TASK_OUTPUT=None
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(obj,indent=2)+'\n');temp.replace(path)
def array_digest(dataset):
    h=hashlib.sha256()
    if dataset.ndim==0:h.update(np.asarray(dataset[()]).tobytes())
    else:
        step=max(1,8*1024**2//max(1,int(np.prod(dataset.shape[1:]))*dataset.dtype.itemsize))
        for i in range(0,len(dataset),step):h.update(np.asarray(dataset[i:i+step]).tobytes())
    return dict(shape=list(dataset.shape),dtype=str(dataset.dtype),array_sha256=h.hexdigest())
def project(source,target):
    assert not target.exists() and not source.is_symlink()
    evidence={}
    with h5py.File(source,'r') as src,h5py.File(target,'x') as dst:
        dst.attrs.update(src.attrs);group=dst.create_group('davis');group.attrs.update(src['davis'].attrs)
        src.copy('davis/left',group,name='left')
        names=[];src['davis/left'].visititems(lambda n,x:names.append(n) if isinstance(x,h5py.Dataset) else None)
        assert names
        for name in names:
            a,b=src['davis/left/'+name],dst['davis/left/'+name]
            first=array_digest(a);second=array_digest(b);assert first==second,'A left-sensor array changed: '+name
            evidence['davis/left/'+name]=first
    return dict(source=str(source),source_sha256=sha(source),path=str(target),size_bytes=target.stat().st_size,
        sha256=sha(target),left_arrays=evidence,scope='Lossless entire-left-sensor projection; only unused right/other sensor groups omitted',
        event_subsampling=False,coordinate_or_time_conversion=False,GT_mask_modified=False)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['native','out','transfer-base','calibration-zip']:p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=False)
    global TASK_OUTPUT
    TASK_OUTPUT=root
    save(root/'launch.json',dict(pid=os.getpid(),utc=now(),argv=sys.argv,status='waiting_for_native_inputs'))
    deadline=time.monotonic()+6*3600
    while not (args.native/'completion.json').exists():
        if (args.native/'failure.json').exists():raise RuntimeError('Native mirror download failed; no projection or upload')
        if time.monotonic()>deadline:raise TimeoutError('Native download not ready within 6 hours')
        time.sleep(15)
    ready=json.loads((args.native/'completion.json').read_text());assert ready['status']=='complete' and not ready['real_scores_computed']
    assert json.loads((args.native/'mirror_reference_checks.json').read_text())['status']=='passed'
    assert shutil.disk_usage(root).free>20*1024**3
    data=root/'outdoor_day';data.mkdir();projections={}
    for kind in ['data','gt']:
        name='outdoor_day1_'+kind+'.hdf5';record=next(r for r in ready['outputs'] if Path(r['path']).name==name)
        source=Path(record['path']);assert source.stat().st_size==record['size_bytes'] and sha(source)==record['sha256']
        projections[kind]=project(source,data/name)
        save(root/'projection_manifest.json',dict(status='projecting',files=projections))
    calibration=data/'outdoor_day_calib.zip';shutil.copyfile(args.calibration_zip,calibration)
    with zipfile.ZipFile(calibration) as f:assert f.testzip() is None
    registration=dict(schema='mvsec_lossless_left_projection_inputs_v1',source_native_completion_sha256=sha(args.native/'completion.json'),
        files={'outdoor_day1':{k:v for kind,item in projections.items() for k,v in [(kind+'_size_bytes',item['size_bytes']),
            (kind+'_sha256',item['sha256']),(kind+'_path_on_g1',TARGET+'/'+Path(item['path']).name)]}},
        calibration_files={'outdoor_day':dict(size_bytes=calibration.stat().st_size,sha256=sha(calibration),path_on_g1=TARGET+'/'+calibration.name)},
        scientific_invariance='Every dataset under davis/left preserved bytewise in array values, dtype and shape',projections=projections)
    regpath=root/'input_registration.json';save(regpath,registration)
    spec=importlib.util.spec_from_file_location('pinned_input_transfer',args.transfer_base)
    transfer=importlib.util.module_from_spec(spec);spec.loader.exec_module(transfer)
    transfer.TARGET=TARGET;transfer.SOURCE=str(data);transfer.REG_SHA=sha(regpath)
    transfer.NAMES=('outdoor_day1_data.hdf5','outdoor_day1_gt.hdf5','outdoor_day_calib.zip')
    transfer.SSH_OPTIONS=[x.replace('ConnectTimeout=20','ConnectTimeout=360') for x in transfer.SSH_OPTIONS]+['-o','ConnectionAttempts=1']
    def specs(path,selected):
        assert sha(path)==transfer.REG_SHA
        records=[]
        for name in selected:
            assert name in transfer.NAMES
            if name.endswith('.hdf5'):
                kind=name[:-5].rsplit('_',1)[1];r=registration['files']['outdoor_day1']
                records.append(dict(name=name,source=str(data/name),target=TARGET+'/'+name,
                    expected_bytes=r[kind+'_size_bytes'],expected_sha256=r[kind+'_sha256']))
            else:
                r=registration['calibration_files']['outdoor_day']
                records.append(dict(name=name,source=str(data/name),target=TARGET+'/'+name,expected_bytes=r['size_bytes'],expected_sha256=r['sha256']))
        return records
    transfer.specs=specs
    rsync_base=transfer.rsync_command
    transfer.rsync_command=lambda item,bwlimit:rsync_base(item,bwlimit)[:1]+['--compress','--compress-level=3']+rsync_base(item,bwlimit)[1:]
    save(root/'projection_manifest.json',dict(status='complete',files=projections,input_registration_sha256=sha(regpath),
        scientific_invariance=registration['scientific_invariance'],lossless_wire_compression='rsync zlib level3'))
    sys.argv=[str(args.transfer_base),'--registration',str(regpath),'--execute','--writer-check-confirmed',
        '--bwlimit-kib','16384','--out',str(root/'transfer')]
    # The remote directory is a newly allocated, isolated input-only root. This
    # check acquires the same nonblocking writer lock used by the actual rsync.
    guard="""import fcntl,json
from pathlib import Path
p=Path(%r)
assert p.is_dir() and p.resolve()==p
owner=json.loads((p.parent/'owner.json').read_text())
assert owner['run_id']=='MVSEC-EINexus-left-20261107-v1'
f=(p/'.stage9_mvsec_transfer.lock').open('a+')
fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('WRITER_LOCK_FREE')
"""%TARGET
    with (root/'remote_writer_check.log').open('x') as log:
        subprocess.run([transfer.SSH,*transfer.SSH_OPTIONS,transfer.HOST,transfer.ssh_program(guard)],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=420,check=True)
    try:transfer.main()
    except RuntimeError as exc:
        errors='\n'.join(p.read_text(errors='replace')[-2500:] for p in (root/'transfer').glob('*.stderr.log'))
        if 'failed with exit 255' not in str(exc) or not any(x in errors.lower() for x in ['timed out','timeout']):raise
        save(root/'one_retry_decision.json',dict(utc=now(),reason='recorded SSH timeout only',cooldown_seconds=60,attempts_after_failure=1))
        time.sleep(60)
        with (root/'remote_writer_recheck.log').open('x') as log:
            subprocess.run([transfer.SSH,*transfer.SSH_OPTIONS,transfer.HOST,transfer.ssh_program(guard)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=420,check=True)
        sys.argv[sys.argv.index('--out')+1]=str(root/'transfer_retry1')
        transfer.main()
    required={kind:{key:value for key,value in rec['left_arrays'].items() if key in
        ['davis/left/events','davis/left/image_raw','davis/left/image_raw_ts','davis/left/image_raw_event_inds','davis/left/flow_dist','davis/left/flow_dist_ts']}
        for kind,rec in projections.items()}
    assert len(required['data'])==4 and len(required['gt'])==2
    payload=base64.b64encode(json.dumps(dict(registration=registration,required_arrays=required)).encode()).decode()
    remote="""import base64,fcntl,hashlib,json
from pathlib import Path
import h5py,numpy as np,zipfile
root=Path(%r)
f=(root/'.stage9_mvsec_transfer.lock').open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
receipt=json.loads(base64.b64decode(%r));required=receipt['required_arrays']
for kind,arrays in required.items():
 with h5py.File(root/('outdoor_day1_'+kind+'.hdf5'),'r') as hf:
  for name,expected in arrays.items():
   d=hf[name];assert list(d.shape)==expected['shape'] and str(d.dtype)==expected['dtype'];h=hashlib.sha256()
   step=max(1,8*1024**2//max(1,int(np.prod(d.shape[1:]))*d.dtype.itemsize))
   for i in range(0,len(d),step):h.update(np.asarray(d[i:i+step]).tobytes())
   assert h.hexdigest()==expected['array_sha256'],name
with zipfile.ZipFile(root/'outdoor_day_calib.zip') as z:assert z.testzip() is None
receipt.update(status='verified_native_left_inputs_ready_unscored',real_scores_computed=False,all_required_arrays_equal_to_source=True)
out=root.parent/'verified_input_receipt.json'
with out.open('x') as f:json.dump(receipt,f,indent=2)
print('MVSEC_LEFT_INPUTS_VERIFIED_NO_SCORING')
"""%(TARGET,payload)
    py='/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28/bin/python'
    with (root/'remote_native_validation.log').open('x') as log:
        subprocess.run([transfer.SSH,*transfer.SSH_OPTIONS,transfer.HOST,shlex.join([py,'-c',remote])],
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
    save(root/'completion.json',dict(status='complete',utc=now(),real_scores_computed=False,server_target=TARGET,input_registration_sha256=sha(regpath)))

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if TASK_OUTPUT is not None:
            save(TASK_OUTPUT/'failure.json',dict(type=type(exc).__name__,error=str(exc),utc=now()))
        raise
