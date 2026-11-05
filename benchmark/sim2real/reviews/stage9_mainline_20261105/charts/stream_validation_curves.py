"""Single-pass synthetic-validation extraction; never opens pose training logs.

No Torch/NumPy/GPU/network. One CPU worker; OMP/MKL/OpenBLAS bounded to2.
Active files are bounded to their exact opening-size prefix; incomplete trailing
bytes are hashed, explicitly reported, and never interpreted as a JSON record.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='2'
import argparse,csv,hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
BUDGETS={'reconstruction':12000,'flow':8910}
METRICS={
    'reconstruction':(('lpips_vgg_equal_rollout_mean','LPIPS-VGG',True),
        ('temporal_consistency_equal_rollout_mean','TC loss',False)),
    'flow':(('model_mean_epe_px_per_30ms','px/30ms',True),
        ('zero_flow_mean_epe_px_per_30ms','px/30ms',True)),
}
FIELDS=('task','method','seed','optimizer_update','epoch','metric','value','unit',
    'source','source_line','recorded_utc','source_sha256','source_prefix_bytes',
    'source_changed_during_scan','point_scope','real_evaluation_status')

def extract(path,task,method,seed,max_line_bytes):
    record=dict(task=task,method=method,seed=seed,path=str(path),status='MISSING_LOG',validation_records=0)
    if not path.is_file():return [],record
    assert task in BUDGETS and '/pose/' not in path.as_posix()
    rows=[];seen={};h=hashlib.sha256();unknown={};trailing=0;line_no=0;consumed=0
    with path.open('rb') as f:
        initial=os.fstat(f.fileno());remaining=initial.st_size
        while remaining:
            line=f.readline(min(remaining,max_line_bytes))
            if not line:raise RuntimeError('Source truncated during scan: '+str(path))
            consumed+=len(line);remaining-=len(line);h.update(line);line_no+=1
            if not line.endswith(b'\n'):
                if remaining:raise RuntimeError(f'Line exceeds bound {max_line_bytes}: {path}:{line_no}')
                trailing=len(line);break
            if not line.strip():continue
            try:value=json.loads(line)
            except (UnicodeDecodeError,json.JSONDecodeError) as exc:
                raise RuntimeError(f'Invalid complete JSON line: {path}:{line_no}') from exc
            if not isinstance(value,dict):raise RuntimeError(f'JSON record is not a dict: {path}:{line_no}')
            event=value.get('event')
            if event!='synthetic_validation':
                if isinstance(event,str) and 'valid' in event.lower():unknown[event]=unknown.get(event,0)+1
                continue
            if value.get('real_data_used') is True or value.get('real_scoring_enabled') is True:
                raise RuntimeError(f'Validation record violates synthetic isolation: {path}:{line_no}')
            if 'generator' in value and value['generator']!=method:
                raise RuntimeError(f'Generator identity mismatch: {path}:{line_no}')
            if 'seed' in value and value['seed']!=seed:
                raise RuntimeError(f'Seed identity mismatch: {path}:{line_no}')
            update=value.get('optimizer_update',value.get('optimizer_updates'))
            if type(update) is not int or not 0<=update<=BUDGETS[task]:
                raise RuntimeError(f'Invalid optimizer update: {path}:{line_no}')
            if 'optimizer_update' in value and 'optimizer_updates' in value and value['optimizer_update']!=value['optimizer_updates']:
                raise RuntimeError(f'Ambiguous optimizer update: {path}:{line_no}')
            record['validation_records']+=1
            for metric,unit,required in METRICS[task]:
                if metric not in value:
                    if required:raise RuntimeError(f'Missing primary validation metric {metric}: {path}:{line_no}')
                    continue
                number=value[metric]
                if type(number) not in (int,float) or not math.isfinite(number):
                    raise RuntimeError(f'Nonfinite validation metric {metric}: {path}:{line_no}')
                key=(update,metric)
                if key in seen:
                    if seen[key]!=number:raise RuntimeError(f'Conflicting duplicate validation: {path}:{line_no}')
                    continue
                seen[key]=number
                rows.append(dict(task=task,method=method,seed=seed,optimizer_update=update,epoch=value.get('epoch',''),
                    metric=metric,value=number,unit=unit,source=str(path),source_line=line_no,
                    recorded_utc=value.get('recorded_utc',value.get('updated_utc','')),
                    point_scope='synthetic_validation',real_evaluation_status='NOT_EVALUATED'))
        after=os.fstat(f.fileno())
    changed=(initial.st_size!=after.st_size or initial.st_mtime_ns!=after.st_mtime_ns)
    digest=h.hexdigest()
    record.update(status='PREFIX_WITH_TRAILING_PARTIAL_LINE' if trailing else 'OPENING_PREFIX_SCANNED',
        source_sha256=digest,source_prefix_bytes=consumed,source_size_at_open=initial.st_size,
        source_size_after_scan=after.st_size,source_changed_during_scan=changed,
        trailing_partial_bytes=trailing,complete_lines=line_no-(1 if trailing else 0),
        ignored_other_validation_event_counts=unknown,exported_metric_rows=len(rows),
        source_mtime_ns_at_open=initial.st_mtime_ns)
    assert consumed==initial.st_size
    for row in rows:row.update(source_sha256=digest,source_prefix_bytes=consumed,source_changed_during_scan=changed)
    return rows,record

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',choices=tuple(BUDGETS),default=list(BUDGETS))
    p.add_argument('--max-line-bytes',type=int,default=8*1024*1024)
    a=p.parse_args();campaign=a.campaign.resolve(strict=True)
    if a.out.exists():raise FileExistsError('Refusing existing extraction output: '+str(a.out))
    rows=[];sources=[]
    for task in a.tasks:
        for seed in range(3):
            for method in METHODS:
                path=campaign/'training'/task/f'seed{seed}'/method/'training_metrics.jsonl'
                extracted,record=extract(path,task,method,seed,a.max_line_bytes);rows.extend(extracted);sources.append(record)
    if not rows:raise RuntimeError('No recognized synthetic validation records; no output created')
    a.out.mkdir(parents=True)
    target=a.out/'validation_points.csv'
    with target.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    manifest=dict(schema='stage9_streamed_synthetic_validation_prefixes_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        campaign=str(campaign),tasks=a.tasks,identity='campaign/training/task/seedN/method path; log metadata checked when present',
        only_event_accepted='synthetic_validation',pose_training_logs_read=False,
        real_data_or_weights_opened=False,real_evaluation_status='NOT_EVALUATED',
        cpu_workers=1,thread_env={k:os.environ[k] for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        source_hash_scope='SHA256 of exact opening-size file prefix, including any unparsed incomplete trailing bytes',
        sources=sources,metric_rows=len(rows),csv_path=str(target),csv_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with (a.out/'source_manifest.json').open('x',encoding='utf-8') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='synthetic_validation_extracted',metric_rows=len(rows),
        sources_found=sum(x['status']!='MISSING_LOG' for x in sources),sources_missing=sum(x['status']=='MISSING_LOG' for x in sources),
        changed_sources=sum(x.get('source_changed_during_scan',False) for x in sources),out=str(a.out)),indent=2))

if __name__=='__main__':main()
