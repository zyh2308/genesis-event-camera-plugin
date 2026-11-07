"""Capture small owned study evidence without exporting model or event arrays."""
import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real')
    names=['20261107-pose-light-'+s+'-v1' for s in ('robustness','anchor','geometry','interpolation','interpolation-refined')]
    names.append('20261107-pose-light-interpolation-refined-v2')
    names.append('20261107-pose-physical-light-train-v1')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=base/('pose_light_evidence_'+stamp+'.zip')
    files={}
    summaries={}
    campaign=base/'20261103-stage9-direct-v1'
    recovery=campaign/'recovery_20261105_mainline'
    dependencies=[recovery/n for n in ('pose_scale_pilot_20261106.py','pose_rotation_pilot_20261106.py',
        'pose_depth_pilot_20261106.py','pose_fresh_confirm_20261106_v3.py','stage9_epose_v5.py')]
    dependencies.extend([base/'20261106-pose-synthetic-diagnostic-v1/causal_tests_v2.py',
        campaign/'sources/benchmark/sim2real/scripts/run_pose_stage2_pilot.py',
        campaign/'sources/stage9_sensor_profile.json',
        recovery/'jobs/119_pose_fresh_confirm_paired_20261106/results/completion.json',
        recovery/'pose_prepared_full_v5/protocol.json'])
    for number,path in enumerate(dependencies):
        assert path.is_file(),str(path)
        files['dependencies/'+str(number).zfill(2)+'_'+path.name]=path
    for name in names:
        root=base/name
        summary={'root':str(root),'status':'running','real_data_used':False}
        for terminal in ('acquisition_blocked','completion','failure'):
            path=root/(terminal+'.json')
            if path.exists():summary.update(json.loads(path.read_text()))
        for folder in ('sources','logs','robustness','smoke','radiance_score','epose_diagnostic',
                       'radiance_score_retry1','radiance_score_retry1_sources','physical_adaptation'):
            for path in (root/folder).glob('*'):
                if path.is_file() and path.suffix in ('.py','.json','.log','.md'):
                    files[str(path.relative_to(base))]=path
        for path in root.glob('*'):
            if path.is_file() and path.suffix in ('.json','.log'):
                if path.name=='launcher.log' and (root/'acquisition_blocked.json').exists():continue
                files[str(path.relative_to(base))]=path
        for path in (root/'radiance').rglob('*.json'):
            files[str(path.relative_to(base))]=path
        for path in (root/'radiance_train').rglob('*.json'):
            files[str(path.relative_to(base))]=path
        completion=root/'robustness/completion.json'
        if completion.exists():
            result=json.loads(completion.read_text())
            summary['gate']=result.get('gate')
            summary['selected_alpha']=result.get('selected_alpha')
            aggregates={}
            for record in result['results']:
                method=record['method']
                for arm,conditions in record['arms'].items():
                    for condition,metrics in conditions.items():
                        key=method+'/'+arm+'/'+condition
                        values=aggregates.setdefault(key,{})
                        for metric,value in metrics['summary'].items():
                            values.setdefault(metric,[]).append(value)
            summary['aggregate']={k:{metric:{'mean':sum(vals)/len(vals),'values':vals}
                for metric,vals in value.items()} for k,value in aggregates.items()}
        weights={str(path.relative_to(root)):{'bytes':path.stat().st_size,'sha256':digest(path)}
                 for path in (root/'robustness').glob('*.pt')}
        summary['weights']=weights
        summaries[name]=summary
    receipts={}
    with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as archive:
        for relative,path in sorted(files.items()):
            payload=path.read_bytes()
            archive.writestr(relative,payload)
            receipts[relative]={'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        archive.writestr('SUMMARY.json',json.dumps({'captured_utc':stamp,'studies':summaries},indent=2)+'\n')
        archive.writestr('INDEX.json',json.dumps(receipts,indent=2)+'\n')
    print(json.dumps({'path':str(output),'bytes':output.stat().st_size,'sha256':digest(output),
                      'files':len(receipts),'captured_utc':stamp}),flush=True)


if __name__=='__main__':main()
