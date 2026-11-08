"""Posthoc segmentation ID at every original C1 peak, not model inputs."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--readout', type=Path, required=True)
    a.add_argument('--peaks', type=Path, required=True)
    a.add_argument('--out', type=Path, required=True)
    v = a.parse_args()
    assert not v.out.exists()
    rc = read(v.readout/'completion.json')
    pc = read(v.peaks/'completion.json')
    assert rc['paired_cases'] == pc['prediction_cases'] == 1152
    refs = {Path(r['path']).name: r for r in rc['artifacts']}
    cp = Path(refs['paired_cases.jsonl']['path'])
    assert sha(cp) == refs['paired_cases.jsonl']['sha256']
    h5s = {r['path']:r for r in rc['source_records'] if '/physical/validation_' in r['path']}
    assert len(h5s) == 12
    for p,r in h5s.items():
        assert sha(p) == r['sha256']
    key = lambda r: (r['method'],r['seed'],r['condition'],r['rid'],r['sample_id'])
    cases = {key(r):r for r in map(json.loads,cp.open())}
    peaks = list(map(json.loads,(v.peaks/'per_prediction.jsonl').open()))
    assert len(cases) == len(peaks) == 1152 and {key(r) for r in peaks} == set(cases)
    cache, groups, unsupported = {}, {}, []
    for row in peaks:
        case = cases[key(row)]
        candidates = [Path(p) for p in h5s if Path(p).stem == row['rid'] and
            (('/'+row['condition']+'/pose_bank/physical/') in p if row['condition']!='normal'
             else '/half/pose_bank/physical/' not in p and '/eighth/pose_bank/physical/' not in p)]
        assert len(candidates) == 1
        p = candidates[0]
        if str(p) not in cache:
            with h5py.File(p,'r') as f:
                g = f['episodes'][row['rid']]
                cache[str(p)] = (g['segmentation_link_idx'][:], int(g.attrs['target_link_id']))
        masks, target = cache[str(p)]
        xy = np.floor(np.asarray(case['peak_uv'])*[346,260]).astype(int)
        k = int(row['sample_id'].rsplit('/',1)[1])+1
        sid = int(masks[k,xy[1],xy[0]])
        assert (sid==target) == row['metrics']['endpoint_peak_point_hit']
        label = 'target' if sid==target else 'unrendered_background' if sid<0 else 'non_target_segmentation_id_'+str(sid)
        group = (row['method'],row['condition'])
        groups.setdefault(group,Counter())[label] += 1
        if row['metrics']['peak_category']=='no_target_support_in_endpoint_or_union_peak_cell':
            unsupported.append({k:row[k] for k in ('method','seed','condition','rid','sample_id')} |
                dict(segmentation_id=sid,target_segmentation_id=target,label=label))
    for p,r in h5s.items():
        assert sha(p) == r['sha256']
    result = dict(schema='original_C1_peak_surface_posthoc_v1',prediction_cases=1152,
        readout_completion_sha256=sha(v.readout/'completion.json'),peak_completion_sha256=sha(v.peaks/'completion.json'),
        per_condition=[dict(method=k[0],condition=k[1],counts=dict(c)) for k,c in sorted(groups.items())],
        no_target_cell_support=unsupported,optimizer_updates=0,model_forward_calls=0,GT_posthoc_only=True,
        interpretation='Segmentation IDs identify rendered surfaces, not event origins, noise labels or unique causal mechanisms.',
        script_sha256=sha(__file__))
    with v.out.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(unsupported=len(unsupported),labels=dict(Counter(r['label'] for r in unsupported)))))


if __name__=='__main__':
    main()
