"""Read-only output-range audit after the complete HQF inference and scoring."""
import argparse,datetime,json,pathlib,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);a=p.parse_args()
root=a.root;protocol=json.loads((root/'protocol/hqf_repair_pair_protocol.json').read_text())
assert json.loads((root/'completion.json').read_text())['status']=='complete'
out=dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='report-only output diagnostics; no selection or training',per_arm={})
for run in protocol['runs']:
    per={};allmean=[];allstd=[];allflat=[];allblack=[]
    for seq in protocol['sequences']:
        manifest=root/'results/inference'/run['run_id']/seq/'inference_manifest.json'
        d=json.loads(manifest.read_text());means=[];stds=[]
        for record in d['registered_inference']['saved_images']:
            array=np.load(record['float_array_path'],allow_pickle=False)
            assert array.dtype==np.float32 and array.shape==(180,240) and np.isfinite(array).all()
            assert array.min()>=0 and array.max()<=1
            means.append(float(array.mean(dtype=np.float64)));stds.append(float(array.std(dtype=np.float64)))
        mean=np.array(means);std=np.array(stds);flat=std<.001;black=(mean<.01)&flat
        per[seq]=dict(frames=len(mean),mean_brightness=float(mean.mean()),mean_spatial_std=float(std.mean()),flat_frame_fraction=float(flat.mean()),near_black_frame_fraction=float(black.mean()))
        allmean.extend(means);allstd.extend(stds);allflat.extend(flat.tolist());allblack.extend(black.tolist())
    assert len(allmean)==15498
    out['per_arm'][run['run_id']]=dict(per_sequence=per,frames=len(allmean),equal_sequence_mean={k:float(np.mean([v[k] for v in per.values()])) for k in ['mean_brightness','mean_spatial_std','flat_frame_fraction','near_black_frame_fraction']},frame_weighted_flat_fraction=float(np.mean(allflat)),frame_weighted_near_black_fraction=float(np.mean(allblack)))
(root/'prediction_range_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:{x:y for x,y in v.items() if x!='per_sequence'} for k,v in out['per_arm'].items()}),flush=True)
