"""Actual frozen reconstruction checkpoint and metric checks on fabricated data."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('repo','campaign','model-root','evreal-root','out'):
        p.add_argument('--'+n,type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    args.out.mkdir(exist_ok=False)
    folder=args.out/'synthetic_fixture'
    folder.mkdir()
    (folder/'synthetic_fixture_manifest.json').write_text(json.dumps(dict(real_data_used=False,
        real_scoring_enabled=False,status='synthetic_inference_and_metric_plumbing_only')))
    fixture=folder/'bike_bay_hdr';fixture.mkdir()
    times=np.linspace(.001,.036,100,dtype=np.float64)
    stamps=np.asarray([.005,.015,.025,.035],dtype=np.float64).reshape(-1,1)
    rng=np.random.default_rng(20261107)
    arrays={'events_ts.npy':times,'events_xy.npy':np.column_stack((rng.integers(240,size=100),rng.integers(180,size=100))).astype(np.int16),
        'events_p.npy':rng.integers(2,size=100,dtype=np.uint8),'images_ts.npy':stamps,
        'image_event_indices.npy':np.searchsorted(times,stamps.ravel(),side='right').astype(np.int64).reshape(-1,1)}
    for name,array in arrays.items():np.save(fixture/name,array,allow_pickle=False)
    (fixture/'metadata.json').write_text(json.dumps(dict(sensor_resolution=[180,240])))
    sources=args.repo/'benchmark/sim2real/scripts'
    receipt=json.loads((args.campaign/'training/reconstruction/seed0/eventgenesis/completion.json').read_text())
    checkpoint=Path(receipt['selected_checkpoint'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==receipt['selected_checkpoint_sha256']
    outputs=[]
    for case,value in [('zero_reference',0),('white_reference',255)]:
        np.save(fixture/'images.npy',np.full((4,180,240,1),value,dtype=np.uint8),allow_pickle=False)
        target=args.out/case
        command=[sys.executable,'-u',str(sources/'infer_hqf_e2vid_candidate.py'),
             '--evreal-root',str(args.evreal_root),'--model-root',str(args.model_root),
             '--checkpoint',str(checkpoint),'--model-config',str(args.campaign/'sources/training_recipe.json'),
             '--synthetic-fixture-sequence',str(fixture),'--output-root',str(target),'--device','cpu',
             '--save-every','1','--save-float-arrays']
        with (args.out/(case+'.log')).open('x') as log:
            run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert run.returncode==0,case
        arrays_out={str(x.relative_to(target)):np.load(x,allow_pickle=False) for x in target.rglob('*.npy')}
        assert len(arrays_out)==3,(case,list(arrays_out))
        assert all(x.shape==(180,240) and np.isfinite(x).all() for x in arrays_out.values())
        outputs.append(arrays_out)
    assert outputs[0].keys()==outputs[1].keys()
    assert all(np.array_equal(outputs[0][k],outputs[1][k]) for k in outputs[0])
    spec=importlib.util.spec_from_file_location('hqf_metric_fixture',sources/'evaluate_synthetic_reconstruction_fixture.py')
    metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)
    zero=np.zeros((64,64),dtype=np.float32)
    identical=metric_module.classical_metrics(zero,zero)
    assert identical['mse']==0 and identical['ssim']==1 and math.isinf(identical['psnr_db'])
    metric,*_=metric_module.load_lpips_alex(args.model_root,torch.device('cpu'))
    with torch.inference_mode():
        tensor=metric_module.lpips_input(zero,torch.device('cpu'))
        distance=float(metric.forward(tensor,tensor,normalize=False).reshape(-1)[0])
    assert math.isfinite(distance) and abs(distance)<1e-7
    (args.out/'completion.json').write_text(json.dumps(dict(status='passed',real_data_used=False,
        actual_checkpoint_sha256=receipt['selected_checkpoint_sha256'],output_frames_per_case=3,
        native_shape=[180,240],reference_pixel_change_does_not_change_predictions=True,
        identical_pair_mse=0,identical_pair_ssim=1,identical_pair_psnr_positive_infinity=True,
        identical_pair_lpips=distance,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print('HQF_SYNTHETIC_RUNTIME_PREFLIGHT_PASSED',flush=True)


if __name__=='__main__':main()
