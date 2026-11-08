import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import audit_pose_data_contract_v2 as a
from run_serializable_audit import json_dump,json_proxy
a.json=json_proxy()


class AuditTests(unittest.TestCase):
    def test_histogram_quantiles_match_independent_numpy(self):
        rng=np.random.default_rng(7)
        for size in (1,2,20,90000):
            values=rng.integers(0,40,size=size)
            np.testing.assert_allclose(a.histogram_quantiles(values),np.quantile(values,[.5,.9,.99]),atol=1e-12,rtol=0)
        self.assertEqual(a.histogram_quantiles([]),[None,None,None])

    def test_exact_clip_threshold_is_eight_not_greater_than_eight(self):
        m=a.count_metrics(np.array([0,1,7,8,9]))
        self.assertEqual(m['nonzero_rate'],.8);self.assertEqual(m['at_least8_rate'],.4)
        self.assertEqual(m['greater_than8_rate'],.2)
        self.assertAlmostEqual(float(np.minimum(np.log1p(8)/np.log1p(8),1)),1)

    def test_fixture_wholeframe_identity_and_deliberate_representation_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            R=Path(tmp)/'campaign';R.mkdir();light=Path(tmp)/'relit';light.mkdir()
            sources=R/'sources/benchmark/sim2real/scripts';sources.mkdir(parents=True)
            source='''NETWORK_SIZE=16
def encode_event_window(timestamp_s,x,y,polarity,start_s,end_s,width,height,count_scale):
 selected=(timestamp_s>start_s)&(timestamp_s<=end_s)
 index=y[selected]*width+x[selected];p=polarity[selected]
 on=np.bincount(index[p==1],minlength=width*height).reshape(height,width)
 off=np.bincount(index[p==0],minlength=width*height).reshape(height,width)
 return np.stack([cv2.resize(np.minimum(np.log1p(v)/np.log1p(count_scale),1).astype(np.float32),(NETWORK_SIZE,NETWORK_SIZE),interpolation=cv2.INTER_LINEAR) for v in (on,off,on+off)])
'''
            (sources/'run_pose_stage2_pilot.py').write_text(source)
            encode,_=a.encoder(sources/'run_pose_stage2_pilot.py')
            vertices=np.array([[x,y,z] for x in (-.03,.03) for y in (-.02,.02) for z in (-.01,.01)],float)
            vertex_path=R/'vertices.npy';np.save(vertex_path,vertices)
            preparation=R/'recovery_20261105_mainline/pose_prepared_full_v5';preparation.mkdir(parents=True)
            (preparation/'protocol.json').write_text(json.dumps(dict(assets=dict(vertices=dict(path=str(vertex_path),sha256=a.sha(vertex_path))))))
            rid='train_000';times=np.arange(21)*.01;K=np.array([[200.,0,173],[0,200,130],[0,0,1]])
            keys=np.vstack([np.zeros(3),vertices]);transforms=np.repeat(np.eye(4)[None],21,axis=0);transforms[:,2,3]=.6
            raw=np.array([[times[k+1],173,130,1.] for k in range(4,20) for _ in range(9)],float)
            banks={'normal':R/'pose_bank','half':light/'half/pose_bank','eighth':light/'eighth/pose_bank'}
            for level,bank in banks.items():
                (bank/'physical').mkdir(parents=True)
                with h5py.File(bank/'physical'/f'{rid}.h5','w') as h:
                    g=h.create_group(f'episodes/{rid}');g.attrs['split']='train'
                    for name,value in [('frame_timestamps_s',times),('camera_intrinsics_K_px',K),('cad_keypoints',keys),('target_pose',transforms),('camera_to_world_cv',np.repeat(np.eye(4)[None],21,axis=0))]:g.create_dataset(name,data=value)
                manifest=dict(rollouts=[dict(rollout_id=rid,split='train')],real_data_used=False)
                (bank/'bank_manifest.json').write_text(json.dumps(manifest))
                for method in a.METHODS:
                    folder=bank/'events'/method;folder.mkdir(parents=True);np.save(folder/(rid+'.npy'),raw)
                    row=dict(rollout_id=rid,events_sha256=a.sha(folder/(rid+'.npy')),profile_sha256='profile',source_cadence_hz=100,command=['--pos_thres','.2'],weight_sha256='weights')
                    event=dict(rollouts=[row],real_data_used=False,physical_manifest_sha256=a.sha(bank/'bank_manifest.json'),core_release='frozen',sensor_profile_sha256='profile',source_commit='v2e')
                    (folder/'manifest.json').write_text(json.dumps(event))
            (light/'completion.json').write_text(json.dumps(dict(status='complete',real_data_used=False,selected=[rid])))
            images=np.stack([encode(raw[:,0],raw[:,1].astype(int),raw[:,2].astype(int),(raw[:,3]>0).astype(np.uint8),times[k],times[k+1],346,260,8.) for k in range(4,20)])
            with h5py.File(banks['normal']/'physical'/f'{rid}.h5') as h:geometry,_=a.geometry(h[f'episodes/{rid}'],rid,vertices)
            ids=list(geometry)
            for method in a.METHODS:
                folder=R/'adapters'/method/'pose'/method/'train';folder.mkdir(parents=True)
                np.savez(folder/(rid+'.npz'),images=images,uv_normalized=np.stack([geometry[s]['uv']/[346,260] for s in ids]).astype(np.float32),target_to_camera=np.stack([geometry[s]['T'] for s in ids]),K=K,cad_keypoints=keys,sample_ids=np.array(ids),label_times_s=times[5:21])
            out=Path(tmp)/'audit.json'
            argv=['audit','--campaign',str(R),'--radiance-train',str(light),'--out',str(out),'--representations']
            with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):a.main()
            result=json.loads(out.read_text());self.assertEqual(result['status'],'passed')
            self.assertEqual(result['split_counts'],{'train':1});self.assertEqual(result['normal_adapter']['eventgenesis']['windows'],16)
            self.assertEqual(result['expected_six_epoch_exposure']['half_expected_visits_per_window'],4.5)
            self.assertEqual(result['expected_six_epoch_exposure']['eighth_expected_visits_per_window'],1.5)
            self.assertEqual(result['representation']['eventgenesis']['normal_npz_vs_fresh_base_max_abs'],0)
            single={ids[0]:geometry[ids[0]]}
            with contextlib.redirect_stdout(io.StringIO()):
                info=a.information_audit({'train':banks},{'train':{level:single for level in banks}},encode,vertices)
            # Entire genuine input_information object, including its np.int64
            # event-count lists, must survive the final write boundary.
            with self.assertRaises(TypeError):json.dumps(dict(input_information=info),allow_nan=False)
            payload=dict(input_information=info,numpy_types=dict(
                signed=np.int64(2**60+1),unsigned=np.uint64(2**63+7),
                small=np.int32(17),single=np.float32(.25),double=np.float64(.5),
                boolean=np.bool_(True),array=np.array([2**60+1,2**60+3],dtype=np.int64)))
            serialized=Path(tmp)/'complete_information.json'
            with serialized.open('x') as stream:json_dump(payload,stream,indent=2,allow_nan=False)
            decoded=json.loads(serialized.read_text())
            self.assertEqual(decoded['numpy_types']['signed'],2**60+1)
            self.assertEqual(decoded['numpy_types']['unsigned'],2**63+7)
            self.assertEqual(decoded['numpy_types']['array'],[2**60+1,2**60+3])
            self.assertEqual(len(decoded['input_information']['summary']),len(info['summary']))
            self.assertEqual(len(decoded['input_information']['raw_event_evidence']),6)
            with self.assertRaises(ValueError):
                json_dump(dict(input_information=info,bad=np.float32(np.nan)),io.StringIO(),allow_nan=False)
            row=next(r for r in info['summary'] if r['split']=='train' and r['method']=='eventgenesis' and r['level']=='normal' and r['region']=='full' and r['channel']=='ON' and r['depth_bin']=='all_z')
            self.assertAlmostEqual(row['metrics']['greater_than8_rate']['window_equal_mean'],1/(346*260))
            self.assertFalse(info['GPU_used']);self.assertFalse(info['training_or_inference_crops'])
            path=R/'adapters/eventgenesis/pose/eventgenesis/train'/f'{rid}.npz'
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['images']=arrays['images'].copy();arrays['images'][0,0,0,0]=.5
            np.savez(path,**arrays);out2=Path(tmp)/'mismatch.json'
            with patch.object(sys,'argv',argv[:-2]+[str(out2),'--representations']),contextlib.redirect_stdout(io.StringIO()):a.main()
            mismatch=json.loads(out2.read_text());self.assertEqual(mismatch['status'],'mismatch')
            self.assertEqual(mismatch['representation']['eventgenesis']['mismatch_windows'],1)


if __name__=='__main__':unittest.main(verbosity=2)
