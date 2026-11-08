"""Mini CPU fixtures; no project bank, model, GPU, or training."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import event_support as S


def physical_fixture():
    ids=np.array([4,17,8,37,16,2],np.int32);objects=np.tile(np.eye(4),(21,6,1,1))
    for j in range(6):objects[:,j,0,3]=j+np.arange(21)*.01
    target=objects[:,3].copy();seg=np.full((21,260,346),-1,np.int32);depth=np.full(seg.shape,np.inf,np.float32)
    for i in range(21):seg[i,40:45,20+i:30+i]=37;depth[i][seg[i]==37]=1.2
    corners=np.tile(np.array([[20,40],[30,40],[20,45],[30,45],[21,41],[29,41],[21,44],[29,44]],float),(21,1,1))
    corners[:,:,0]+=np.arange(21)[:,None]
    return dict(object_segmentation_link_ids=ids,object_to_world=objects,target_pose=target,
        segmentation_link_idx=seg,depth_m=depth,target_visible_pixels=np.full(21,50,np.int64),
        frame_timestamps_s=np.arange(21)*.01,projected_corners_uv=corners,
        projected_centroid_uv=corners.mean(1),camera_intrinsics_K_px=np.eye(3),camera_to_world_cv=np.tile(np.eye(4),(21,1,1)),cad_keypoints=np.zeros((9,3)))


def region(mask):
    return S.RegionMask(np.asarray(mask,bool),dict(kind='physical_instance_endpoint_exact',visibility_exact_at_frame=True,event_time_visibility_exact=False,timestamp_s=.05))


class EventSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):
        import torch
        torch.set_num_threads(cls.threads)
    def test_target_identity_is_dynamic_and_double_proof_rejects_bad_count_depth_or_pose(self):
        spec=importlib.util.spec_from_file_location('support_unregistered_fixture',Path(S.__file__))
        fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
        self.assertEqual(fresh.RegionMask(np.ones((2,2),bool),dict(kind='fixture')).area,4)
        g=physical_fixture();masks,t,slot,target_id=S.resolve_episode(g)
        self.assertEqual((slot,target_id),(3,37));self.assertEqual(masks[9].area,50)
        for key in ('target_visible_pixels','depth_m','object_to_world'):
            bad=copy.deepcopy(g)
            if key=='target_visible_pixels':bad[key][8]=49
            elif key=='depth_m':bad[key][8,40,28]=np.nan
            else:bad[key][:,2]=bad['target_pose']
            with self.assertRaises(ValueError):S.resolve_episode(bad)

    def test_actual_hdf_reader_keeps_endpoint_vs_union_and_missing_window_failure(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            bank=Path(tmp);(bank/'physical').mkdir();rid='validation_fixture';path=bank/'physical'/(rid+'.h5');g=physical_fixture()
            with h5py.File(path,'w') as f:
                group=f.create_group('episodes/'+rid)
                for k,v in g.items():group.create_dataset(k,data=v)
            row=dict(id=rid+'/004',rid=rid,K=g['camera_intrinsics_K_px'],uv=np.array([g['projected_centroid_uv'][5]/[346.,260.]],np.float32))
            value=S.regions_for_rows(bank,[row],strict=True)[row['id']]
            self.assertEqual(value['endpoint'].area,50);self.assertEqual(value['endpoint_union_approx'].area,55)
            self.assertTrue(value['endpoint'].metadata['visibility_exact_at_frame'])
            self.assertFalse(value['endpoint_union_approx'].metadata['event_time_visibility_exact'])
            self.assertAlmostEqual(value['motion']['max_projected_corner_displacement_px'],1.)
            self.assertEqual(value['geometric_sanity']['visible_outside_hull_dilated1px'],0)
            self.assertTrue(value['geometric_sanity']['GT_anchor_inside_projected_bbox_hull'])
            matched=S.regions_for_rows(bank,[row],strict=True);self.assertTrue(S.verify_matched_regions({row['id']:value},matched)['passed'])
            matched[row['id']]['physical_contract_sha256']='0'*64
            with self.assertRaises(ValueError):S.verify_matched_regions({row['id']:value},matched)
            bad=dict(row,id='missing/004',rid='missing');failed=S.regions_for_rows(bank,[bad])
            self.assertEqual(len(failed),1);self.assertEqual(failed['missing/004']['status'],'failed')

    def test_fractional_mapping_matches_original_operators_and_area_normalized_attention(self):
        import cv2,torch
        from torch.nn import functional as F
        mask=np.zeros((260,346),bool);mask[:130,:173]=True;r=region(mask);before=mask.copy();p=S.region_pyramid(r)
        self.assertTrue(mask.flags.writeable)
        with self.assertRaises(ValueError):r.mask[0,0]=False
        direct=F.interpolate(F.interpolate(torch.from_numpy(cv2.resize(mask.astype(np.float32),(416,416),interpolation=cv2.INTER_LINEAR))[None,None],(128,128),mode='area'),(16,16),mode='area')[0,0].numpy()
        np.testing.assert_array_equal(p['attention16'],direct);np.testing.assert_array_equal(mask,before)
        a=np.full((16,16),1/256.);summary=S.summarize_attention(r,a,[.9,.9],pyramid=p)
        self.assertAlmostEqual(summary['region_attention_mass'],p['region_fraction_attention'])
        self.assertAlmostEqual(summary['region_mass_enrichment'],1.)
        self.assertFalse(summary['prediction_in_region']);self.assertTrue(summary['peak_in_region'])
        wrong=S.region_pyramid(region(~mask))
        with self.assertRaises(ValueError):S.summarize_attention(r,a,[.5,.5],pyramid=wrong)

    def test_empty_unavailable_regions_and_nonfinite_attention_do_not_become_quality_zero(self):
        r=region(np.zeros((260,346),bool));s=S.summarize_attention(r,np.full((16,16),1/256.),[.5,.5])
        self.assertFalse(s['available']);self.assertIsNone(s['region_attention_mass']);self.assertIsNone(s['region_mass_enrichment']);self.assertIsNone(s['prediction_in_region'])
        unavailable=S.RegionMask(None,dict(kind='unavailable'));self.assertIsNone(S.region_pyramid(unavailable)['attention16'])
        with self.assertRaises(ValueError):S.summarize_attention(r,np.full((16,16),np.nan),[.5,.5])
        full=region(np.ones((260,346),bool));self.assertFalse(S.point_membership(full,[1.,.5])['inside'])

    def test_causal_features_match_fixed_baf_and_do_not_read_future_or_reset_per_window(self):
        # Local independent scalar official rule is an oracle, not another threshold.
        rng=np.random.default_rng(272);t=np.cumsum(rng.integers(0,600,400)).astype(float)/1e6
        x,y=rng.integers(0,8,(2,len(t)));originals=[v.copy() for v in (t,x,y)]
        actual=S.causal_features(t,x,y,8,8);surface=np.zeros((8,8));keep=[]
        for now,xx,yy in zip(t*1e6,x,y):
            neighbours=[surface[b,a] for b in range(max(0,yy-1),min(8,yy+2)) for a in range(max(0,xx-1),min(8,xx+2)) if (a,b)!=(xx,yy)]
            keep.append(any(now-v<2000 for v in neighbours));surface[yy,xx]=now
        np.testing.assert_array_equal(actual['baf_keep'],keep)
        prefix=S.causal_features(t[:175],x[:175],y[:175],8,8)
        for key in ('baf_keep','neighbour8_age_us','same_pixel_interval_us'):np.testing.assert_array_equal(prefix[key],actual[key][:175])
        for a,b in zip(originals,(t,x,y)):np.testing.assert_array_equal(a,b)
        t=np.array([.003,.004,.006]);x=np.array([2,3,4]);y=np.array([2,2,2]);v=S.causal_features(t,x,y,8,8)
        np.testing.assert_array_equal(v['baf_keep'],[False,True,False])

    def test_window_endpoint_vs_union_region_retention_empty_and_neighbor_scope(self):
        t=np.array([.003,.003001,.003002]);x=np.array([1,2,3]);y=np.array([1,1,1]);features=S.causal_features(t,x,y,6,4)
        mask=np.zeros((4,6),bool);mask[1,2]=True;r=region(mask)
        value=S.window_support(t,x,y,features['baf_keep'],.003,.003002,r,features)
        self.assertEqual(value['total_raw_events'],2);self.assertEqual(value['foreground']['raw_events'],1)
        self.assertEqual(value['foreground']['raw_same_window_occupied_neighbour8_mean'],1.)
        self.assertEqual(value['foreground']['raw_same_window_same_region_occupied_neighbour8_mean'],0.)
        empty=S.window_support(t,x,y,features['baf_keep'],.02,.03,r,features)
        self.assertTrue(empty['empty_window']);self.assertIsNone(empty['foreground']['event_retention_ratio'])
        self.assertEqual(empty['foreground']['raw_occupancy_ratio'],0.)
        zero=region(np.zeros((4,6),bool));stats=S.window_support(t,x,y,features['baf_keep'],0.,.02,zero,features)
        self.assertIsNone(stats['foreground']['raw_occupancy_ratio']);json.dumps(stats,allow_nan=False)


if __name__=='__main__':unittest.main(verbosity=2)
