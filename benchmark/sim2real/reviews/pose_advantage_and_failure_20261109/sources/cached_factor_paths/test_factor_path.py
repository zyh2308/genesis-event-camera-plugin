import copy, itertools, math, unittest
from pathlib import Path
import numpy as np
import factor_path as d

H=d.load_helper(Path(__file__).resolve().parents[2]/'pose_factor_diagnostic_20261108/factor_geometry.py')

def fixture():
 verts=np.array(list(itertools.product([-.02,.02],repeat=3)));anchor=np.array([.017,-.008,.011])
 keys=np.vstack((anchor,verts));assets=H.FactorAssets(verts,keys,.04*math.sqrt(3))
 a=.7;R=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
 T=np.eye(4);T[:3,:3]=R;T[:3,3]=[.03,-.02,.6];K=np.array([[300.,0,173],[0,301,130],[0,0,1]])
 xyz=R@anchor+T[:3,3];p=K@xyz;uv=p[:2]/p[2]/H.PIXELS
 rec=dict(method='eventgenesis',seed=0,condition='normal',rid='r0',sample_id='w0',K=K.tolist(),T_gt=T.tolist(),anchor=anchor.tolist(),uv_gt=uv.tolist(),gt_log_size=math.log(.1),gt_log_shape=0.)
 flat=[]
 for arm in ('C0','C1','slow'):
  flat.append(dict(rec,arm=arm,pred_uv=(uv+np.array([.08,-.04])).tolist(),z_m=float(xyz[2]),R_pred=np.eye(3).tolist(),size=.1,R_crop_gt_center=R.tolist()))
 return flat,assets

class Tests(unittest.TestCase):
 def test_cached_cells_and_new_crop_path(self):
  flat,assets=fixture();rec=H.group_flat_predictions(flat)[0];identity={k:rec[k] for k in ('method','seed','condition','rid','sample_id')}
  scores=[dict(identity,**x) for x in H.factor_oracles(rec,'C1',assets) if x['variant'].endswith('r0')]
  pg=H.aggregate_rollout_equal(scores)
  baseline=next(x for x in pg if x['variant']=='oracle/C1/u0z0r0')
  native=[dict(method='eventgenesis',seed=0,condition='normal',arm='C1',score=dict(summary=baseline['summary'],by_rollout=baseline['by_rollout']))]
  pred=dict(rec['predictions']['C1']);pred.update(uv=H._gt(rec)['uv'],z_m=H._gt(rec)['z_m'],R=np.asarray(flat[1]['R_crop_gt_center']))
  pcr=H.aggregate_rollout_equal([dict(identity,variant='rotation_crop/GTtranslation/GTcenter_crop_R',metrics=H.score_factors(rec,pred,assets))])
  before=copy.deepcopy(flat);macro,new,parity=d.assemble(H,flat,assets,native,pg,pcr,strict=False)
  self.assertEqual(flat,before);self.assertEqual(len(macro),8);self.assertEqual(len(new),1)
  self.assertTrue(all(p['status']=='passed' for p in parity.values()))
  held=next(x for x in macro if x['variant']=='held_R/u1z0')['summary']
  cropped=next(x for x in macro if x['variant']=='center_crop_R/u1z0')['summary']
  self.assertGreater(held['translation_error_m'],.005);self.assertLess(cropped['translation_error_m'],1e-14)
  self.assertEqual(cropped['success'],1.);self.assertGreater(held['rotation_error_deg'],30)
  self.assertEqual(cropped['depth_mae_m'],held['depth_mae_m'])
 def test_same_ids_wrong_label_rejected(self):
  flat,assets=fixture();flat[1]['T_gt'][0][3]+=.001
  with self.assertRaises(ValueError):H.group_flat_predictions(flat)
 def test_saved_rotation_must_be_so3(self):
  flat,assets=fixture();record=H.group_flat_predictions(flat)[0];flat[1]['R_crop_gt_center'][0][0]=5
  with self.assertRaises(ValueError):d.new_crop_center_score(H,record,flat[1],assets)

if __name__=='__main__':unittest.main()
