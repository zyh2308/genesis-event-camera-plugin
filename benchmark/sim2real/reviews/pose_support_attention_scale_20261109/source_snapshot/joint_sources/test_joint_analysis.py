"""Full-schema synthetic CPU fixtures; never read project runs/models/events."""
import copy,csv,itertools,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import analyze_joint as A
import attention_scale_metrics as M


def matrix_fixture():
    methods=('eventgenesis','v2e');lights=('normal','half','eighth');rids=('validation_000','validation_031','validation_063','validation_095')
    support=[];shared={}
    for method,light,rid,k in itertools.product(methods,lights,rids,range(4,20)):
        empty=k==4
        part=lambda area,n:dict(region_pixels=area,raw_events=0 if empty else n,baf_events=0 if empty else n//2,raw_occupancy_ratio=0. if empty else .1,baf_occupancy_ratio=0. if empty else .05,
            raw_same_window_spatially_isolated_event_fraction=None if empty else .3,raw_prior_neighbour8_within2ms_fraction=None if empty else .5)
        s=dict(method=method,condition=light,status='complete',rid=rid,sample_id=f'{rid}/{k:03}',start_s=k*.01,end_s=(k+1)*.01,
            empty_window=empty,regions={region:dict(foreground=part(100,10),background_region=part(89900,20)) for region in A.REGIONS})
        support.append(s);shared[(method,light,s['sample_id'])]=s
    rows=[]
    for method,seed,arm,branch,light,view,rid,k in itertools.product(methods,range(3),('C0','C1'),('center','depth'),lights,('raw','baf'),rids,range(4,20)):
        offset=0 if view=='raw' else 1;sample=f'{rid}/{k:03}';s=shared[(method,light,sample)];ridx=rids.index(rid)
        attention={region:dict(center_error_px=seed+ridx+2+offset,region_attention_mass=.2-.1*offset,region_mass_enrichment=10-5*offset,peak_in_region=not offset,prediction_in_region=not offset) for region in A.REGIONS}
        physical=dict(motion=dict(max_projected_corner_displacement_px=ridx+1.,projected_anchor_displacement_px=ridx+.5),target_identity=dict(segmentation_id=1),
            geometric_sanity=dict(GT_anchor_uv_px=[173.,130.]),nominal_GT_heat_reference=dict(note='fixed synthetic fixture'),visible_geometry=dict(endpoint=dict(area_px=100)))
        illumination=lights.index(light)*.05;e_size=.2+seed*.01+ridx*.001+illumination+offset*.03;e_shape=.1+seed*.01+ridx*.001+illumination+offset*.04
        scale=M.scale_statistics(e_size,math.log(2)+e_shape,0.,0.,2.,2*math.exp(e_shape-e_size)) if branch=='depth' else None
        rows.append(dict(method=method,seed=seed,arm=arm,branch=branch,condition=light,input_view=view,rid=rid,sample_id=sample,pred_uv=[.5+(seed+ridx+2+offset)/346,.5],peak_uv=[.5,.5],
            attention=attention,labels=dict(gt_log_size=0.,gt_z_m=2.,focal_log=0.),scale=scale,features=dict(local_feature=[0.]*64,context_feature=[0.]*64),
            support={k:v for k,v in s.items() if k not in ('method','condition')},**physical))
    hybrids=[]
    for method,seed,arm,light,rid,k in itertools.product(methods,range(3),('C0','C1'),('half','eighth'),rids,range(4,20)):
        for f,a in itertools.product(('normal',light),repeat=2):
            size=.2+(f!='normal')*.05;shape=.1+(f!='normal')*.05+(a!='normal')*.02;z=2*math.exp(shape-size)
            hybrids.append(dict(method=method,seed=seed,arm=arm,paired_light=light,rid=rid,sample_id=f'{rid}/{k:03}',feature_source=f,attention_source=a,
                input_view='raw',log_size=size,log_shape=math.log(2)+shape,z_m=z,gt_z_m=2.,gt_log_size=0.,gt_log_shape=math.log(2),signed_depth_error_m=z-2,diagnostic_only=True,deployable=False,context_source='same_as_features'))
    return rows,support,hybrids


class JointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.rows,cls.support,cls.hybrids=matrix_fixture()

    def test_full_pairing_and_support_do_not_multiply_event_counts_by_seeds_or_models(self):
        paired,support,shared=A.matched_pairs(self.rows,self.support);self.assertEqual(len(paired),576)
        self.assertTrue(all(r['windows']==16 and r['delta']['center_epe_px']==1 for r in paired))
        sg=A.support_groups(support,shared);self.assertEqual(len(sg),48)
        for r in sg.values():self.assertEqual(r['foreground']['raw_events'],150);self.assertEqual(r['foreground']['event_retention_ratio'],.5)
        table=A.seed_ranges(paired,sg);self.assertEqual(len(table),192);self.assertTrue(all(r['delta_center_epe_px_seed_min']==r['delta_center_epe_px_seed_max']==1 for r in table))
        self.assertEqual(sum(int(r['sample_id'].rsplit('/',1)[1])==12 for r in self.rows),576)

    def test_duplicate_and_mismatched_support_geometry_or_views_are_rejected(self):
        for change in ('duplicate','support','geometry','view'):
            rows=self.rows.copy();r=copy.deepcopy(rows[1]);rows[1]=r
            if change=='duplicate':rows[1]=rows[0]
            elif change=='support':r['support']['empty_window']=not r['support']['empty_window']
            elif change=='geometry':r['motion']['max_projected_corner_displacement_px']=999.
            else:r['input_view']='unknown'
            with self.assertRaises(ValueError):A.matched_pairs(rows,self.support)

    def test_signed_bias_scatter_identity_and_all_four_hybrid_cells_retained(self):
        within,drift,hybrid=A.scale_tables(self.rows,self.hybrids)
        self.assertEqual((len(within),len(drift),len(hybrid)),(288,192,384))
        self.assertTrue(all(abs(r['moments']['log_depth']['bias'])<1e-12 for r in drift))
        self.assertTrue(all(abs(r['compensation']['square_identity_residual'])<1e-12 for r in within+drift))
        self.assertEqual(len(M.hybrid_summaries(self.hybrids)['three_seed_means']),32)
        bad=self.hybrids.copy();bad[1]=bad[0]
        with self.assertRaises(ValueError):A.scale_tables(self.rows,bad)
        null=A.null_counts([dict(scale=r['scale']) for r in self.rows]);self.assertEqual(null['/scale'],4608)

    def test_end_to_end_full_schema_readonly_export_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);full=root/'full';full.mkdir();out=root/'derived'
            for name,rows in (('per_window',self.rows),('event_support',self.support),('raw_feature_attention_hybrids',self.hybrids)):A.dump_lines(full/(name+'.jsonl'),rows)
            A.write(full/'raw_hybrid_summaries.json',M.hybrid_summaries(self.hybrids));artifacts=[A.record(p) for p in full.iterdir()]
            metric=Path(M.__file__);self.assertEqual(A.sha(metric),A.METRICS_SHA)
            c=dict(schema='pose_support_attention_scale_frozen_v1',status='complete',mode='full',optimizer_updates=0,real_data_used=False,GT_modified_forward=False,GT_used_only_posthoc=True,
                source_manifest_sha256='fixture_source_manifest',
                raw_parity_passed=True,raw_parity_records=4608,filters_adopted=False,hybrids_deployable=False,attention_RF_interpretation=False,clean384_mixed=False,
                diagnostic_rows=9216,raw_hybrid_rows=6144,support_physical_method_light_rows=384,model_records=[dict(unchanged=True) for _ in range(24)],artifacts=artifacts,source_records=[A.record(metric)])
            A.write(full/'completion.json',c);digest=A.sha(full/'completion.json')
            with patch('sys.argv',['analyze_joint.py','--input-root',str(full),'--completion-sha256',digest,'--out',str(out)]):A.main()
            self.assertEqual(A.read(out/'completion.json')['status'],'complete');self.assertEqual(A.sha(full/'completion.json'),digest)
            self.assertEqual(len(A.jsonl(out/'sentinel_k12_models.jsonl')),576);self.assertEqual(len(A.read(out/'sentinel_k12_shared.json')),24)
            with (out/'per_rid_three_seed_ranges.csv').open() as f:self.assertEqual(len(list(csv.DictReader(f))),192)
            summary=A.read(out/'joint_summary.json');self.assertFalse(summary['p_values_computed']);self.assertFalse(summary['hybrid_selected']);self.assertEqual(summary['counts']['physical_rollouts'],4)


if __name__=='__main__':unittest.main(verbosity=2)
