"""Meaningful CPU admission and aggregation tests; no real input/model/network."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
import shutil

import evaluate_selected_hqf as b


def contracts():
    plan = dict(schema='recon_selected_full_plan_v1', methods=list(b.METHODS), seeds=list(b.SEEDS),
        epochs=200, training_budget_updates=12000, final_activation='sigmoid',
        loss=dict(lpips_vgg=1., temporal=2., auxiliary_l1=0.), frozen_core=b.CORE,
        recipe_sha256=b.RECIPE_SHA, trainer_sha256=b.TRAINER_SHA, real_sequences=list(b.SEQUENCES),
        real_result_scope=b.SCOPE, real_scoring_enabled=False,
        no_real_feedback_to_training_or_selection=True, original_hqf14_results_preserved=True)
    matrix = dict(schema='recon_selected_frozen_checkpoints_v1', status='frozen_all15_before_real_scoring',
        methods=list(b.METHODS), seeds=list(b.SEEDS), real_sequences=list(b.SEQUENCES), result_scope=b.SCOPE,
        recipe={'sha256': b.RECIPE_SHA}, runs=[dict(run_id=m+'_seed'+str(s), method=m, seed=s,
            completed_updates=12000, initial_state_sha256=str(s)*64) for s in b.SEEDS for m in b.METHODS])
    completion = dict(status='full15_training_complete_selected_eval_pending', training_runs=15,
        evaluated_sequence_runs=0, real_scoring_enabled=False, real_result_scope=b.SCOPE,
        full14_historical_results_unchanged=True)
    return plan, matrix, completion


def fake_score():
    arms = {}
    for method in b.METHODS:
        for seed in b.SEEDS:
            sequences = {}
            for seq, value, n in zip(b.SEQUENCES, (1., 4., 10.), (1, 2, 100)):
                value += seed
                sequences[seq] = dict(scored_frame_count=n, num_frames_including_anchor=n+1,
                    initial_anchor_excluded=True, metrics=dict(lpips_alex_mean=value,
                    ssim_mean=value / 20, mse_mean=value / 50,
                    psnr=dict(mean_db=value * 2, mean_is_positive_infinity=False)))
            arms[method+'_seed'+str(seed)] = dict(per_sequence=sequences, sequence_count=3)
    return {'per_arm': arms}


class AdmissionTests(unittest.TestCase):
    def test_all15_required_and_scope_frozen(self):
        plan, matrix, complete = contracts()
        self.assertEqual(len(b.matrix_contract(plan, matrix, complete)), 15)
        mutations = [
            ('missing method/seed', lambda p,m,c: m['runs'].pop()),
            ('duplicate replacing missing seed', lambda p,m,c: m['runs'].__setitem__(-1, copy.deepcopy(m['runs'][0]))),
            ('full budget incomplete', lambda p,m,c: m['runs'][0].__setitem__('completed_updates',11999)),
            ('unpaired initial state', lambda p,m,c: m['runs'][0].__setitem__('initial_state_sha256','x'*64)),
            ('not frozen before real', lambda p,m,c: m.__setitem__('status','running')),
            ('real scoring was enabled', lambda p,m,c: c.__setitem__('real_scoring_enabled',True)),
            ('selected subset changed', lambda p,m,c: p.__setitem__('real_sequences',['boxes','desk','still_life'])),
            ('loss changed for favored method', lambda p,m,c: p['loss'].__setitem__('auxiliary_l1',.25)),
            ('recipe changed', lambda p,m,c: m['recipe'].__setitem__('sha256','a'*64)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                p,m,c=copy.deepcopy((plan,matrix,complete));mutate(p,m,c)
                with self.assertRaises(ValueError):b.matrix_contract(p,m,c)

    def test_original_helper_is_exactly_pinned(self):
        helper = b.load_helper()
        self.assertEqual(helper.CORE,b.CORE)
        self.assertEqual(b.sha(Path(helper.__file__)),b.HELPER_SHA)

    def test_selected_rows_remain_byte_equivalent_and_hash_guarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);hqf=root/'hqf';rows=[]
            for seq in [*b.SEQUENCES,*('unused'+str(i) for i in range(11))]:
                mem=hqf/seq;files={}
                for name in b.load_helper().FILES:
                    if seq in b.SEQUENCES:
                        mem.mkdir(parents=True,exist_ok=True);path=mem/name;path.write_bytes((seq+name).encode())
                        r=b.record(path);files[name]={k:r[k] for k in ('size_bytes','sha256')}
                    else:files[name]=dict(size_bytes=1,sha256='0'*64)
                rows.append(dict(sequence=seq,status='ready_unscored',evreal_memmap=dict(path=str(mem),
                    exists=True,missing_files=[],files=files)))
            parent=dict(status='ready_unscored',expected_sequence_count=14,ready_unscored_sequence_count=14,
                real_scoring_enabled=False,real_gt_metrics_computed=False,sequences=rows)
            original=root/'original.json';b.dump_new(original,parent);original_bytes=original.read_bytes()
            parent_rec=b.record(original);selected=b.selected_manifest(parent,parent_rec)
            path=root/'selected.json';b.dump_new(path,selected)
            protocol=dict(original_data_manifest=parent_rec,data_manifest=b.record(path),
                original_sequences=[r['sequence'] for r in rows],hqf_root=str(hqf))
            actual,_=b.verify_selected_manifest(protocol)
            self.assertEqual(actual['sequences'],rows[:3]);self.assertEqual(original.read_bytes(),original_bytes)
            # Forging a fresh digest does not make a modified row an exact subset.
            selected['sequences'][0]['evreal_memmap']['files']['metadata.json']['size_bytes']+=1
            path.write_text(json.dumps(selected));protocol['data_manifest']=b.record(path)
            with self.assertRaisesRegex(ValueError,'unchanged HQF14 rows'):b.verify_selected_manifest(protocol)
            path.write_text(json.dumps(b.selected_manifest(parent,parent_rec)));protocol['data_manifest']=b.record(path)
            (hqf/b.SEQUENCES[0]/'metadata.json').write_bytes(b'changed actual bytes')
            with self.assertRaisesRegex(ValueError,'Bound file changed'):b.verify_selected_manifest(protocol)

    def test_checkpoint_source_and_selected_boundary(self):
        plan,matrix,_=contracts();row=matrix['runs'][0]
        recipe={'reconstruction':{'model_args':{'final_activation':'sigmoid'}},'reconstruction_repair':{'version':'same'}}
        selection={'optimizer_update':4800,'epoch':80}
        cp=dict(schema='e2vid_stage4_resume_v1',generator=row['method'],seed=0,optimizer_update=4800,
            next_epoch=80,next_batch_in_epoch=0,optimizer_step_counts=[4800]*30,
            config_sha256=b.RECIPE_SHA,trainer_script_sha256=b.TRAINER_SHA,data_manifest_sha256='d'*64,
            model_args=recipe['reconstruction']['model_args'],real_data_used=False,
            initial_state_sha256='0'*64,reconstruction_repair=recipe['reconstruction_repair'],
            train_rollouts=list(range(24)),validation_rollouts=list(range(24,32)))
        b.validate_checkpoint_metadata(cp,row,recipe,plan,selection,'0'*64,'d'*64)
        for field,value in [('real_data_used',True),('config_sha256','bad'),('trainer_script_sha256','bad'),
                             ('optimizer_update',4801),('initial_state_sha256','bad'),('validation_rollouts',list(range(8)))]:
            with self.subTest(field=field):
                changed=copy.deepcopy(cp);changed[field]=value
                with self.assertRaises(ValueError):b.validate_checkpoint_metadata(changed,row,recipe,plan,selection,'0'*64,'d'*64)

    def test_equal_sequence_then_seed_sample_sd(self):
        result=b.aggregate(fake_score());row=result['method_table'][0]
        self.assertAlmostEqual(row['lpips_alex_mean'],6.)
        self.assertAlmostEqual(row['lpips_alex_sample_sd'],1.)
        self.assertEqual(result['per_seed'][0]['lpips_alex'],5.)
        pooled=(1+4*2+10*100)/103
        self.assertNotAlmostEqual(result['per_seed'][0]['lpips_alex'],pooled)
        self.assertEqual(len(result['per_sequence']),45);self.assertEqual(len(result['per_seed']),15)
        partial=fake_score();partial['per_arm'].pop('v2e_seed2')
        with self.assertRaises(ValueError):b.aggregate(partial)
        inconsistent=fake_score();inconsistent['per_arm']['v2e_seed2']['per_sequence']['boxes']['scored_frame_count']=2
        with self.assertRaises(ValueError):b.aggregate(inconsistent)
        invalid=fake_score();invalid['per_arm']['eventgenesis_seed0']['per_sequence']['boxes']['metrics']['mse_mean']=float('nan')
        with self.assertRaises(ValueError):b.aggregate(invalid)

    def test_positive_infinite_psnr_not_silently_nan(self):
        score=fake_score();score['per_arm']['eventgenesis_seed1']['per_sequence']['boxes']['metrics']['psnr']=dict(mean_db=None,mean_is_positive_infinity=True)
        r=b.aggregate(score)['method_table'][0]
        self.assertIsNone(r['psnr_db_mean']);self.assertIsNone(r['psnr_db_sample_sd'])
        self.assertTrue(r['psnr_mean_is_positive_infinity'])
        json.dumps(b.aggregate(score),allow_nan=False)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'requires CPU PyTorch for tiny fake-checkpoint fixture')
    def test_complete_file_lineage_and_partial_log_rejection(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'training_root';src=root/'sources';src.mkdir(parents=True)
            actual=Path(os.environ.get('RECON_SELECTED_TRAINING_SOURCES',str(b.HERE.parent/'recon_selected_full_20261108')))
            for name in ('training_recipe.json','train_e2vid_repaired.py'):
                shutil.copyfile(actual/name,src/name)
            plan,matrix,done=contracts();plan.update(root=str(root),campaign=str(root/'campaign'))
            b.dump_new(src/'registered_plan.json',plan)
            b.dump_new(src/'source_manifest.json',{n:b.sha(src/n) for n in ('registered_plan.json','training_recipe.json','train_e2vid_repaired.py')})
            sm=b.sha(src/'source_manifest.json');matrix.update(source_manifest_sha256=sm,recipe=b.record(src/'training_recipe.json'))
            initial=root/'initial.json';b.dump_new(initial,{'tasks':{'reconstruction':{'hashes':{str(s):str(s)*64 for s in b.SEEDS}}}})
            b.dump_new(root/'cpu_admission.json',dict(passed=True,source_manifest_sha256=sm,initial_state_reference=b.record(initial)))
            b.dump_new(root/'ALL15_SYNTHETIC_GATE.json',dict(passed=True,source_manifest_sha256=sm))
            recipe=b.read(src/'training_recipe.json');tensor=torch.tensor([1.],dtype=torch.float32)
            tensor_sha=b.hashlib.sha256(b'weight'+tensor.numpy().tobytes()).hexdigest()
            for row in matrix['runs']:
                run=root/'training'/('seed'+str(row['seed']))/row['method'];run.mkdir(parents=True)
                adapter=root/'campaign/adapters'/row['method']/'reconstruction/adapter_manifest.json'
                if not adapter.exists():adapter.parent.mkdir(parents=True);b.dump_new(adapter,{'synthetic':True})
                a_sha=b.sha(adapter)
                cp=dict(schema='e2vid_stage4_resume_v1',generator=row['method'],seed=row['seed'],optimizer_update=300,
                    next_epoch=5,next_batch_in_epoch=0,optimizer_step_counts=[300]*30,model={'weight':tensor},model_sha256=tensor_sha,
                    config_sha256=b.RECIPE_SHA,trainer_script_sha256=b.TRAINER_SHA,data_manifest_sha256=a_sha,
                    model_args=recipe['reconstruction']['model_args'],real_data_used=False,initial_state_sha256=row['initial_state_sha256'],
                    reconstruction_repair=recipe['reconstruction_repair'],train_rollouts=list(range(24)),validation_rollouts=list(range(24,32)))
                path=run/'synthetic_best.pt';torch.save(cp,path);row['checkpoint']=b.record(path)
                selection=dict(selection_source='synthetic validation only',checkpoint=str(path),sha256=b.sha(path),size_bytes=path.stat().st_size,
                    optimizer_update=300,epoch=5,synthetic_validation_lpips_vgg=1.)
                b.dump_new(run/'synthetic_best.json',selection);row['selection_record']=b.record(run/'synthetic_best.json')
                summary=dict(status='complete',optimizer_update=12000,optimizer_budget=12000,completed_epochs=200,
                    config_sha256=b.RECIPE_SHA,trainer_script_sha256=b.TRAINER_SHA,data_manifest_sha256=a_sha,
                    initial_model_sha256=row['initial_state_sha256'],real_scoring_enabled=False)
                (run/'last.pt').write_bytes(b'fake final full-budget fixture checkpoint')
                summary.update(final_checkpoint=str(run/'last.pt'),final_checkpoint_sha256=b.sha(run/'last.pt'))
                b.dump_new(run/'synthetic_output_stability_guard.json',dict(passed=True,optimizer_update=12000,
                    real_data_used=False,rollouts=[dict(minimum=0.,maximum=1.)]*8,contrast_ratio_mean=1.,degenerate_fraction_mean=0.))
                b.dump_new(run/'boundary_preflight.json',dict(passed=True,oracle={'status':'pass'}))
                b.dump_new(run/'run_summary.json',summary)
                b.dump_new(run/'completion.json',dict(status='complete',method=row['method'],seed=row['seed'],optimizer_update=12000,
                    real_data_used=False,real_scoring_enabled=False,source_manifest_sha256=sm,selected_checkpoint=row['checkpoint'],
                    selection_record=row['selection_record'],summary=summary))
                with (run/'training_metrics.jsonl').open('w') as f:
                    for u in range(1,12001):
                        f.write(json.dumps(dict(event='optimizer_update',optimizer_update=u))+'\n')
                        if u%300==0:
                            f.write(json.dumps(dict(event='synthetic_validation',optimizer_update=u,
                                lpips_vgg_equal_rollout_mean=1.+u/300-1,rollouts=[{}]*8))+'\n')
            b.dump_new(root/'frozen_checkpoint_matrix.json',matrix);b.dump_new(root/'completion.json',done)
            lineage,_,_,_=b.collect_training(root)
            self.assertEqual(len(lineage['runs']),15)
            log=root/'training/seed2/rife_esim/training_metrics.jsonl'
            with log.open('a') as f:f.write(json.dumps(dict(event='optimizer_update',optimizer_update=12000))+'\n')
            with self.assertRaisesRegex(ValueError,'complete update inventory'):b.collect_training(root,inspect_models=False)


if __name__=='__main__':unittest.main(verbosity=2)
