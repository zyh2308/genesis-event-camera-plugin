"""CPU-only meaningful fixtures; actual GPU phase0/smoke/study remain unexecuted."""
import copy,hashlib,importlib.util,json,os,sys,tempfile,types,unittest
from pathlib import Path
import numpy as np
import torch
import census_contract as K
import soft_census_v2 as C
import soft_census as V1
import flow_sampling_short as B

torch.set_num_threads(2)

def strata():return {'r'+str(i).zfill(2):K.STRATA[i//6] for i in range(24)}
def calibration():
    rows=[]
    for m in K.METHODS:
        for i in range(8):
            native=2.;ratio=1. if m=='eventgenesis' else 3.
            rows.append(dict(method=m,batch=i,split='train',optimizer_updates=0,GT_in_lambda=False,replay_sha256='replay',
                gradient_space='all_requires_grad_parameters_zero_unused',native_finite_gradient_tensors=70,census_finite_gradient_tensors=70,
                total_requires_grad_parameters=200,parameter_signature_sha256='p'*64,
                native_photo_gradient_l2=native,census_photo_gradient_l2=native*ratio,census_to_native_ratio=ratio,
                model_state_sha256='s'*64,labels_and_augmentation_sha256=str(i)*64))
    return rows
def quality_rows():
    return [dict(method=m,arm=a,validation=dict(window_count=528,
        regions={n:dict(epe_px=1. if a=='native_control' else .99) for n in ('all_physical_valid','common_event_active_valid')},
        strata={s:dict(all_physical_valid=dict(epe_px=1. if a=='native_control' else 1.04)) for s in K.STRATA}))
        for m in K.METHODS for a in K.ARMS]

class Toy(torch.nn.Module):
    def __init__(self):super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Linear(1,1) for _ in range(35)])
    def forward(self,x):return sum(layer(x) for layer in self.layers)
    def bias_parameters(self):return [p for n,p in self.named_parameters() if 'bias' in n]
    def weight_parameters(self):return [p for n,p in self.named_parameters() if 'weight' in n]
def state_sha(model):
    h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):h.update(n.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class ContractTests(unittest.TestCase):
    def test_full_parameter_norm_keeps_unused_as_zero(self):
        from run_census_paired_short import gradient_norm,parameter_signature
        first=torch.nn.Parameter(torch.tensor([3.]));unused=torch.nn.Parameter(torch.tensor([17.]))
        loss=first.square().sum();grads=torch.autograd.grad(loss,(first,unused),allow_unused=True)
        self.assertIsNone(grads[1]);self.assertEqual(gradient_norm(grads,(first,unused),expected_non_none=1),6.)
        model=Toy();model.register_parameter('unused',torch.nn.Parameter(torch.ones(1)))
        rows,sha,named=parameter_signature(model)
        self.assertEqual(len(rows),71);self.assertEqual(len(named),71);self.assertEqual(len(sha),64)
    def test_old_control_schedule_identical_in_every_arm(self):
        from sampling import schedules
        old=schedules(strata());new=K.paired_replay(strata())
        self.assertEqual(new['arms']['native_control'],old['arms']['control'])
        self.assertEqual(new['arms']['census'],old['arms']['control'])
        self.assertEqual(new['augmentation_seed_sequences'],old['augmentation_seed_sequences'])
        self.assertEqual(len(new['arms']['census']),1600)
        self.assertEqual(set(new['arms']),{'native_control','census'})

    def test_parameter_median_not_previous_output_rms(self):
        rows=calibration();r=K.shared_lambda(rows,'replay')
        self.assertEqual(r['shared_census_weight'],.5)
        rms=(sum(x['native_photo_gradient_l2']**2 for x in rows)/sum(x['census_photo_gradient_l2']**2 for x in rows))**.5
        self.assertNotAlmostEqual(r['shared_census_weight'],rms)
        for field,value in [('GT_in_lambda',True),('split','validation'),('native_finite_gradient_tensors',69),
                            ('census_photo_gradient_l2',float('nan')),('labels_and_augmentation_sha256','bad'),('model_state_sha256','bad')]:
            with self.subTest(field=field):
                v=copy.deepcopy(rows);v[0][field]=value
                with self.assertRaises(ValueError):K.shared_lambda(v,'replay')
        with self.assertRaises(ValueError):K.shared_lambda(rows[:-1],'replay')

    def test_original_gate_keeps_both_generators_and_5percent_strata(self):
        rows=quality_rows();self.assertTrue(K.quality(rows)['passed'])
        v=copy.deepcopy(rows);v[-1]['validation']['strata'][K.STRATA[-1]]['all_physical_valid']['epe_px']=1.051
        self.assertFalse(K.quality(v)['passed'])
        v=copy.deepcopy(rows);v[-1]['validation']['regions']['common_event_active_valid']['epe_px']=1.
        self.assertFalse(K.quality(v)['passed'])
        with self.assertRaises(ValueError):K.quality(rows[:-1])

    def test_raw_domain_strict_internal_interpolation_no_clamp(self):
        ones=torch.ones(2,1,64,64);torch.manual_seed(0);flow=(torch.rand(2,2,64,64)*2-1).requires_grad_(True)
        with self.assertRaises(ValueError):V1.census_loss_sum(ones,ones,flow)
        untouched=ones.clone();value,_=C.census_loss_sum(ones,ones,flow)
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.equal(ones,untouched))
        self.assertTrue(torch.isfinite(torch.autograd.grad(value,flow)[0]).all())
        upper=torch.nextafter(ones,torch.full_like(ones,float('inf')));saved=upper.clone()
        with self.assertRaises(ValueError):C.census_loss_sum(upper,upper,torch.zeros_like(flow))
        C.census_loss_sum(upper,upper,torch.zeros_like(flow),inputs_interpolated=True)
        self.assertTrue(torch.equal(upper,saved))
        raw={'gray_prev':upper.numpy(),'gray_next':ones.numpy()}
        with self.assertRaises(ValueError):K.raw_endpoint_domain(raw)
        K.raw_endpoint_domain({'gray_prev':ones.numpy(),'gray_next':ones.numpy()})
        bad=torch.full_like(ones,1.001)
        with self.assertRaises(ValueError):C.census_loss_sum(bad,bad,torch.zeros_like(flow),inputs_interpolated=True)

    def test_actual_original_augmentation_white_endpoints_v2(self):
        here=Path(__file__).resolve().parent;contract=here/'contract.json'
        candidates=[]
        if contract.exists():candidates.append(Path(K.read(contract)['frozen_inputs']['training_source']))
        candidates.append(here.parents[2]/'outputs/sim2real_retraining_week_20261103/server_export/train_steflow_stage9.py')
        if os.environ.get('CENSUS_NATIVE_TRAINER'):candidates.insert(0,Path(os.environ['CENSUS_NATIVE_TRAINER']))
        path=next((p for p in candidates if p.exists()),None)
        self.assertIsNotNone(path,'Actual pinned trainer fixture is required')
        self.assertEqual(K.sha(path),'4da4438141afd76a929b2f90f6dcd521424692bc4d84fcf5fcbd619338fd8d8d')
        fallback=Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105/benchmark/sim2real')
        sys.path.insert(0,str(fallback/'scripts'));sys.path.insert(0,str(fallback/'compat'));sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location('census_actual_native_augmentation_fixture',path)
        trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)
        sample=dict(events=np.zeros((1,2,260,346,5),np.float32),gray_prev=np.ones((1,260,346),np.float32),
            gray_next=np.ones((1,260,346),np.float32),flow_gt=np.zeros((1,2,260,346),np.float32),flow_valid=np.ones((1,260,346),np.uint8))
        K.raw_endpoint_domain(sample)
        for micro in range(3):
            _,gray,_,_=trainer.apply_joint_transform(sample,np.random.default_rng(np.random.SeedSequence([0,45,0,micro])),torch.device('cpu'))
            before=gray.clone();outputs=[torch.zeros(1,2,n,n,requires_grad=True) for n in (256,128,64,32)]
            loss,_=C.multiscale_census_sum(outputs,gray[:,0:1],gray[:,1:2],inputs_interpolated=True)
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.equal(before,gray))
            self.assertTrue(all(torch.isfinite(g).all() for g in torch.autograd.grad(loss,outputs)))

    def test_reused_save_helper_independent70_adam_checkpoint(self):
        old=torch.nn.Module.cuda;torch.nn.Module.cuda=lambda self,*a,**k:self
        try:
            model=Toy();opt=torch.optim.Adam([{'params':model.bias_parameters(),'weight_decay':0.},
                {'params':model.weight_parameters(),'weight_decay':.0004}],lr=.0004,betas=(.9,.999))
            model(torch.ones(2,1)).square().mean().backward();opt.step()
            self.assertEqual(len(opt.state),70)
            for state in opt.state.values():state['step'].fill_(8911)
            scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,milestones=[5,10,20],gamma=.7)
            trainer=types.SimpleNamespace(state_sha256=state_sha);models=types.SimpleNamespace(steflow=Toy)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'checkpoint.pt';before=torch.get_rng_state().clone()
                rec=B.save_checkpoint(path,trainer,models,model,opt,scheduler,dict(path='fixture_only',sha256='f'*64),1)
                self.assertTrue(rec['model_rebuild_passed']);self.assertTrue(rec['optimizer_state_equal'])
                self.assertTrue(torch.equal(before,torch.get_rng_state()))
                loaded=torch.load(path,map_location='cpu',weights_only=False)
                self.assertEqual(loaded['optimizer_update'],8911);self.assertEqual(loaded['scheduler'],scheduler.state_dict())
        finally:torch.nn.Module.cuda=old

    def test_unmodified_helper_pins(self):
        self.assertEqual(K.sha(Path(B.__file__)),'5340990638eecf0865d2a322e45cb903f990a9d4077cf047f7cb40b5de3f4319')
        self.assertEqual(K.sha(Path(__file__).parent/'soft_census.py'),'7ceae6b8774322a12fac69ee8c2defff2368233eae024e0106c1e4e276f471e7')


if __name__=='__main__':unittest.main(verbosity=2)
