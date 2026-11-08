"""Pure CPU rejection checks; no real arrays, model construction or training."""
import copy,unittest
from selected_common import *
from recon_selected_gate import decide

class Contracts(unittest.TestCase):
    def row(self,method,seed):
        metric=dict(clipped_lpips=.3,raw_lpips=.3,zero_lpips=.7,gray_lpips=.6,mse=.01,zero_mse=.2,
            clipped_std=.05,target_std=.15,correlation=.5,raw_min=.1,raw_max=.9,negative_fraction=0.)
        return dict(method=method,seed=seed,arm='sigmoid',fixed_train_batch=False,updates=300,real_data_used=False,
            before={'equal_rollout_mean':{'clipped_lpips':.5}},after=dict(equal_rollout_mean=metric,sequence_length=200,
            rollouts=[{'rollout':str(i)} for i in range(8)],degenerate_frame_fraction=0.),recovery_model_hash_match=True,
            optimizer_counters=[300],initial_state_sha256=str(seed)*64,sample_label_hashes_sha256=str(seed)*64)
    def matrix(self):return [self.row(m,s) for s in SEEDS for m in METHODS]
    def test_full15_gate(self):self.assertTrue(decide(self.matrix())['passed'])
    def test_missing_seed_rejected(self):
        with self.assertRaises(ValueError):decide(self.matrix()[:-1])
    def test_degenerate_rejected(self):
        x=self.matrix();x[-1]['after']['degenerate_frame_fraction']=.1;self.assertFalse(decide(x)['passed'])
    def test_shared_label_mismatch(self):
        x=self.matrix();x[-1]['sample_label_hashes_sha256']='x'*64;self.assertFalse(decide(x)['passed'])
    def test_real_training_rejected(self):
        x=self.matrix();x[-1]['real_data_used']=True
        with self.assertRaises(ValueError):decide(x)
    def test_cursor_exact(self):
        x=dict(schema='e2vid_stage4_resume_v1',optimizer_update=5264,next_epoch=87,next_batch_in_epoch=44,
            optimizer_step_counts=[5264]*30,rng_state=dict(python=0,numpy=0,torch_cpu=0,torch_cuda=[0]))
        cursor_check(x,5264);x['next_batch_in_epoch']=45
        with self.assertRaises(ValueError):cursor_check(x,5264)
    def test_adam_counter_rejected(self):
        x=dict(schema='e2vid_stage4_resume_v1',optimizer_update=5306,next_epoch=88,next_batch_in_epoch=26,
            optimizer_step_counts=[5306]*29+[5305],rng_state=dict(python=0,numpy=0,torch_cpu=0,torch_cuda=[0]))
        with self.assertRaises(ValueError):cursor_check(x,5306)
    def test_frozen_plan(self):
        p=read(Path(__file__).parent/'registered_plan.json');plan_check(p)
        p['loss']['auxiliary_l1']=.25
        with self.assertRaises(ValueError):plan_check(p)
    def test_real_scoring_closed(self):
        p=read(Path(__file__).parent/'registered_plan.json');p['real_scoring_enabled']=True
        with self.assertRaises(ValueError):plan_check(p)

if __name__=='__main__':unittest.main()
