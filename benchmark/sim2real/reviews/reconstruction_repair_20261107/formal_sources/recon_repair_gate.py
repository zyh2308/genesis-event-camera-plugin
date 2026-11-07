"""Predeclared synthetic-only adoption checks; shared by study and formal launcher."""
import math
METHODS=('eventgenesis','v2e')
def adoption_checks(results,arm='sigmoid'):
    expected={(method,seed) for method in METHODS for seed in [0,1,2]}
    actual=[(r['method'],r['seed']) for r in results]
    if len(actual)!=6 or set(actual)!=expected:raise ValueError('Complete paired three-seed matrix required')
    checks=[]
    def check(name,passed,**detail):checks.append(dict(name=name,passed=bool(passed),**detail))
    for r in results:
        tag=f"{r['method']}_seed{r['seed']}";a=r['after'];m=a['equal_rollout_mean'];before=r['before']['equal_rollout_mean']
        if r['arm']!=arm or r['fixed_train_batch'] or r['updates']<120 or r['real_data_used']:raise ValueError(tag+' invalid training scope')
        if a['sequence_length']!=200 or len(a['rollouts'])!=8 or len({x['rollout'] for x in a['rollouts']})!=8:raise ValueError(tag+' incomplete full validation')
        if any(not math.isfinite(v) for v in m.values()):raise ValueError(tag+' non-finite metrics')
        check(tag+'/lpips_improves_initial',m['clipped_lpips']<before['clipped_lpips'],before=before['clipped_lpips'],after=m['clipped_lpips'])
        check(tag+'/lpips_beats_black',m['clipped_lpips']<m['zero_lpips'],after=m['clipped_lpips'],baseline=m['zero_lpips'])
        check(tag+'/lpips_beats_gray',m['clipped_lpips']<m['gray_lpips'],after=m['clipped_lpips'],baseline=m['gray_lpips'])
        check(tag+'/mse_beats_black',m['mse']<m['zero_mse'],after=m['mse'],baseline=m['zero_mse'])
        ratio=m['clipped_std']/max(m['target_std'],1e-12)
        check(tag+'/spatial_contrast',ratio>.1,ratio=ratio,required_greater_than=.1)
        check(tag+'/spatial_correlation',m['correlation']>.1,correlation=m['correlation'],required_greater_than=.1)
        check(tag+'/degenerate_fraction',a['degenerate_frame_fraction']<.05,fraction=a['degenerate_frame_fraction'],required_less_than=.05)
        check(tag+'/checkpoint_restore',r['recovery_model_hash_match'] and r['optimizer_counters']==[r['updates']])
        if arm=='sigmoid':
            check(tag+'/output_range',m['raw_min']>=0 and m['raw_max']<=1 and m['negative_fraction']==0)
            check(tag+'/raw_clip_metric_agreement',abs(m['clipped_lpips']-m['raw_lpips'])<1e-6)
    for seed in [0,1,2]:
        rows=[r for r in results if r['seed']==seed]
        check(f'seed{seed}/shared_initialization',len({r['initial_state_sha256'] for r in rows})==1)
        check(f'seed{seed}/shared_labels_and_augmentation',len({r['sample_label_hashes_sha256'] for r in rows})==1)
    return checks

def decide(results,arm='sigmoid'):
    checks=adoption_checks(results,arm)
    return dict(passed=all(c['passed'] for c in checks),arm=arm,checks=checks,
        real_data_used=False,checkpoint_selection='full synthetic validation only; no real scores',
        policy='all paired methods and all seeds required; no exclusions or threshold relaxation')
