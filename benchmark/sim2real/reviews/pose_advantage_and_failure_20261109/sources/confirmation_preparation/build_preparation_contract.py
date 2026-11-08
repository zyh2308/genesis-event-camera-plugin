"""Bind the actual scientific registration and existing frozen files; CPU only."""
import argparse,json
from pathlib import Path
import prepare_confirmation as P

R=Path('/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1')
RUNS=R.parent
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--relight-source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();N=Path(P.ROOT).parent;source=R/'sources';pds=RUNS/'20261108-pose-factor-diagnostic-v1/sources'
    fixed=['pose_physical_worker.py','stage9_sensor_profile.json','v2e_source_pin.json','replay_new_events.py','replay_v2e_parallel.py','pose_bank_protocol.json','training_recipe.json']
    paths=[source/n for n in fixed]+sorted(p for p in (source/'benchmark').rglob('*.py') if p.is_file())
    paths=list(dict.fromkeys(paths));P.need(len(paths)>=len(fixed)+1,'Missing benchmark code inventory')
    c=dict(schema='pose_frozen_c1_advantage_confirmation32_v1',frozen_core=P.CORE,physical_seed=P.BASE,rollouts=32,root=P.ROOT,campaign=str(R),
        methods=list(P.METHODS),seeds=[0,1,2],conditions={'normal':1.,'half':.5,'eighth':.125},primary_arm='C1_RAW',optimizer_updates=0,real_data_used=False,
        primary_conditions=['normal','half'],bootstrap_draws=10000,bootstrap_seed=P.BOOTSTRAP_SEED,per_condition_ci_level=.975,all_three_seed_deltas_positive=True,
        scientific_registration=P.record(N/'advantage_registration.json'),seed_inventory=P.record(N/'seed_inventory.json'),core_integrity=P.record(N/'frozen_core_integrity.json'),
        fresh_source=P.record(R/'recovery_20261105_mainline/pose_fresh_confirm_20261106_v3.py'),relight_source=P.record(a.relight_source),
        export_source=P.record(pds/'export_pose_factors.py'),base_contract=P.record(pds/'contract.json'),
        baseline_scores=P.record(RUNS/'20261109-pose-predicted-peak-readout-v1/full/scores.json'),source_records=[P.record(p) for p in paths],
        inventory_scope='Original registered protocols and corresponding226 H5 attributes; not an exhaustive filesystem search',
        no_actual_gpu_calls_in_builder=True)
    P.validate_registration(c);P.check_science_and_core(c)
    P.need(c['export_source']['sha256']=='fdd5d96dfd6191fb73923cfa06699d5995ab0750a8e1579223ef8a1b3b5c61cd','Frozen export source differs')
    P.need(c['base_contract']['sha256']=='d2368c9f246a61efc4382238741f1e7a62cf891aa8d6d653136832ed64ea3b62','Frozen base contract differs')
    P.need(c['baseline_scores']['sha256']=='74f6ba1d0ccb37594102f508868229485eade02f87e712c8aeb3a9147b16d253','Frozen RAW C1 scores differ')
    P.write(a.out,c);print(json.dumps(dict(path=str(a.out),sha256=P.sha(a.out),source_files=len(paths),optimizer_updates=0,real_data_used=False)))

if __name__=='__main__':main()
