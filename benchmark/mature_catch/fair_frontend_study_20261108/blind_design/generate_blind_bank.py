"""Metadata only: stratify the pinned Catch It! launch distribution, never run physics.

The original sampler's position/velocity AST is executed for numerical verification.
Reset release time, gravity, PID/noise and quaternion are deliberately NOT fabricated.
"""
from pathlib import Path
import argparse,ast,copy,csv,datetime,hashlib,itertools,json,math,random,shutil,sys,types

HERE=Path(__file__).resolve().parent
UPSTREAM=HERE.parents[1]/'upstream'
OLD_RESULTS=HERE.parents[1]/'advantage_diagnosis_20261108/delivery_local/contact_windows10_unseen25/EVALUATION_PROTOCOL.json'
SEEDS=list(range(1371082001,1371082033))
SHUFFLE_SEED=91574029


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_from_uniforms(u):
    ux,uy,ul,uh,ur,ua,uz=u
    x=ux-.5;y=2.2+.3*uy
    low_factor=False if ul<.5 else True
    height=.7+.3*uh if low_factor else 1.+.6*uh
    r_vel=1.+ur;alpha=math.pi*(ua/6.+5./12.)
    vx=r_vel*math.cos(alpha);vy=-r_vel*math.sin(alpha);vz=.5*uz+2.
    if y>2.25:vy-=.4
    if height<1.:vz+=1.
    return dict(position=[x,y,height],velocity=[vx,vy,vz])


def original_prefix_verifier(source):
    tree=ast.parse(source);fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='random_object_pose')
    prefix=copy.deepcopy(fn)
    stop=next(i for i,n in enumerate(prefix.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='object_static_time' for t in n.targets))
    prefix.body=prefix.body[:stop]
    module=ast.fix_missing_locations(ast.Module(body=[prefix],type_ignores=[]))
    def verify(u,expected):
        draws=iter(u);np_stub=types.SimpleNamespace(random=types.SimpleNamespace(rand=lambda:next(draws)),array=lambda x:list(x))
        space={'np':np_stub,'math':math};exec(compile(module,'pinned_original_launch_prefix','exec'),space)
        obj=types.SimpleNamespace();space['random_object_pose'](obj)
        assert obj.object_pos3d==expected['position'] and obj.object_vel6d[:3]==expected['velocity']
        assert list(draws)==[]
    return fn,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE);p.add_argument('--supersede-plan-sha256',help='Explicit pre-execution protocol revision only; immutable launch parameters must still match');a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    target=out/'BLIND32_BANK.json';previous=None
    if target.exists():
        assert a.supersede_plan_sha256 and sha(target)==a.supersede_plan_sha256,'Refuse to overwrite a registered bank without its exact prior SHA'
        previous=json.loads(target.read_text(encoding='utf-8'));archive=out/'superseded_unexecuted_plan';archive.mkdir(exist_ok=True)
        archive_target=archive/(a.supersede_plan_sha256+'.json');assert not archive_target.exists();shutil.copyfile(target,archive_target)
    env=UPSTREAM/'gym_dcmm/envs/DcmmVecEnv.py';cfg=UPSTREAM/'configs/env/DcmmCfg.py';source=env.read_text(encoding='utf-8');fn,verify=original_prefix_verifier(source)
    strata=list(itertools.product(range(4),range(2),range(2),range(2)))
    random.Random(SHUFFLE_SEED).shuffle(strata)
    cases=[];launches=out/'launch_configs';launches.mkdir(exist_ok=previous is not None)
    conditions=[dict(endpoint_hz=hz,method=method) for hz in [100.,50.,25.,12.5] for method in ['ours','v2e']]
    for i,(seed,stratum) in enumerate(zip(SEEDS,strata)):
        xb,hb,sb,yb=stratum;rng=random.Random(seed+608405117)
        u=[(xb+rng.random())/4.,(yb+rng.random())/2.,(.5+.5*rng.random()) if hb==0 else (.5*rng.random()),rng.random(),(sb+rng.random())/2.,rng.random(),rng.random()]
        launch=launch_from_uniforms(u);verify(u,launch)
        case_id=f'blind_b{i+1:03d}';launch_file=launches/(case_id+'.json')
        if launch_file.exists():assert json.loads(launch_file.read_text(encoding='utf-8'))==launch,'Protocol revision may not change registered launches'
        else:launch_file.write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8',newline='\n')
        # A predeclared balanced16 half-factorial permits a genuine budget choice
        # before any blind outcome; it must never become a result-based early stop.
        half=(xb+hb+sb+yb)%2
        order_base=conditions if i<16 else list(reversed(conditions))
        sequence=order_base[i%8:]+order_base[:i%8]
        cases.append(dict(case_id=case_id,seed=seed,block_index=i+1,stratum=dict(x_quartile=xb,height_branch='low' if hb==0 else 'high',horizontal_magnitude_half=sb,y_half=yb),
            preset_half='A' if half==0 else 'B',sampler_uniforms=u,launch=launch,
            launch_config='launch_configs/'+case_id+'.json',actual_release_time_s=None,actual_world_gravity_m_s2=None,
            primary_condition_order=sequence,
            reference_method_order=['state_reference','init_only'] if i%2==0 else ['init_only','state_reference']))
    assert len(cases)==32 and len({x['seed'] for x in cases})==32
    assert sum(c['preset_half']=='A' for c in cases)==16
    for axis,bins in [('x_quartile',4),('horizontal_magnitude_half',2),('y_half',2)]:
        assert all(sum(c['stratum'][axis]==v for c in cases)==32//bins for v in range(bins))
    assert sum(c['stratum']['height_branch']=='low' for c in cases)==16
    for hz in [100.,50.,25.,12.5]:
        order=lambda c:[x['method'] for x in c['primary_condition_order'] if x['endpoint_hz']==hz]
        assert sum(order(c).index('ours')<order(c).index('v2e') for c in cases)==16
    for condition in conditions:
        assert all(sum(c['primary_condition_order'][position]==condition for c in cases)==4 for position in range(8))
    old=json.loads(OLD_RESULTS.read_text(encoding='utf-8'))
    fixed={k:old[k] for k in ['checkpoint','checkpoint_sha256','core_root','core_files','profile_file','profile_sha256','v2e_checkpoint','v2e_checkpoint_sha256','v2e_source_files']}
    parameters=json.dumps([dict(case_id=c['case_id'],seed=c['seed'],launch=c['launch'],stratum=c['stratum'],preset_half=c['preset_half']) for c in cases],sort_keys=True,separators=(',',':')).encode()
    if previous is not None:assert previous['parameters_sha256']==hashlib.sha256(parameters).hexdigest(),'Blind launch parameters are immutable'
    bank=dict(schema='catchit_original_distribution_stratified_blind32_four_rates_v2',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),planning_date='2026-11-08',
        pre_execution_plan_revision=dict(supersedes_bank_sha256=a.supersede_plan_sha256,launch_parameters_unchanged=previous is not None,reason='Pre-execution metadata-only revision: root selected full32/four rates and balanced forward/reverse condition order; launches unchanged. Superseded plans were not executed.'),
        purpose='Independent broader paired test after separately budget-matched frontend tuning; no outcome-based case filtering or subsequent tuning',
        parameters_sha256=hashlib.sha256(parameters).hexdigest(),generator_sha256=sha(__file__),
        source=dict(upstream_commit='b6af28c67586fc029f3ab43faf5d787e7d3a1609',module='gym_dcmm/envs/DcmmVecEnv.py',module_sha256=sha(env),config_sha256=sha(cfg),
            function='random_object_pose',function_ast_sha256=hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),position_velocity_prefix_verified_exactly=True,
            RNG_note='Distribution formulas and conditional offsets match pinned source. Parameter RNG is a separate preregistered stratified sampler, not a bitwise replay of upstream reset RNG.'),
        sampling=dict(factors='x4 x height2 x horizontal_magnitude2 x y2 =32 equally weighted strata',shuffle_seed=SHUFFLE_SEED,
            x_support_m=[-.5,.5],y_support_m=[2.2,2.5],height_mixture=dict(low=[.7,1.],high=[1.,1.6],weights=[.5,.5]),
            horizontal_magnitude_before_y_offset_m_s=[1.,2.],azimuth_degrees=[75.,105.],vertical_base_m_s=[2.,2.5],
            conditional_offsets='vy-=0.4 when y>2.25; vz+=1 when height<1.0; azimuth is independent of x, no lane or centre-directed correction',
            estimator_inputs_forbidden=['launch_config','object GT pose/velocity','actual release flag/time']),
        reset_randomization=dict(actual_values_not_generated_here=True,release_time_support_s=[.5,.75],world_gravity_z_support_m_s2=[-10.31,-9.31],
            mechanism='Existing runner seeds the normal reset and calls original_random_pose first, then overrides position/velocity only. Original release/quaternion/gravity/PID/noise RNG sequence is retained.',
            audit='Log actual reset release time and world gravity after runtime reset; require exact equality within every paired block. Public world gravity may remain a shared estimator input; release must be scoring-only.',
            coverage='Release-time spread is not guaranteed by offline metadata. Report its observed span without replacing any seed.'),
        fixed_scene=dict(camera_pitch_degrees=7,sphere_radius_m=.04,asset_initialization_seed=1702,contact_profile='combined',lighting='normal',outer_physics_dt_s=.002,policy_hz=25,
            initialization='one actual rendered RGB-D frame; then events and robot proprioception',original_policy_unchanged=True,frozen_event_core_unchanged=True),
        pinned_inputs=fixed,frontend_lock=dict(status='PENDING_BEFORE_FIRST_BLIND_RUN',keys=[method+'@'+str(hz) for hz in ['100','50','25','12.5'] for method in ['ours','v2e']],
            requirement='Each selected frontend config and code hash, tuning-bank hash, candidate budget, and selected result must be frozen before any blind outcome. No blind outcomes may be used for selection.'),
        evaluation=dict(primary_source_rates_hz=[100.,50.,25.,12.5],primary_methods=['ours','v2e'],references=['state_reference','init_only'],full_bank=dict(selected_this_run=True,blocks=32,primary_episodes=256,reference_episodes=64,total_episodes=320),
            budget_preset_A=dict(selected_this_run=False,blocks=16,rule='Unused historical budget preset only; full32 was explicitly selected, so this must not become result-dependent early stopping',primary_episodes=128,reference_episodes=32,total_episodes=160),
            reference_reuse='Run each reference once per block; reuse across source rates because it consumes no subsequent event stream. Do not count reuse as new episodes.',
            independence='32 independent seed blocks, each repeated across eight primary conditions; cadence repetitions are not new independent cases.',
            primary_score='Original2.5s horizon AND not terminated AND actual hand contact occurred AND final40ms controller step contains hand contact; retain all raw flags',
            retain='All initialization failures, non-reachable throws, GT-policy failures and method failures remain in denominator; no GT filtering or fallback. Infrastructure retries preserve same seed/config and original attempt.',
            no_result_based_stop=True),cases=cases)
    target.write_text(json.dumps(bank,indent=2,ensure_ascii=False)+'\n',encoding='utf-8',newline='\n')
    rows=[]
    for c in cases:
        rows.append(dict(case_id=c['case_id'],seed=c['seed'],preset_half=c['preset_half'],**c['stratum'],x_m=c['launch']['position'][0],y_m=c['launch']['position'][1],z_m=c['launch']['position'][2],vx_m_s=c['launch']['velocity'][0],vy_m_s=c['launch']['velocity'][1],vz_m_s=c['launch']['velocity'][2],launch_speed_m_s=math.sqrt(sum(v*v for v in c['launch']['velocity']))))
    with (out/'BLIND32_PARAMETERS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    stats={k:dict(min=min(x[k] for x in rows),max=max(x[k] for x in rows)) for k in ['x_m','y_m','z_m','vx_m_s','vy_m_s','vz_m_s','launch_speed_m_s']}
    (out/'PARAMETER_VALIDATION.json').write_text(json.dumps(dict(status='PASS',blocks=32,original_AST_position_velocity_checks=32,
        equal_weight_strata=32,preset_A_blocks=16,parameter_ranges=stats,release_and_gravity_actuals_generated=False,physics_steps=0,renderer_calls=0,bank_sha256=sha(target),parameters_sha256=bank['parameters_sha256']),indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS','bank':str(target),'sha256':sha(target),'parameters_sha256':bank['parameters_sha256'],'blocks':32,'simulator_calls':0}))


if __name__=='__main__':main()
