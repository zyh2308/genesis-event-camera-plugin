"""Pinned, unmodified V2E sensor versus its own independent continuous law.

Common optical fixtures are log-linear radiance in I units; V2E receives
float32 DN=255*I, with no gamma, integer quantization or clipping.  Only fixtures
whose complete physical trajectory stays strictly in 20<DN<255 are qualified.
The native bandwidth is (DN+20)/275, deliberately NOT replaced by Genesis's
clip(I,0,1).  Numerical error against each model's own law is not realism.

No model source, mapped values, event times, precision or RNG is monkey-patched.
The public generate_events NumPy-to-device interface and FP32 output survive.
Noise is disabled only for this explicitly named diagnostic.  0.5ms refractory
is retained; every monotonic fixture has at most one threshold crossing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
import torch

INTERVAL = .010
DURATION = .030
FC = 300.
TAU = 1/(2*math.pi*FC)
C = .2
REFRACTORY = .0005
SHAPE = (8,8)
DEFAULT_NS = [32,63,126,189,252,377,754,1508]
PINS = {
    'v2ecore/emulator.py': '305dc78517d9d82ed1b6a2199546632a41a6645843a72c837c106bcca0510375',
    'v2ecore/emulator_utils.py': '064cb7dcbf0cc300aa6cd3fb53c572fe3c119c6c50d87133539cb276bb628de9',
}
BUDGETS = dict(state_rmse_over_C=.025,state_p99_over_C=.05,
    event_matching_tolerance_s=50e-6,event_f1=.99,matched_time_p95_s=20e-6,
    absolute_count_bias_fraction=.01)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def levels(light,sign,delta):
    left=float(np.float32(light))
    return left,float(np.float32(left*math.exp(sign*delta)))


def make_cases(illumination,kind):
    cases=[]
    for light in illumination:
        for sign in (1,-1):
            for delta in (.24,.30,.36):
                widths=(.002,.005,.010) if kind=='smooth' else (None,)*3
                phases=(.001137,)*3 if kind=='smooth' else (.001137,.002911,.004073)
                for width,phase in zip(widths,phases):
                    a,b=levels(light,sign,delta)
                    name=(f'ramp_I{light:g}_p{sign:+d}_d{delta:g}_r{width:.3f}' if kind=='smooth'
                          else f'step_I{light:g}_p{sign:+d}_d{delta:g}_t{phase:.6f}')
                    cases.append(dict(name=name,group='smooth_single_crossing_ramp' if kind=='smooth' else 'single_crossing_step',
                        kind=kind,protected=True,initial=a,final=b,illumination=light,sign=sign,
                        nominal_delta_log=delta,effective_delta_log=math.log(b)-math.log(a),phase_s=phase,
                        ramp_duration_s=width or 0.,changes=[[phase,b]] if kind=='step' else []))
        a=float(np.float32(light))
        cases.append(dict(name=f'{kind}_constant_I{light:g}',group='zero_event_control',kind=kind,
            protected=True,initial=a,final=a,illumination=light,sign=0,nominal_delta_log=0.,
            effective_delta_log=0.,phase_s=0.,ramp_duration_s=0.,changes=[]))
    if len(cases)>np.prod(SHAPE):
        raise ValueError('At most 64 cases in the registered 8x8 layout')
    for case in cases:
        lo,hi=255*min(case['initial'],case['final']),255*max(case['initial'],case['final'])
        if not (20<lo<=hi<255):
            raise ValueError('Unqualified DN domain (lin-log kink/HDR): '+case['name'])
        case['continuous_DN_min']=lo;case['continuous_DN_max']=hi
    return cases


def physical_intensity(case,stamps):
    stamps=np.asarray(stamps,dtype=np.float64)
    if not case['sign']:
        return np.full(stamps.shape,case['initial'],dtype=np.float64)
    if case['kind']=='step':
        return np.where(stamps>=case['phase_s'],case['final'],case['initial'])
    w=np.clip((stamps-case['phase_s'])/case['ramp_duration_s'],0,1)
    values=np.exp(math.log(case['initial'])+w*case['effective_delta_log'])
    return np.where(w==0,case['initial'],np.where(w==1,case['final'],values))


def sample_intensity(case,stamps):
    # Match the earlier Ours fixtures' float32 radiance samples exactly.
    return physical_intensity(case,stamps).astype(np.float32)


def native_value(intensity):
    # Independent continuous, unrounded native mapping; domain gate above means
    # the logarithmic branch applies.  Production's 1e-8 rounding / FP32 target
    # remain in the candidate and are counted as implementation approximation.
    dn=255*np.asarray(intensity,dtype=np.float64)
    return np.where(dn<=20,dn*(math.log(20)/20),np.log(dn))


def native_rate(intensity):
    return (255*np.asarray(intensity,dtype=np.float64)+20)/(275*TAU)


class Reference:
    def __init__(self,case,queries):
        self.case=case
        self.initial=float(native_value(case['initial']))
        self.target=self.initial+case['sign']*C
        if case['kind']=='step' or not case['sign']:
            self._step(queries)
        else:
            self._smooth(queries)

    def _step(self,queries):
        c=self.case;phase=c['phase_s']
        final=float(native_value(c['final']));rate=float(native_rate(c['final']))
        def state(stamps):
            stamps=np.asarray(stamps,dtype=np.float64)
            elapsed=np.maximum(stamps-phase,0)
            return final+(self.initial-final)*np.exp(-rate*elapsed)
        self.state=state
        if c['sign']:
            delta=final-self.initial
            if not C<abs(delta)<2*C: raise AssertionError('Single-crossing range failed')
            root=phase-math.log1p(-C/abs(delta))/rate
            if not phase<root<DURATION:raise AssertionError('Root outside horizon')
            error=abs(float(state(root))-self.target)
            if error>1e-11:raise AssertionError('Closed-form root check failed')
            self.events=np.asarray([[root,c['sign']]],dtype=np.float64)
        else:
            root=None;error=0.;self.events=np.empty((0,2),dtype=np.float64)
        self.receipt=dict(case=c['name'],reference_type='independent closed-form native V2E RC',
            passed=True,threshold_root_error=error,root_times_s=[root] if root is not None else [],
            state_max_refinement_difference_over_C=0.,event_root_difference_s=0.,
            rounding_of_native_lin_log_not_inserted_into_continuous_law=True)

    def _smooth(self,queries):
        from scipy.integrate import solve_ivp
        c=self.case
        def rhs(t,y):
            intensity=float(physical_intensity(c,t))
            return [float(native_rate(intensity))*(float(native_value(intensity))-y[0])]
        def crossing(t,y):return y[0]-self.target
        crossing.direction=c['sign'];crossing.terminal=False
        maxstep=min(.0002,c['ramp_duration_s']/16)
        answers=[];metadata=[]
        for refinement in (1.,.5):
            sol=solve_ivp(rhs,(0,DURATION),[self.initial],method='DOP853',
                rtol=1e-11*refinement,atol=1e-13*refinement,max_step=maxstep*refinement,
                dense_output=True,events=crossing)
            if not sol.success or len(sol.t_events[0])!=1:
                raise AssertionError('Continuous native oracle failed: '+c['name'])
            root=float(sol.t_events[0][0]);error=abs(float(sol.sol(root)[0])-self.target)
            if error>1e-10:raise AssertionError('Numerical root residual failed')
            answers.append(sol);metadata.append(dict(method='DOP853',rtol=1e-11*refinement,
                atol=1e-13*refinement,max_step_s=maxstep*refinement,nfev=sol.nfev,
                root_time_s=root,root_state_error=error))
        common=np.unique(np.r_[np.linspace(0,DURATION,1201),queries,c['phase_s'],c['phase_s']+c['ramp_duration_s']])
        difference=(answers[0].sol(common)[0]-answers[1].sol(common)[0])/C
        rms=float(np.sqrt(np.mean(difference**2)));p99=float(np.percentile(np.abs(difference),99))
        root_error=abs(metadata[0]['root_time_s']-metadata[1]['root_time_s'])
        passed=rms<.2*BUDGETS['state_rmse_over_C'] and p99<.2*BUDGETS['state_p99_over_C'] and root_error<.2*BUDGETS['matched_time_p95_s']
        if not passed:raise AssertionError('Reference convergence unresolved: '+c['name'])
        self.state=lambda stamps:np.asarray(answers[1].sol(np.asarray(stamps,dtype=np.float64))[0])
        self.events=np.asarray([[metadata[1]['root_time_s'],c['sign']]],dtype=np.float64)
        self.receipt=dict(case=c['name'],reference_type='independent continuous native V2E ODE',passed=True,
            coarse=metadata[0],refined=metadata[1],state_rmse_refinement_difference_over_C=rms,
            state_p99_refinement_difference_over_C=p99,
            state_max_refinement_difference_over_C=float(np.max(np.abs(difference))),
            event_root_difference_s=root_error,
            convergence_budget_fraction=.2,
            rounding_of_native_lin_log_not_inserted_into_continuous_law=True)


def snapshot(emu):
    state={}
    for name in ('new_frame','log_new_frame','lp_log_frame','base_log_frame','diff_frame','timestamp_mem','photoreceptor_noise_arr'):
        value=getattr(emu,name,None)
        state[name]=np.empty((0,),dtype=np.float64) if value is None else value.detach().cpu().numpy().copy()
    state['rng_cpu']=torch.get_rng_state().numpy().copy()
    if str(emu.device).startswith('cuda'):state['rng_cuda']=torch.cuda.get_rng_state().cpu().numpy().copy()
    state['rng_numpy_hash']=np.asarray(hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())
    state['rng_python_hash']=np.asarray(hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
    return state


def equal(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def flatten(values):
    nonempty=[x for x in values if x is not None and len(x)]
    return np.concatenate(nonempty) if nonempty else np.empty((0,4),dtype=np.float32)


def configuration(device):
    return dict(pos_thres=C,neg_thres=C,sigma_thres=0.,cutoff_hz=FC,
        leak_rate_hz=0.,refractory_period_s=REFRACTORY,shot_noise_rate_hz=0.,
        photoreceptor_noise=False,leak_jitter_fraction=0.,noise_rate_cov_decades=0.,seed=42,
        output_folder=None,dvs_h5=None,dvs_aedat2=None,dvs_aedat4=None,dvs_text=None,
        show_dvs_model_state=None,save_dvs_model_state=False,output_width=SHAPE[1],output_height=SHAPE[0],
        device=device,cs_lambda_pixels=None,cs_tau_p_ms=None,hdr=False,scidvs=False,
        record_single_pixel_states=None,label_signal_noise=False)


def run_one(n,kind,cases,references,queries,Emulator,metric_function,args):
    out=args.out/kind/f'N{n:04d}';out.mkdir(parents=True,exist_ok=False)
    count=int(round(DURATION/INTERVAL))*n
    # Same absolute grid construction as the previous production runner.
    times=np.concatenate([j*INTERVAL+np.arange(1,n+1,dtype=np.float64)/n*INTERVAL
                          for j in range(int(round(DURATION/INTERVAL)))])
    times[-1]=DURATION
    grid=np.r_[0.,times]
    right=np.searchsorted(grid,queries,side='left').clip(0,count)
    left=np.maximum(right-1,0)
    needed=set(np.r_[left,right].tolist())
    initial=np.full(SHAPE,.25,dtype=np.float32)
    radiance=np.full((count,*SHAPE),.25,dtype=np.float32)
    for index,case in enumerate(cases):
        y,x=divmod(index,SHAPE[1]);initial[y,x]=case['initial'];radiance[:,y,x]=sample_intensity(case,times)
    initial_dn=(initial*np.float32(255)).astype(np.float32)
    frames=(radiance*np.float32(255)).astype(np.float32)
    if not (float(frames.min())>20 and float(frames.max())<255):raise AssertionError('Sample DN domain escaped')
    cfg=configuration(args.device)
    def fresh():
        obj=Emulator(**cfg)
        if obj.generate_events(initial_dn,0.) is not None:raise AssertionError('Initialization emitted events')
        return obj
    emu=fresh();initial_state=snapshot(emu)
    captured={0:initial_state['lp_log_frame']};events=[]
    torch.cuda.synchronize();begun=time.perf_counter()
    with torch.no_grad():
        for index,(stamp,frame) in enumerate(zip(times,frames),start=1):
            events.append(emu.generate_events(frame,float(stamp)))
            if index in needed:captured[index]=emu.lp_log_frame.detach().cpu().numpy().copy()
    torch.cuda.synchronize();trace_wall=time.perf_counter()-begun
    actual=flatten(events);final_state=snapshot(emu)
    scalars={k:getattr(emu,k) for k in ('t_previous','frame_counter','num_events_total','num_events_on','num_events_off')}
    emu.cleanup()
    # Uninstrumented timing/replay: same seed and public interface, no state
    # transfers at intermediate ticks. V2E's mandatory event .cpu() survives.
    replay=fresh();chunks=[]
    torch.cuda.synchronize();begun=time.perf_counter()
    with torch.no_grad():
        for stamp,frame in zip(times,frames):chunks.append(replay.generate_events(frame,float(stamp)))
    torch.cuda.synchronize();wall=time.perf_counter()-begun
    replay_events=flatten(chunks);replay_state=snapshot(replay)
    physical_equal={key:equal(value,replay_state[key]) for key,value in final_state.items()}
    physical_equal['raw_events']=equal(actual,replay_events)
    physical_equal['scalar_state']=scalars=={k:getattr(replay,k) for k in scalars}
    replay.cleanup()
    weights=np.divide(queries-grid[left],grid[right]-grid[left],out=np.zeros_like(queries),where=right!=left)
    states=np.stack([(1-w)*captured[int(lo)]+w*captured[int(hi)] for lo,hi,w in zip(left,right,weights)])
    np.save(out/'events.npy',actual);np.save(out/'replay_events.npy',replay_events)
    np.save(out/'times.npy',times);np.save(out/'query_times.npy',queries)
    np.save(out/'actual_lp_queries.npy',states)
    np.savez_compressed(out/'native_inputs.npz',initial_DN=initial_dn,frames_DN=frames,physical_radiance_samples=radiance)
    np.savez_compressed(out/'initial_state.npz',**initial_state)
    np.savez_compressed(out/'final_state.npz',**final_state)
    receipt=dict(n_per_10ms=n,actual_input_count=count,dt_min_s=float(np.diff(grid).min()),dt_max_s=float(np.diff(grid).max()),
        trace_wall_s=trace_wall,public_sensor_replay_wall_s=wall,
        timing_scope='8x8 noise-off sensor diagnostic; public per-frame host-to-device and returned events included; initialization/state capture/file I/O excluded',
        native_event_dtype=str(actual.dtype),native_state_dtype=str(final_state['lp_log_frame'].dtype),
        DN_input_dtype=str(frames.dtype),max_unclamped_epsilon=float(np.max((frames.astype(np.float64)+20)/275*np.diff(grid)[:,None,None]/TAU)),
        native_mapping='official lin_log, including native rounding and FP32 output; all fixtures qualified in logarithmic branch',
        native_bandwidth='(DN+20)/275 / tau; not replaced by Genesis bandwidth law',
        config=cfg,trace_vs_uninstrumented_replay_raw_equal=physical_equal,
        noise_off_rng_policy='official event-row randperm consumes CPU RNG; reproducibility checked, not RNG immutability',
        sampler_used=False,interpolator_used=False,sampler_cap_used=False,core_modified=False,
        cuda_path_qualified=True,scalar_state=scalars)
    dump(out/'receipt.json',receipt)
    if not all(physical_equal.values()):raise AssertionError('Instrumentation/replay divergence: '+str(physical_equal))
    padding=int(np.count_nonzero(actual[:,2]*SHAPE[1]+actual[:,1]>=len(cases)))
    if padding:raise AssertionError('Constant padding emitted events')
    rows=[];oracle_rows=[];oracle_states=[]
    for index,(case,ref) in enumerate(zip(cases,references)):
        y,x=divmod(index,SHAPE[1]);observed=actual[(actual[:,1]==x)&(actual[:,2]==y)][:,[0,3]].astype(np.float64)
        truth=ref.state(queries);error=(states[:,y,x]-truth)/C
        rms=float(np.sqrt(np.mean(error**2)));p99=float(np.percentile(np.abs(error),99))
        metrics=metric_function(observed,ref.events,BUDGETS['event_matching_tolerance_s'])
        root_ok=metrics['matched_time_p95_s'] is not None and metrics['matched_time_p95_s']<=BUDGETS['matched_time_p95_s'] if len(ref.events) else len(observed)==0
        count_ok=abs(metrics['count_bias_fraction'])<=BUDGETS['absolute_count_bias_fraction'] if len(ref.events) else len(observed)==0
        passed=bool(rms<=BUDGETS['state_rmse_over_C'] and p99<=BUDGETS['state_p99_over_C'] and metrics['f1']>=BUDGETS['event_f1'] and root_ok and count_ok)
        rows.append(dict(n=n,batch=kind,name=case['name'],group=case['group'],protected=True,
            illumination=case['illumination'],sign=case['sign'],phase_s=case['phase_s'],
            state_rmse_over_C=rms,state_p99_over_C=p99,state_max_over_C=float(np.max(np.abs(error))),
            metrics=metrics,oracle_validation=ref.receipt,passes_registered_pilot_budgets=passed,
            cuda_path_qualified=True,qualified_cuda_passing=passed,reference_model='native_v2e_continuous',
            analytic_events=ref.events.tolist(),actual_events=observed.tolist(),
            raw_events_not_timestamp_normalized_or_requantized=True))
        oracle_rows.extend([[t,x,y,p] for t,p in ref.events]);oracle_states.append(truth)
    np.save(out/'reference_events.npy',np.asarray(oracle_rows,dtype=np.float64).reshape(-1,4))
    np.save(out/'reference_lp_queries.npy',np.stack(oracle_states,axis=1))
    dump(out/'metrics.json',rows)
    return rows,receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--metric-harness',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--n',nargs='+',type=int,default=DEFAULT_NS)
    parser.add_argument('--illumination',nargs='+',type=float,default=[.12,.4,.65])
    parser.add_argument('--kinds',nargs='+',choices=['step','smooth'],default=['step','smooth'])
    parser.add_argument('--query-count',type=int,default=301)
    parser.add_argument('--reference-only',action='store_true')
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:
        args.source_root=args.source_root.resolve(strict=True)
        for name,pin in PINS.items():
            if sha(args.source_root/name)!=pin:raise AssertionError('Source SHA mismatch: '+name)
        inventory={str(p.relative_to(args.source_root)):sha(p) for p in (args.source_root/'v2ecore').rglob('*.py') if '__pycache__' not in p.parts}
        groups={kind:make_cases(args.illumination,kind) for kind in args.kinds}
        queries=np.linspace(0,DURATION,args.query_count,dtype=np.float64)
        references={kind:[Reference(case,queries) for case in cases] for kind,cases in groups.items()}
        reference_receipts={kind:[r.receipt for r in refs] for kind,refs in references.items()}
        dump(args.out/'reference_convergence.json',dict(status='complete',all_passed=True,groups=reference_receipts))
        dump(args.out/'registration.json',dict(schema='native_v2e_sampling_quality_v1',source_root=str(args.source_root),source_sha256=inventory,
            code_sha256=sha(__file__),metric_harness_sha256=sha(args.metric_harness),cases=groups,ns=args.n,
            budgets=BUDGETS,interval_s=INTERVAL,duration_s=DURATION,query_times_s=queries.tolist(),shape=SHAPE,
            model_config=configuration(args.device),radiance_to_native_DN='float32(float32(I)*float32(255)); no gamma, uint8 quantization or clipping',
            native_continuous_law='lin_log(255*I), bandwidth=(255*I+20)/(275*tau); strictly20<DN<255 so map=log(DN)',
            reference_is_ours_law=False,continuous_reference_target_unrounded=True,
            comparison_scope='same optical signals/grids, numerical errors relative to each own native continuous law, not common physical-fidelity or total-simulator ranking',
            auto_sampling_used=False,noise_diagnostic_off=True,production_sources_modified=False,real_data_used=False))
        if args.reference_only:
            print('REFERENCE_ONLY_COMPLETE',flush=True);return
        if not torch.cuda.is_available() or not args.device.startswith('cuda'):
            raise RuntimeError('Actual CUDA required; no silent CPU replacement')
        if any(n<1 or n*3>=32768 for n in args.n):raise ValueError('Invalid registered N')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_device(torch.device(args.device))
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(args.source_root))
        logging.getLogger('v2ecore.emulator').setLevel(logging.ERROR)
        logging.getLogger('v2ecore.emulator_utils').setLevel(logging.ERROR)
        emulator=importlib.import_module('v2ecore.emulator')
        if not Path(emulator.__file__).resolve().is_relative_to(args.source_root):raise AssertionError('Wrong V2E imported')
        spec=importlib.util.spec_from_file_location('previous_sampling_metrics',args.metric_harness)
        metric_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric_module)
        if metric_module.BUDGETS!=BUDGETS:raise AssertionError('Pilot budgets differ')
        rows=[];summaries=[]
        for kind,cases in groups.items():
            for n in args.n:
                print('V2E_SENSOR_START',kind,n,flush=True)
                new,receipt=run_one(n,kind,cases,references[kind],queries,emulator.EventEmulator,metric_module.event_metrics,args)
                rows.extend(new)
                summary=dict(kind=kind,n=n,case_count=len(new),pass_count=sum(x['qualified_cuda_passing'] for x in new),
                    worst_state_rmse_over_C=max(x['state_rmse_over_C'] for x in new),
                    worst_state_p99_over_C=max(x['state_p99_over_C'] for x in new),
                    public_sensor_wall_s=receipt['public_sensor_replay_wall_s'],
                    maximum_unclamped_epsilon=receipt['max_unclamped_epsilon'])
                summaries.append(summary);dump(args.out/'partial_metrics.json',rows)
                print('V2E_SENSOR_DONE',json.dumps(summary),flush=True)
        after={str(p.relative_to(args.source_root)):sha(p) for p in (args.source_root/'v2ecore').rglob('*.py') if '__pycache__' not in p.parts}
        if inventory!=after:raise AssertionError('Pinned source changed')
        dump(args.out/'metrics.json',rows)
        dump(args.out/'summary.json',dict(status='complete',source_unchanged=True,summaries=summaries,
            reference_all_converged=True,universal_or_production_step_recommendation=False,
            own_law_numerical_accuracy_only=True,raw_case_count=len(rows)))
        with (args.out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
            columns=['n','batch','name','group','protected','illumination','sign','phase_s','state_rmse_over_C','state_p99_over_C','state_max_over_C','passes_registered_pilot_budgets','cuda_path_qualified','qualified_cuda_passing']
            writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows({k:r[k] for k in columns} for r in rows)
        index=[dict(path=str(p.relative_to(args.out)),bytes=p.stat().st_size,sha256=sha(p)) for p in args.out.rglob('*') if p.is_file()]
        dump(args.out/'artifact_index.json',index)
        print('V2E_SENSOR_SWEEP_COMPLETE',flush=True)
    except Exception as exc:
        dump(args.out/'FAILURE.json',dict(status='failed',type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
