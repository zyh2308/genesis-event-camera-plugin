"""Bounded, native V2E noise audit; run only in the parent's GPU2 lease.

No V2E source is patched. Native lin-log, bandwidth offset, noise reset,
initialization, event time assignment and FP32 output are all retained.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                  PYTHONDONTWRITEBYTECODE='1')
import argparse
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import sys
import time
import traceback

PIN = 'cac638444557a224ba6ee5c29260916f152c00bd'
HASHES = {
    'v2ecore/emulator.py': '305dc78517d9d82ed1b6a2199546632a41a6645843a72c837c106bcca0510375',
    'v2ecore/emulator_utils.py': '064cb7dcbf0cc300aa6cd3fb53c572fe3c119c6c50d87133539cb276bb628de9',
}
GPU = 'GPU-a02affe0-7598-579e-ca39-f551f35ad6e9'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--v2e-root', '--source-root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n', default='63,126,189,377')
    ap.add_argument('--seconds', type=float, default=.1)
    ap.add_argument('--replicates', type=int, default=3)
    ap.add_argument('--spatial-seed', type=int, default=42)
    ap.add_argument('--seed-base', type=int, default=202611050000)
    ap.add_argument('--expected-gpu', default=GPU)
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != args.expected_gpu:
            raise RuntimeError('Require the parent-leased physical GPU2 UUID')
        if not (.01 <= args.seconds <= .2 and 2 <= args.replicates <= 8):
            raise ValueError('This is a bounded .01–.2s, 2–8 replicate diagnostic')
        ns = [int(v) for v in args.n.split(',')]
        if len(ns) != len(set(ns)) or not all(2 <= n <= 1508 for n in ns):
            raise ValueError('Unique N values between2 and1508 required')
        root = args.v2e_root.resolve(strict=True)
        actual_hashes = {name:sha(root/name) for name in HASHES}
        if actual_hashes != HASHES:
            raise RuntimeError(f'Wrong locked V2E source bytes: {actual_hashes}')
        if any(k == 'v2ecore' or k.startswith('v2ecore.') for k in sys.modules):
            raise RuntimeError('Import V2E only after selecting its pinned package root')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(root))
        import numpy as np
        import torch
        from scipy.stats import chi2
        emulator_module = importlib.import_module('v2ecore.emulator')
        helper_module = importlib.import_module('v2ecore.emulator_utils')
        for module in (emulator_module, helper_module):
            if not Path(module.__file__).resolve().is_relative_to(root):
                raise RuntimeError('A different installed V2E module was imported')
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one parent-leased CUDA device required')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # V2E normally permits its own RNG/event ordering. Do not replace it by a
        # deterministic custom sensor or patch probability/arithmetic helpers.
        torch.cuda.set_device(0)
        height, width = 32, 32
        optical_levels = np.array([.01, .1, .5, 1.], dtype=np.float64)
        band = np.repeat(np.arange(4), width//4)[None].repeat(height, axis=0)
        frame = (255*optical_levels[band]).astype(np.float32)
        optical = optical_levels[band]
        kwargs = dict(pos_thres=.2, neg_thres=.2, sigma_thres=.03, cutoff_hz=300.,
            leak_rate_hz=.01, refractory_period_s=.0005, shot_noise_rate_hz=.001,
            photoreceptor_noise=False, leak_jitter_fraction=.1,
            noise_rate_cov_decades=.1, seed=args.spatial_seed,
            output_folder=None, dvs_h5=None, dvs_aedat2=None, dvs_aedat4=None,
            dvs_text=None, show_dvs_model_state=[], save_dvs_model_state=False,
            output_width=width, output_height=height, device='cuda:0',
            cs_lambda_pixels=None, cs_tau_p_ms=None, hdr=False, scidvs=False,
            record_single_pixel_states=None, label_signal_noise=False)
        result = dict(schema='native_v2e_bounded_noise_audit_v1', status='running',
            pinned_commit=PIN, source_root=str(root), source_sha256=actual_hashes,
            source_script_sha256=sha(__file__), imported_modules={
                m.__name__:str(m.__file__) for m in (emulator_module,helper_module)},
            resolved_constructor_kwargs=kwargs, parameter_formulas_unchanged=True,
            optical_levels=optical_levels.tolist(), input_DN_levels=(255*optical_levels).tolist(),
            radiometric_mapping='single fixed gain: DN=255*optical_I; float32 DN, no uint8 quantization',
            native_intensity_mapping='(DN+20)/275, not physical I; native lin-log retained',
            ns=ns,seconds=args.seconds,replicates=args.replicates,height=height,width=width,
            spatial_seed_fixed_across_all_runs=args.spatial_seed,
            temporal_seeds_independent_between_N_and_replicates=True,
            torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
            gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],cpu_threads=2,
            starts_from_native_fresh_initialization=True,initial_phase_assumed=False,
            initialization_comment_not_taken_as_executable_behavior=True,
            noise_off_RNG_unchanged_gate_used=False,
            noise_off_rng_note='Native signal-event output may call randperm even with temporal noise disabled.',
            real_data_used=False,training_started=False,production_sources_modified=False,
            full_noise_equivalence_claimed=False,rows=[],primitive_checks=[])
        np.savez_compressed(out/'input_fixture.npz',float32_DN=frame,optical_I=optical,band=band)
        write(out/'noise_results.json',result)
        def snapshot(emu, path):
            state={}
            for key in ('lp_log_frame','base_log_frame','pos_thres','neg_thres',
                        'noise_rate_array','timestamp_mem','new_frame','log_new_frame'):
                value=getattr(emu,key,None)
                if value is not None:
                    state[key]=(value.detach().cpu().numpy().copy() if torch.is_tensor(value)
                                else np.asarray(value))
            state['rng_cpu']=torch.get_rng_state().numpy()
            state['rng_cuda']=torch.cuda.get_rng_state().cpu().numpy()
            np.savez_compressed(path,**state)
            return {key:dict(dtype=str(value.dtype),shape=list(value.shape),
                    sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
                    for key,value in state.items()}
        def small_stats(values):
            a=np.asarray(values,dtype=np.float64)
            return dict(count=int(a.size),mean=float(a.mean()) if a.size else None,
                std=float(a.std()) if a.size else None,
                min=float(a.min()) if a.size else None,max=float(a.max()) if a.size else None)
        def rate_summary(count, exposure):
            lo=0. if count==0 else .5*float(chi2.ppf(.025,2*count))
            hi=.5*float(chi2.ppf(.975,2*(count+1)))
            return dict(count=int(count),pixel_seconds=float(exposure),
                rate_hz_per_pixel=float(count/exposure),
                rare_count_Poisson_approx_CI95=[lo/exposure,hi/exposure],
                ci_scope='Poisson counting approximation only, not proof of full stateful sensor-law equivalence',
                sufficient_events_for_distribution_claim=bool(count>=100))
        spatial_hashes = None
        last_emu=None
        began=time.perf_counter()
        for n in ns:
            steps=int(round(args.seconds/.01*n))
            if abs(steps*(.01/n)-args.seconds)>1e-12:
                raise ValueError('duration must be an integer number of N-derived timesteps')
            times=np.linspace(0.,args.seconds,steps+1,dtype=np.float64)
            for rep in range(args.replicates):
                directory=out/f'N{n:04d}_rep{rep}'
                directory.mkdir()
                emu=emulator_module.EventEmulator(**kwargs)
                initial_events=emu.generate_events(frame,t_frame=0.)
                if initial_events is not None and len(initial_events):
                    raise RuntimeError('Native initialization unexpectedly emitted events')
                # Measure native initialization; do not trust stale comments or
                # subtract a synthetic random phase to force agreement.
                offset=(emu.lp_log_frame-emu.base_log_frame).detach().cpu().numpy()
                native_initial=snapshot(emu,directory/'native_initial_state.npz')
                maps={k:native_initial[k]['sha256'] for k in ('pos_thres','neg_thres','noise_rate_array')}
                if spatial_hashes is None: spatial_hashes=maps
                if maps!=spatial_hashes: raise RuntimeError('Fixed spatial seed did not reproduce native spatial maps')
                seed=args.seed_base+n*1009+rep*97
                torch.manual_seed(seed)
                cpu_initial=torch.get_rng_state().numpy().tobytes()
                cuda_initial=torch.cuda.get_rng_state().cpu().numpy().tobytes()
                chunks=[]
                torch.cuda.synchronize()
                started=time.perf_counter()
                for stamp in times[1:]:
                    ev=emu.generate_events(frame,t_frame=float(stamp))
                    if ev is not None and len(ev): chunks.append(np.asarray(ev).copy())
                torch.cuda.synchronize()
                wall=time.perf_counter()-started
                events=np.concatenate(chunks) if chunks else np.empty((0,4),dtype=np.float32)
                if events.ndim!=2 or events.shape[1]!=4 or not np.isfinite(events).all():
                    raise RuntimeError('Malformed native event output')
                np.save(directory/'events.npy',events,allow_pickle=False)
                final=snapshot(emu,directory/'final_state.npz')
                xy=events[:,1:3].astype(np.int64)
                if len(events) and (not np.array_equal(xy,events[:,1:3]) or
                    (xy<0).any() or (xy[:,0]>=width).any() or (xy[:,1]>=height).any()):
                    raise RuntimeError('Native output pixel addresses invalid')
                if len(events) and not np.isin(events[:,3],[-1,1]).all():
                    raise RuntimeError('Unexpected native polarity convention')
                timestamps=events[:,0].astype(np.float64)
                quantization=2*float(np.spacing(np.float32(max(args.seconds,1e-3))))
                if len(events) and (timestamps.min() < -quantization or timestamps.max()>args.seconds+quantization):
                    raise RuntimeError('Native timestamp outside diagnostic interval')
                isis=[]
                for pixel in np.unique(xy[:,1]*width+xy[:,0]) if len(events) else []:
                    ids=np.flatnonzero(xy[:,1]*width+xy[:,0]==pixel)
                    isis.extend(np.diff(np.sort(timestamps[ids])).tolist())
                isis=np.asarray(isis,dtype=np.float64)
                nbin=int(round(args.seconds/.01))
                if abs(nbin*.01-args.seconds)>1e-12: raise ValueError('Use complete10ms count windows')
                window_counts=np.zeros((height*width,nbin),dtype=np.int64)
                if len(events):
                    bins=np.clip(np.floor(timestamps/.01).astype(np.int64),0,nbin-1)
                    np.add.at(window_counts,(xy[:,1]*width+xy[:,0],bins),1)
                mean=float(window_counts.mean())
                intensity=helper_module.rescale_intensity_frame(emu.new_frame)
                factor=(emu.shot_noise_rate_hz/2)*(.01/n)*((emu.SHOT_NOISE_INTEN_FACTOR-1)*intensity+1)
                pon=factor*emu.pos_thres_pre_prob;poff=factor*emu.neg_thres_pre_prob
                per_band=[]
                for b,level in enumerate(optical_levels):
                    mask=(band==b)
                    use=(band[xy[:,1],xy[:,0]]==b) if len(events) else np.zeros(0,dtype=bool)
                    counts=events[use,3]
                    per_band.append(dict(optical_I=float(level),DN=float(level*255),
                        pixels=int(mask.sum()),on=rate_summary(int((counts==1).sum()),mask.sum()*args.seconds),
                        off=rate_summary(int((counts==-1).sum()),mask.sum()*args.seconds)))
                row=dict(n=n,replicate=rep,temporal_seed=seed,steps=steps,
                    dt_min_s=float(np.diff(times).min()),dt_max_s=float(np.diff(times).max()),
                    initial_lp_minus_base=small_stats(offset),native_random_base_offset_measured_nonzero=bool(np.any(offset!=0)),
                    initial_state=native_initial,final_state=final,spatial_maps_sha256=maps,
                    event_output_dtype=str(events.dtype),new_frame_dtype=str(emu.new_frame.dtype),
                    lp_dtype=str(emu.lp_log_frame.dtype),base_dtype=str(emu.base_log_frame.dtype),
                    shot_probability_dtype=str(pon.dtype),uniform_draw_dtype='torch.float32',
                    actual_intensity_fraction=small_stats(intensity.detach().cpu().numpy()),
                    p_on_min=float(pon.min()),p_on_max=float(pon.max()),p_off_min=float(poff.min()),p_off_max=float(poff.max()),
                    initial_temporal_rng_cpu_sha256=hashlib.sha256(cpu_initial).hexdigest(),
                    initial_temporal_rng_cuda_sha256=hashlib.sha256(cuda_initial).hexdigest(),
                    events=int(len(events)),on=rate_summary(int((events[:,3]==1).sum()),height*width*args.seconds),
                    off=rate_summary(int((events[:,3]==-1).sum()),height*width*args.seconds),
                    per_band=per_band,ISI_s=small_stats(isis),
                    refractory_strict_less_count=int((isis<kwargs['refractory_period_s']).sum()),
                    refractory_beyond_2_FP32_ULP_count=int((isis<kwargs['refractory_period_s']-quantization).sum()),
                    timestamp_2_FP32_ULP_s=quantization,
                    fixed_10ms_count_mean=mean,fixed_10ms_count_variance=float(window_counts.var(ddof=1)),
                    fixed_10ms_Fano_descriptive=float(window_counts.var(ddof=1)/mean) if mean>0 else None,
                    ISI_or_Fano_distribution_claimed=False,wall_s=wall,
                    low_count_warning=bool(len(events)<100),full_noise_equivalence_passed=False)
                np.save(directory/'ISI_s.npy',isis,allow_pickle=False)
                np.save(directory/'counts_10ms.npy',window_counts,allow_pickle=False)
                write(directory/'metrics.json',row)
                result['rows'].append(row)
                write(out/'noise_results.json',result)
                print('V2E_NOISE',n,rep,len(events),round(wall,3),flush=True)
                last_emu=emu
        # Verify native probability comparison with an identical actual CUDA draw.
        # This is a semantics check, not another67m-value random-tail experiment.
        intensity=helper_module.rescale_intensity_frame(last_emu.new_frame)
        for n in ns:
            seed=args.seed_base+700000+n
            torch.manual_seed(seed)
            cstate=torch.get_rng_state();gstate=torch.cuda.get_rng_state()
            on,off=helper_module.generate_shot_noise(.001,.01/n,last_emu.SHOT_NOISE_INTEN_FACTOR,
                intensity,last_emu.pos_thres_pre_prob,last_emu.neg_thres_pre_prob)
            cpu_after=torch.get_rng_state();gpu_after=torch.cuda.get_rng_state()
            torch.set_rng_state(cstate);torch.cuda.set_rng_state(gstate)
            u=torch.rand(size=intensity.shape,dtype=torch.float32,device=intensity.device)
            factor=(.001/2)*(.01/n)*((last_emu.SHOT_NOISE_INTEN_FACTOR-1)*intensity+1)
            pon=factor*last_emu.pos_thres_pre_prob;poff=factor*last_emu.neg_thres_pre_prob
            same=(torch.equal(on,u>(1-pon)) and torch.equal(off,u<poff)
                  and torch.equal(cpu_after,torch.get_rng_state()) and torch.equal(gpu_after,torch.cuda.get_rng_state()))
            if not same: raise RuntimeError('Native shot masks or RNG call differ from same-draw formula')
            gap=float(2**-24)
            result['primitive_checks'].append(dict(n=n,same_draw_native_formula_equal=True,
                probability_dtype=str(pon.dtype),uniform_dtype=str(u.dtype),
                p_on_min=float(pon.min()),p_on_max=float(pon.max()),
                FP32_largest_below_one=float(1-gap),FP32_upper_gap=gap,
                on_branch_structurally_unreachable_pixels=int((pon<=gap).sum()),
                all_fixture_on_probabilities_below_upper_gap=bool((pon<=gap).all()),
                premise='Actual native CUDA torch.rand is in[0,1); FP32 representability near1, not a uniform24bit-grid assumption',
                complete_noise_law_failure_claimed=False))
        if {name:sha(root/name) for name in HASHES}!=actual_hashes:
            raise RuntimeError('V2E source changed during read-only audit')
        result.update(status='complete',source_bytes_unchanged=True,
            measured_native_base_offsets_all_zero=all(not r['native_random_base_offset_measured_nonzero'] for r in result['rows']),
            per_N=[dict(n=n,total_events=sum(r['events'] for r in result['rows'] if r['n']==n),
                on=sum(r['on']['count'] for r in result['rows'] if r['n']==n),
                off=sum(r['off']['count'] for r in result['rows'] if r['n']==n),
                pixel_seconds=args.replicates*height*width*args.seconds,
                comparison_to_EventGenesis_is_descriptive_not_ranking=True) for n in ns],
            wall_total_s=time.perf_counter()-began,
            conclusion='Short fresh-state native-noise audit only. Sparse event counts do not establish distribution equivalence or generator ranking.')
        write(out/'noise_results.json',result)
        write(out/'artifact_index.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p))
            for p in sorted(out.rglob('*')) if p.is_file()])
        print('V2E_NOISE_COMPLETE_NO_EQUIVALENCE_CLAIM',flush=True)
    except Exception as exc:
        write(out/'FAILURE.json',dict(status='failed',type=type(exc).__name__,message=str(exc),
            traceback=traceback.format_exc(),script_sha256=sha(__file__)))
        raise

if __name__=='__main__': main()
