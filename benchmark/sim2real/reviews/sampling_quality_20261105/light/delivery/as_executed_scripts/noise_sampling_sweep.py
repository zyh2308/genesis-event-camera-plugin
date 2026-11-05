"""Bounded full-profile sensor/noise diagnostics; isolated and read-only core.

This is deliberately NOT a noise-equivalence acceptance test.  The nominal shot
rate is sparse and the nominal leak reference takes roughly 100 s to move by a
threshold.  Short fresh-state runs cannot establish steady-state noise rates.
The script reports exposure, uncertainty, and insufficient-count limitations.
It never increases the noise rates or changes simulator source to obtain counts.

Three separate measurements are kept separate:
  1. Actual gpu_sensor.generate_device_sequence, full registered physical profile.
  2. Actual subtract_leak_current primitive: unthresholded integrated state.
  3. Bounded CUDA float32-uniform tail probe at actual shot probabilities, with
     one independent actual generate_shot_noise/manual-mask equivalence check.
The tail probe shares a pool across N only as a numerical RNG diagnostic.  Full
sensor temporal replicates use distinct seeds across every N and replicate.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from statistics import NormalDist
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
sys.dont_write_bytecode = True

import numpy as np
import torch


MAP_KEYS = ('pos_thres', 'neg_thres', 'noise_rate_array')
EXPECTED = {'cutoff_hz': 300., 'pos_thres': .2, 'neg_thres': .2,
            'sigma_thres': .03, 'leak_rate_hz': .01,
            'shot_noise_rate_hz': .001, 'leak_jitter_fraction': .1,
            'refractory_period_s': .0005}
UPSTREAM_UNIFORM = 'https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/DistributionTemplates.h#L400-L472'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def desc(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(value)
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(a.tobytes()).hexdigest())


def dump(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def finite_summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return dict(n=0, mean=None, std=None, p50=None, p95=None, p99=None)
    return dict(n=int(a.size), mean=float(a.mean()),
                std=float(a.std(ddof=1)) if a.size>1 else None,
                p50=float(np.quantile(a,.5)), p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)), min=float(a.min()), max=float(a.max()))


def poisson_cdf(k, mean):
    if k < 0:
        return 0.
    if mean == 0:
        return 1.
    terms = [i*math.log(mean)-math.lgamma(i+1)-mean for i in range(k+1)]
    maximum = max(terms)
    return min(1., math.exp(maximum)*sum(math.exp(x-maximum) for x in terms))


def poisson_interval(k, confidence=.95):
    """Garwood inversion, explicitly used as a rare-event Poisson model CI."""
    alpha = 1-confidence
    def inverse_cdf(j, target):
        lo, hi = 0., max(1., float(k+1))
        while poisson_cdf(j, hi) > target:
            hi *= 2
        for _ in range(72):
            mid = (lo+hi)/2
            if poisson_cdf(j, mid) > target:
                lo = mid
            else:
                hi = mid
        return (lo+hi)/2
    return [0. if not k else inverse_cdf(k-1,1-alpha/2),
            inverse_cdf(k,alpha/2)]


def uncertainty_for_count(k, exposure):
    interval = poisson_interval(int(k))
    return dict(observed_count=int(k), exposure=float(exposure),
                rate=float(k/exposure),
                poisson_model_rate_ci95=[x/exposure for x in interval],
                ci_assumption='Rare independent-event Poisson approximation; not full sensor-process equivalence',
                sufficient_for_distribution_claim=False,
                sparse_count_warning=int(k)<20)


def event_stats(events, height, width, seconds, refractory):
    if events.shape[1:] != (4,) or events.dtype != np.float64 or not np.isfinite(events).all():
        raise RuntimeError('Actual sensor event contract failed')
    if len(events):
        if np.any(np.diff(events[:,0])<0) or events[:,0].min()<0 or events[:,0].max()>seconds+1e-12:
            raise RuntimeError('Actual event times unsorted or out of range')
        if not (np.all(events[:,1:3] == np.floor(events[:,1:3]))
                and np.all((events[:,1]>=0)&(events[:,1]<width))
                and np.all((events[:,2]>=0)&(events[:,2]<height))
                and np.all(np.isin(events[:,3], [-1,1]))):
            raise RuntimeError('Invalid actual event address/polarity')
    ids = (events[:,2]*width+events[:,1]).astype(np.int64)
    intervals = []
    for pixel in np.unique(ids):
        intervals.extend(np.diff(events[ids==pixel,0]).tolist())
    intervals = np.asarray(intervals,dtype=np.float64)
    edges = np.arange(round(seconds/.01)+1,dtype=np.float64)*.01
    counts = np.histogram(events[:,0],edges)[0]
    return dict(total=int(len(events)), on=int(np.count_nonzero(events[:,3]>0)),
                off=int(np.count_nonzero(events[:,3]<0)),
                pixel_time_s=height*width*seconds,
                isi_seconds=finite_summary(intervals),
                refractory_violations_strict=int(np.count_nonzero(intervals<refractory)),
                refractory_violations_beyond_1ps=int(np.count_nonzero(intervals<refractory-1e-12)),
                window_s=.01, fixed_window_origin_s=0., window_counts=counts.tolist(),
                window_count_variance=float(counts.var(ddof=1)) if len(counts)>1 else None,
                window_count_fano=float(counts.var(ddof=1)/counts.mean()) if len(counts)>1 and counts.mean()>0 else None,
                isi_distribution_sufficient=False,
                no_steady_state_event_rate_claim=True)


def actual_uniform_header():
    path = Path(torch.__file__).resolve().parent/'include/ATen/native/cuda/DistributionTemplates.h'
    result = dict(torch_version=torch.__version__, torch_git_version=torch.version.git_version,
                  upstream_v280_url=UPSTREAM_UNIFORM,
                  float32_dtype_alone_does_not_determine_uniform_mapping=True,
                  uniform_grid_of_2_to_minus24_assumed=False)
    if path.is_file():
        text = path.read_text(encoding='utf-8')
        fragments = {}
        for token in ('void uniform_and_transform(', 'void uniform_kernel('):
            start = text.find(token)
            if start>=0:
                fragments[token] = text[start:start+2300]
        result.update(installed_header=str(path), installed_header_sha256=sha(path),
                      installed_fragments=fragments)
    else:
        result.update(installed_header_available=False,
                      warning='Upstream source does not prove the installed binary mapping; inspect measured CUDA values')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--n',default='63,126,189,377')
    parser.add_argument('--height',type=int,default=32)
    parser.add_argument('--width',type=int,default=32)
    parser.add_argument('--seconds',type=float,default=.1)
    parser.add_argument('--replicates',type=int,default=3)
    parser.add_argument('--tail-draws',type=int,default=1<<26)
    parser.add_argument('--tail-chunk',type=int,default=1<<20)
    parser.add_argument('--seed-base',type=int,default=202611050000)
    args = parser.parse_args()
    if args.height not in (32,64) or args.width not in (32,64):
        raise ValueError('This bounded registered probe supports 32/64-pixel sides only')
    if args.replicates<2 or args.replicates>8 or not (.01<=args.seconds<=1):
        raise ValueError('Use 2–8 short independent replicates, duration .01–1 s')
    if abs(args.seconds/.01-round(args.seconds/.01))>1e-9:
        raise ValueError('Duration must be a multiple of 10 ms')
    ns = [int(x) for x in args.n.split(',')]
    if len(set(ns))!=len(ns) or not set(ns).issubset({63,126,189,252,377}):
        raise ValueError('Explicit bounded N choices required')
    if not (0<args.tail_draws<=1<<28 and 0<args.tail_chunk<=1<<21):
        raise ValueError('Uniform probe draw/memory budget exceeded')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one parent-assigned visible CUDA GPU is required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    core = args.core.resolve(strict=True)
    if any(k=='genesis_event_plugin' or k.startswith('genesis_event_plugin.') for k in sys.modules):
        raise RuntimeError('Use a fresh process for the selected static package')
    profile_path = (args.profile or core/'stage9_sensor_profile.json').resolve(strict=True)
    profile = json.loads(profile_path.read_text(encoding='utf-8'))
    for key,value in EXPECTED.items():
        if profile['noise_config_kwargs'].get(key)!=value:
            raise RuntimeError(f'Registered physical parameter changed: {key}')
    for name,expected in profile['core_canonical_LF_sha256'].items():
        actual = hashlib.sha256((core/'genesis_event_plugin'/name).read_text(encoding='utf-8').encode()).hexdigest()
        if actual!=expected:
            raise RuntimeError('Frozen canonical source mismatch: '+name)
    inventory = {str(p.relative_to(core)):sha(p) for p in core.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix in {'.py','.cu','.cpp','.json'}}
    args.out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('EVENTGENESIS_GPU_BUILD_DIR',str(args.out/'kernel_build'))
    sys.path.insert(0,str(core))
    from genesis_event_plugin.dvs_emulator import DvsEmulator
    from genesis_event_plugin.noise_model import DvsNoiseConfig
    from genesis_event_plugin.gpu_sensor import generate_device_sequence, supported
    from genesis_event_plugin.utils.dvs_core import subtract_leak_current, generate_shot_noise
    config = DvsNoiseConfig(**profile['noise_config_kwargs'])
    if not Path(inspect.getfile(DvsEmulator)).resolve().is_relative_to(core):
        raise RuntimeError('Imported sensor is outside the selected static release')
    levels = np.array([.01,.1,.5,1.],dtype=np.float64)
    band = np.broadcast_to(np.arange(args.width)//(args.width//4),(args.height,args.width))
    frame = levels[band].copy()
    device = torch.device('cuda:0')
    frame_device = torch.tensor(frame,dtype=torch.float64,device=device)
    def create_sensor(seed):
        emu = DvsEmulator(res=frame.shape,config=config,device=str(device))
        emu.initialize(frame,t=0.)
        if not supported(emu):
            raise RuntimeError('Actual registered sensor profile unsupported by selected GPU path')
        # Spatial draws always use cfg.seed.  Only after those maps are created
        # assign a distinct temporal stream for this independent replicate.
        emu.rng.manual_seed(seed)
        return emu
    template = create_sensor(args.seed_base)
    map_descriptors = {key:desc(getattr(template,key)) for key in MAP_KEYS}
    dump(args.out/'uniform_mapping_source.json',actual_uniform_header())
    np.savez(args.out/'fixed_spatial_maps.npz',**{key:getattr(template,key).cpu().numpy() for key in MAP_KEYS},
             linear_intensity=frame)
    reference_rate_array = template.noise_rate_array.cpu().numpy()
    pos_map = template.pos_thres.cpu().numpy()
    characteristic_first_leak = 1/(config.leak_rate_hz*reference_rate_array)
    result = dict(schema='bounded_full_profile_sampling_noise_v1',status='running',
                  core=str(core),core_source_sha256=inventory,
                  sensor_source=str(inspect.getfile(generate_device_sequence)),
                  source_script_sha256=sha(Path(__file__)),profile=str(profile_path),profile_sha256=sha(profile_path),
                  resolved_noise_config=asdict(config),physical_parameters_unchanged=True,
                  height=args.height,width=args.width,linear_intensity_bands=levels.tolist(),
                  ns=ns,seconds=args.seconds,replicates=args.replicates,
                  fixed_spatial_maps=map_descriptors,torch=torch.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(),visible_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  full_noise_temporal_seeds_independent_between_ns=True,
                  first_leak_characteristic_time_s=finite_summary(characteristic_first_leak.ravel()),
                  starts_from_fresh_sensor_not_stationary_age_distribution=True,
                  steady_state_noise_rate_claimed=False,noise_equivalence_gate_passed=False,
                  noise_evidence_status='insufficient_by_design_until_counts_and_distribution_coverage_support_more',
                  rows=[],primitive_leak=[],uniform_tail_probe={},
                  real_data_used=False,training_started=False,production_sources_modified=False)
    dump(args.out/'noise_sampling_results.json',result)
    started = time.perf_counter()
    try:
        # Full sensor path: a short constant input, never a fabricated mature leak
        # population and never a higher rate than the registered physical model.
        for n in ns:
            for rep in range(args.replicates):
                seed = args.seed_base+100000+n*101+rep
                emu = create_sensor(seed)
                if {key:desc(getattr(emu,key)) for key in MAP_KEYS}!=map_descriptors:
                    raise RuntimeError('Spatial fixed-pattern maps changed between replicates')
                rng_initial=desc(emu.rng.get_state())
                chunks=[]
                begin=time.perf_counter()
                sum_dt2=0.
                actual_dt_min=float('inf');actual_dt_max=0.
                for window in range(round(args.seconds/.01)):
                    ts=(np.arange(1,n+1,dtype=np.float64)/n+window)*.01
                    dt=np.diff(np.r_[float(emu.t_previous),ts])
                    sum_dt2+=float(np.sum(dt*dt))
                    actual_dt_min=min(actual_dt_min,float(dt.min()))
                    actual_dt_max=max(actual_dt_max,float(dt.max()))
                    stack=frame_device[None].expand(n,-1,-1)
                    chunks.extend(generate_device_sequence(emu,stack,ts.tolist()))
                    if getattr(emu,'_candidate_fallbacks',0):
                        raise RuntimeError('Actual sensor silently fell back')
                torch.cuda.synchronize()
                wall=time.perf_counter()-begin
                events=np.concatenate(chunks,axis=0) if chunks else np.empty((0,4),dtype=np.float64)
                stats=event_stats(events,args.height,args.width,args.seconds,config.refractory_period_s)
                prefix=f'full_noise_N{n}_rep{rep}'
                np.save(args.out/(prefix+'_events.npy'),events,allow_pickle=False)
                np.savez(args.out/(prefix+'_state.npz'),base_log_frame=emu.base_log_frame.cpu().numpy(),
                         lp_log_frame=emu.lp_log_frame.cpu().numpy(),timestamp_mem=emu.timestamp_mem.cpu().numpy(),
                         rng_state=emu.rng.get_state().cpu().numpy())
                row=dict(n=n,replicate=rep,temporal_seed=seed,dt_s=.01/n,
                         actual_dt_min_s=actual_dt_min,actual_dt_max_s=actual_dt_max,
                         full_profile=True,active_gpu_path=True,fallbacks=0,
                         wall_s=wall,sum_dt_squared_s2=sum_dt2,
                         events=desc(events),statistics=stats,
                         final_state={key:desc(getattr(emu,key)) for key in ('base_log_frame','lp_log_frame','timestamp_mem')},
                         rng_initial=rng_initial,rng_final=desc(emu.rng.get_state()),
                         state_reset_at_t0_only=True)
                result['rows'].append(row)
                dump(args.out/'noise_sampling_results.json',result)
                print('FULL_NOISE_REPLICATE',n,rep,len(events),round(wall,3),flush=True)

        # Unthresholded actual leak primitive.  Collapsing the independent draws
        # into bounded 3-D batches changes no formula, rate, map, jitter or dtype;
        # it is a distribution experiment, not the production draw trajectory.
        for n in ns:
            all_residuals=[]
            dt=.01/n
            steps=round(args.seconds/.01)*n
            mean_drift=(config.leak_rate_hz*template.noise_rate_array*template.pos_thres*args.seconds)
            sd_drift=(config.leak_rate_hz*template.noise_rate_array*template.pos_thres*
                      config.leak_jitter_fraction*math.sqrt(steps*dt*dt))
            for rep in range(args.replicates):
                seed=args.seed_base+300000+n*101+rep
                rng=torch.Generator(device=device).manual_seed(seed)
                drift=torch.zeros_like(template.base_log_frame)
                for left in range(0,steps,128):
                    size=min(128,steps-left)
                    base=torch.zeros((size,args.height,args.width),dtype=torch.float64,device=device)
                    changed=subtract_leak_current(base,config.leak_rate_hz,dt,
                        template.pos_thres[None].expand_as(base),config.leak_jitter_fraction,
                        template.noise_rate_array[None].expand_as(base),generator=rng)
                    drift+=(-changed).sum(dim=0)
                all_residuals.append(((drift-mean_drift)/sd_drift).cpu().numpy().ravel())
            z=np.concatenate(all_residuals)
            count=len(z);variance=float(z.var(ddof=1));df=count-1
            def chi_approx(p):
                return df*(1-2/(9*df)+NormalDist().inv_cdf(p)*math.sqrt(2/(9*df)))**3
            row=dict(n=n,steps=steps,dt_s=dt,draw_dtype='float32',state_dtype='float64',
                     primitive='actual subtract_leak_current; no threshold, reset or refractory in this primitive test',
                     independent_replicates=args.replicates,normalized_residual=finite_summary(z),
                     normalized_mean_ci95=[float(z.mean()-1.96/math.sqrt(count)),float(z.mean()+1.96/math.sqrt(count))],
                     normalized_variance_ci95_approx=[df*variance/chi_approx(.975),df*variance/chi_approx(.025)],
                     ci_assumption='Independent Gaussian primitive draws; variance quantiles use Wilson-Hilferty approximation',
                     mean_log_drift=float(mean_drift.mean()),
                     mean_conditional_variance_log= float((sd_drift*sd_drift).mean()),
                     formula='Var(sum leak)= (rate*noise_map*pos_threshold*jitter)^2 * sum(dt^2)',
                     consequence='At fixed pixel-time the registered per-step jitter integral variance scales with dt; not a density-invariant continuous stochastic law',
                     full_noise_event_distribution_passed=False)
            result['primitive_leak'].append(row)
            dump(args.out/'noise_sampling_results.json',result)
            print('LEAK_PRIMITIVE',n,round(variance,6),flush=True)

        # The production helper's masks must equal the same unmodified formula
        # with a reproduced draw.  This validates comparison promotion/dtypes,
        # without monkeypatching any production RNG or source.
        shape=(1,args.height,args.width)
        emu=template
        actual_helper=[]
        for n in ns:
            dt=.01/n
            gen1=torch.Generator(device=device).manual_seed(args.seed_base+500000+n)
            gen2=torch.Generator(device=device).manual_seed(args.seed_base+500000+n)
            intensity=emu._intensity_fraction(frame_device)[None]
            pos_ratio=(config.pos_thres/emu.pos_thres)[None]
            neg_ratio=(config.neg_thres/emu.neg_thres)[None]
            on,off=generate_shot_noise(config.shot_noise_rate_hz,dt,config.shot_noise_inten_factor,
                                       intensity,pos_ratio,neg_ratio,generator=gen1)
            uniform=torch.rand(shape,dtype=torch.float32,device=device,generator=gen2)
            factor=(config.shot_noise_rate_hz/2)*dt*((config.shot_noise_inten_factor-1)*intensity+1)
            p_on=factor*pos_ratio;p_off=factor*neg_ratio
            equal=bool(torch.equal(on,uniform.double()>(1-p_on)) and torch.equal(off,uniform.double()<p_off)
                       and torch.equal(gen1.get_state(),gen2.get_state()))
            if not equal:
                raise RuntimeError('Actual primitive probability/comparison semantics differ from measured probe')
            actual_helper.append(dict(n=n,actual_helper_vs_manual_same_draw_equal=True,
                                      uniform_dtype=str(uniform.dtype),probability_dtype=str(p_on.dtype),
                                      p_on_min=float(p_on.min()),p_on_max=float(p_on.max()),
                                      p_off_min=float(p_off.min()),p_off_max=float(p_off.max())))

        tail=dict(draws=0,requested_draws=args.tail_draws,chunk_values_max=args.tail_chunk,
                  seed=args.seed_base+700000,uniform_dtype='float32',threshold_dtype='float64',
                  independent_of_full_sensor_temporal_replicates=True,
                  shared_uniforms_across_ns_for_numeric_probe_only=True,
                  torch_v280_mapping_source=UPSTREAM_UNIFORM,
                  observed_float32_support_not_assumed_uniform_24_bit_grid=True,
                  actual_helper_checks=actual_helper,zeros=0,ones=0,
                  min_positive=None,max=None,not_on_2minus24_grid=0,rare_values_bits={},rows=[])
        counts={n:{'on':np.zeros(4,dtype=np.int64),'off':np.zeros(4,dtype=np.int64)} for n in ns}
        generator=torch.Generator(device=device).manual_seed(tail['seed'])
        probability={}
        fraction=emu._intensity_fraction(frame_device)
        for n in ns:
            factor=(config.shot_noise_rate_hz/2)*(.01/n)*((config.shot_noise_inten_factor-1)*fraction+1)
            probability[n]=(factor*(config.pos_thres/emu.pos_thres),factor*(config.neg_thres/emu.neg_thres))
        values_per_plane=args.height*args.width
        planes_per_chunk=max(1,args.tail_chunk//values_per_plane)
        total_planes=args.tail_draws//values_per_plane
        if not total_planes:
            raise ValueError('Tail draw budget must contain at least one full spatial plane')
        band_device=torch.tensor(band,device=device)
        for left in range(0,total_planes,planes_per_chunk):
            planes=min(planes_per_chunk,total_planes-left)
            u=torch.rand((planes,args.height,args.width),dtype=torch.float32,device=device,generator=generator)
            ud=u.double()
            zero=int((u==0).sum());tail['zeros']+=zero;tail['ones']+=int((u==1).sum())
            if zero<u.numel():
                value=float(u[u>0].min())
                tail['min_positive']=value if tail['min_positive'] is None else min(tail['min_positive'],value)
            value=float(u.max());tail['max']=value if tail['max'] is None else max(tail['max'],value)
            scaled=ud*(1<<24)
            tail['not_on_2minus24_grid']+=int((scaled!=scaled.round()).sum())
            rare=u[(u<1e-6)|(u>1-1e-6)].cpu().numpy().view(np.uint32)
            for bits,num in zip(*np.unique(rare,return_counts=True)):
                key=hex(int(bits));tail['rare_values_bits'][key]=tail['rare_values_bits'].get(key,0)+int(num)
            for n in ns:
                p_on,p_off=probability[n]
                on=ud>(1-p_on[None]);off=ud<p_off[None]
                for b in range(4):
                    counts[n]['on'][b]+=int(on[:,band_device==b].sum())
                    counts[n]['off'][b]+=int(off[:,band_device==b].sum())
            tail['draws']+=u.numel()
        for n in ns:
            for b,level in enumerate(levels):
                cell_count=int(np.count_nonzero(band==b))
                trials=total_planes*cell_count
                p_on,p_off=probability[n]
                mean_on=float(p_on[band_device==b].mean());mean_off=float(p_off[band_device==b].mean())
                tail['rows'].append(dict(n=n,linear_intensity=float(level),trials=trials,
                    probability_nominal_on_mean=mean_on,probability_nominal_off_mean=mean_off,
                    nominal_expected_on=trials*mean_on,nominal_expected_off=trials*mean_off,
                    on=uncertainty_for_count(counts[n]['on'][b],trials),
                    off=uncertainty_for_count(counts[n]['off'][b],trials),
                    equivalent_pixel_time_s=trials*(.01/n),
                    distribution_equivalence_claimed=False))
        tail['source_and_empirical_support_require_joint_interpretation']=True
        tail['no_global_float32_grid_inference_from_dtype']=True
        result['uniform_tail_probe']=tail
        print('UNIFORM_TAIL_PROBE',tail['draws'],tail['zeros'],tail['not_on_2minus24_grid'],flush=True)

        # Conservative independent-replicate uncertainty.  The Hoeffding bound
        # uses only the refractory count bound and is intentionally very wide;
        # a zero observed count must never produce a zero-width rate CI.
        groups=[]
        for n in ns:
            rows=[row for row in result['rows'] if row['n']==n]
            totals=[row['statistics']['total'] for row in rows]
            exposure=args.height*args.width*args.seconds*len(rows)
            rate=np.asarray(totals,dtype=np.float64)/(args.height*args.width*args.seconds)
            max_rate=(math.floor(args.seconds/config.refractory_period_s)+1)/args.seconds
            radius=max_rate*math.sqrt(math.log(2/.05)/(2*len(rows)))
            groups.append(dict(n=n,total_events=int(sum(totals)),total_pixel_time_s=exposure,
                on_total=sum(row['statistics']['on'] for row in rows),
                off_total=sum(row['statistics']['off'] for row in rows),
                replicate_rates_hz_per_pixel=rate.tolist(),mean_rate=float(rate.mean()),
                distribution_free_replicate_mean_rate_ci95=[max(0.,float(rate.mean())-radius),min(max_rate,float(rate.mean())+radius)],
                ci_method='Hoeffding over independent temporal replicate rates, refractory-bounded; descriptive and deliberately low-powered',
                refractory_violations=sum(row['statistics']['refractory_violations_beyond_1ps'] for row in rows),
                low_count_or_short_duration_warning=True,steady_state_comparison_qualified=False,
                noise_distribution_equivalence_passed=False))
        result['full_profile_summary']=groups
        for relative,expected in inventory.items():
            if sha(core/relative)!=expected:
                raise RuntimeError('Static core changed during measurement')
        result.update(status='complete',wall_total_s=time.perf_counter()-started,
                      all_local_execution_contracts_passed=True,
                      physical_parameters_unchanged=True,
                      conclusions=['Full-profile short fresh-state results and exact pixel-time are reported, not steady-state leak rates.',
                      'Integrated primitive leak variance is intrinsically step-dependent under the unchanged registered jitter formula.',
                      'Uniform tail probabilities are measured at the unchanged actual thresholds; float32 dtype alone is not a grid model.',
                      'Sparse event counts and short duration do not establish stochastic noninferiority; no noise gate is declared passed.'])
        dump(args.out/'noise_sampling_results.json',result)
        print('NOISE_SWEEP_COMPLETE_EVIDENCE_INSUFFICIENT_FOR_EQUIVALENCE',flush=True)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error),
                      wall_total_s=time.perf_counter()-started)
        dump(args.out/'noise_sampling_results.json',result)
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
