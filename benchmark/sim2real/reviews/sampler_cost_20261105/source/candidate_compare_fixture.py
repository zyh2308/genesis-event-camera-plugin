"""Matched-resource comparison; immutable references and lossless memory I/O."""
import argparse,atexit,contextlib,gc,hashlib,json,logging,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import cv2,h5py,numpy as np,torch
from PIL import Image

ROOT=Path(__file__).resolve().parent
P=Path('/home/zhaoyuhan/Eventbased_WAM');R=P/'runs/sim2real/20261103-stage9-direct-v1'
OLD=P/'runs/simulator_gpu_resident_20261105b'
SRC=P/'external/sources/v2e-cac6384445/v2e-pinned'
WEIGHT=P/'external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt'
PROFILE=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
POINTER=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())
CORE=Path('/home/zhaoyuhan/Eventbased_WAM/runs/simulator_sampler_optimization_20261105e/candidate_release')
for name,digest in PROFILE['core_canonical_LF_sha256'].items():
    assert hashlib.sha256((CORE/'genesis_event_plugin'/name).read_text().encode()).hexdigest()==digest
sys.path[:0]=[str(CORE),str(OLD),str(SRC)]
from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
from genesis_event_plugin.noise_model import DvsNoiseConfig
from genesis_event_plugin.adaptive_sampler import AdaptiveSamplingConfig
from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
def make_plugin_class(base,mode):
    assert mode=='exact'
    return OptimizedGenesisPhysicsEventPlugin
from v2ecore.slomo import SuperSloMo
from v2ecore.emulator import EventEmulator

torch.set_num_threads(2);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
logging.basicConfig(level=logging.WARNING)
Resident=make_plugin_class(GenesisPhysicsEventPlugin,'exact')

def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

class Fixture:
    def __init__(self,case,intervals=20):
        self.case=case;kind='pose' if case.startswith('train') else 'rich'
        index=0 if kind=='pose' else 100
        self.path=R/(kind+'_bank/physical')/(case+'.h5')
        with h5py.File(self.path,'r') as h:
            g=h['episodes'][case];self.ts=np.asarray(g['frame_timestamps_s'][index:index+intervals+1],dtype=np.float64)
            self.K=np.asarray(g['camera_intrinsics_K_px']);self.ids=np.asarray(g['object_segmentation_link_ids'])
            self.data={key:np.asarray(g[key][index:index+intervals+1]) for key in ['radiance_linear_rgb','depth_m','segmentation_link_idx','object_to_world','camera_to_world_cv']}
        assert len(self.ts)==intervals+1
        self.ts=self.ts-self.ts[0];self.H,self.W=self.data['depth_m'].shape[1:]
        self.input_dir=ROOT/'inputs'/(case+'_'+str(intervals));self.input_dir.mkdir(parents=True,exist_ok=True)
        original=R/(kind+'_bank/display_inputs')/case
        self.images={}
        for j in range(intervals+1):
            image=cv2.imread(str(original/(str(index+j).zfill(6)+'.png')),cv2.IMREAD_GRAYSCALE)
            assert image is not None and image.shape==(self.H,self.W)
            path=self.input_dir/(str(j).zfill(6)+'.npy')
            if not path.exists():np.save(path,image)
            else:assert np.array_equal(np.load(path),image)
            self.images[str(path)]=image
        self.description=dict(case=case,intervals=intervals,index=index,shape_hw=[self.H,self.W],simulated_s=float(self.ts[-1]),
            source=str(self.path),timestamps_sha256=digest(self.ts),physical_sha256={k:digest(v) for k,v in self.data.items()},
            display_sha256=[digest(x) for x in self.images.values()],intrinsics_sha256=digest(self.K),segmentation_ids_sha256=digest(self.ids))

def emulator(fixture):
    n=PROFILE['noise_config_kwargs']
    names=['pos_thres','neg_thres','sigma_thres','cutoff_hz','leak_rate_hz','leak_jitter_fraction','shot_noise_rate_hz','noise_rate_cov_decades','refractory_period_s','seed']
    return EventEmulator(**{k:n[k] for k in names},output_width=fixture.W,output_height=fixture.H,device='cuda',
        output_folder=None,dvs_h5=None,dvs_aedat2=None,dvs_aedat4=None,dvs_text=None,photoreceptor_noise=False)

def validate(events,fixture):
    assert events.ndim==2 and events.shape[1]==4 and np.isfinite(events).all()
    if len(events):
        assert np.all(np.diff(events[:,0])>=0)
        assert events[:,0].min()>=-1e-7 and events[:,0].max()<=fixture.ts[-1]+1e-7
        assert np.all((events[:,1]>=0)&(events[:,1]<fixture.W)&(events[:,2]>=0)&(events[:,2]<fixture.H))
    return dict(count=len(events),sha256=digest(events),dtype=str(events.dtype),
        t_min=float(events[:,0].min()) if len(events) else None,t_max=float(events[:,0].max()) if len(events) else None)

def write_events(events,path):
    begin=time.perf_counter()
    with h5py.File(path,'x') as h:
        h.create_dataset('t_us',data=np.rint(events[:,0]*1e6).astype(np.uint64))
        h.create_dataset('xy',data=events[:,1:3].astype(np.uint16))
        h.create_dataset('p',data=np.where(events[:,3]>0,1,-1).astype(np.int8))
    return time.perf_counter()-begin

def run_ours(f,variant):
    unrelated_allocated_bytes=torch.cuda.memory_allocated()
    cursor={'i':0};scene=SimpleNamespace(cur_t=0.);cam=SimpleNamespace(res=(f.W,f.H),intrinsics=f.K)
    def frames(*args):
        j=cursor['i'];return dict(radiance=f.data['radiance_linear_rgb'][j],depth=f.data['depth_m'][j],seg=f.data['segmentation_link_idx'][j])
    def motion(*args):
        j=cursor['i'];return dict(camera_to_world=f.data['camera_to_world_cv'][j],object_to_world={int(s):f.data['object_to_world'][j,k] for k,s in enumerate(f.ids)})
    begin=time.perf_counter()
    cls=Resident if variant=='ours_resident' else GenesisPhysicsEventPlugin
    plugin=cls(noise_config=DvsNoiseConfig(**PROFILE['noise_config_kwargs']),sampler_config=AdaptiveSamplingConfig(**PROFILE['sampler_kwargs']),
        frame_provider=frames,motion_state_provider=motion,**PROFILE['plugin_kwargs'])
    plugin.attach(scene,cam).start_episode()
    if variant=='ours_resident':assert plugin.optimization_active,plugin.optimization_status
    assert len(plugin.capture())==0;torch.cuda.synchronize()
    setup=time.perf_counter()-begin;torch.cuda.reset_peak_memory_stats();start=time.perf_counter();chunks=[];sampling=[]
    for j in range(1,len(f.ts)):
        cursor['i']=j;scene.cur_t=float(f.ts[j]);chunks.append(plugin.capture());sampling.append(int(plugin.interpolator.last_sampling.samples))
    events=np.concatenate(chunks);events=events[np.argsort(events[:,0],kind='stable')]
    torch.cuda.synchronize();wall=time.perf_counter()-start
    row=dict(variant=variant,wall_s=wall,setup_s=setup,peak_bytes=torch.cuda.max_memory_allocated()-unrelated_allocated_bytes,worker_peak_bytes=torch.cuda.max_memory_allocated(),unrelated_allocated_bytes=unrelated_allocated_bytes,substeps=sampling,
        fallbacks=getattr(plugin,'fallbacks',0),events=validate(events,f),phases=None)
    plugin.close();del plugin;gc.collect()
    return row,events

@contextlib.contextmanager
def memory_images(slomo,fixture,cache):
    original_save=Image.Image.save;original_load=np.load;original_all=slomo._SuperSloMo__all_images
    def save(image,path,*args,**kwargs):
        cache[int(Path(path).stem)]=np.array(image,copy=True)
    def load(path,*args,**kwargs):
        key=str(path)
        if key in fixture.images:return fixture.images[key].copy()
        return original_load(path,*args,**kwargs)
    Image.Image.save=save;np.load=load
    slomo._SuperSloMo__all_images=lambda folder:[str(Path(folder)/(str(i)+'.png')) for i in sorted(cache)]
    try:yield
    finally:Image.Image.save=original_save;np.load=original_load;slomo._SuperSloMo__all_images=original_all

def interpolate(slomo,f,memory=True,folder=None):
    cache={};folder=folder or str(ROOT/'virtual_png')
    context=memory_images(slomo,f,cache) if memory else contextlib.nullcontext()
    with context:relative,factor=slomo.interpolate(str(f.input_dir),str(folder),(f.W,f.H))
    if memory:images=[cache[i] for i in sorted(cache)]
    else:images=[cv2.imread(str(Path(folder)/(str(i)+'.png')),cv2.IMREAD_GRAYSCALE) for i in range(len(relative))]
    # Native explicit timestamp helper matches the actual midpoint images.
    # Stock CLI applies a separate duration rescaling; it is audited separately.
    timestamps=slomo.get_interpolated_timestamps(f.ts)
    assert len(images)==len(timestamps)==len(relative)==(len(f.ts)-1)*slomo.upsampling_factor
    assert all(x.shape==(f.H,f.W) and x.dtype==np.uint8 for x in images)
    return images,timestamps,float(factor)

def new_slomo(factor=377,batch=8):
    return SuperSloMo(str(WEIGHT),auto_upsample=False,upsampling_factor=factor,batch_size=batch,
        video_path=None,vid_orig=None,vid_slomo=None,preview=False)

def run_v2e(f,slomo):
    start=time.perf_counter();emu=emulator(f);torch.cuda.synchronize();setup=time.perf_counter()-start
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    images,times,factor=interpolate(slomo,f);torch.cuda.synchronize();geometry=time.perf_counter()-start
    start_sensor=time.perf_counter();chunks=[]
    with torch.no_grad():
        for image,t in zip(images,times):
            value=emu.generate_events(image.astype(np.float32),float(t))
            if value is not None and len(value):chunks.append(value)
    events=np.concatenate(chunks) if chunks else np.empty((0,4),dtype=np.float32)
    events=events[np.argsort(events[:,0],kind='stable')];torch.cuda.synchronize()
    wall=time.perf_counter()-start;row=dict(variant='v2e_ssm_memory',wall_s=wall,setup_s=setup,peak_bytes=torch.cuda.max_memory_allocated(),
        events=validate(events,f),substeps=[slomo.upsampling_factor]*(len(f.ts)-1),phases=dict(interpolation_s=geometry,sensor_sort_s=time.perf_counter()-start_sensor),
        input_frames=len(images),first_frame_s=float(times[0]),last_frame_s=float(times[-1]),max_dt_s=float(np.diff(times).max()),factor=factor)
    emu.cleanup();atexit.unregister(emu.cleanup);del emu,images;gc.collect();return row,events

def equivalence(out):
    f=Fixture('s2_00',2);slomo=new_slomo(8,1);start=time.perf_counter()
    frames,times,_=interpolate(slomo,f);folder=out/'actual_png';folder.mkdir()
    disk_frames,disk_times,_=interpolate(slomo,f,False,folder)
    assert np.array_equal(times,disk_times)
    assert len(frames)==len(disk_frames) and all(np.array_equal(a,b) for a,b in zip(frames,disk_frames))
    results=[]
    for images in [frames,disk_frames]:
        e=emulator(f);chunks=[]
        for image,t in zip(images,times):
            v=e.generate_events(image.astype(np.float32),float(t))
            if v is not None and len(v):chunks.append(v)
        results.append(np.concatenate(chunks));e.cleanup()
    assert np.array_equal(*results)
    data=dict(passed=True,frames=len(frames),image_hashes=[digest(x) for x in frames],events_bitwise_equal=True,
        event_count=len(results[0]),time_s=time.perf_counter()-start)
    (out/'memory_equivalence.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--mode',choices=['equivalence','pilot','suite'],default='pilot');p.add_argument('--case',default='s2_00');p.add_argument('--intervals',type=int,default=20);p.add_argument('--trials',type=int,default=3)
    args=p.parse_args();out=Path(args.out);out.mkdir(exist_ok=False,parents=True)
    if args.mode=='equivalence':equivalence(out);return
    cases=[args.case] if args.mode=='pilot' else ['s0_00','s1_00','s2_00','s3_00','train_000']
    rows=[]
    for ci,case in enumerate(cases):
        f=Fixture(case,args.intervals);(out/(case+'_input.json')).write_text(json.dumps(f.description,indent=2)+'\n')
        slomo=new_slomo(2);warm=Fixture(case,2)
        for variant in ['ours_resident','v2e_ssm_memory']:
            r,e=run_ours(warm,variant) if variant=='ours_resident' else run_v2e(Fixture(case,16),slomo)
            (out/(case+'_warm_'+variant+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        slomo.upsampling_factor=377;slomo.batch_size=8
        for trial in range(args.trials):
            variants=['ours_resident','v2e_ssm_memory']
            if (ci+trial)%2:variants.reverse()
            for variant in variants:
                r,e=run_ours(f,variant) if variant=='ours_resident' else run_v2e(f,slomo)
                r.update(case=case,trial=trial,simulated_s=float(f.ts[-1]),device=torch.cuda.get_device_name(),gpu_uuid=os.environ.get('CUDA_VISIBLE_DEVICES'),factor=377)
                r['serialize_s']=write_events(e,out/(case+'_'+variant+'_'+str(trial)+'.h5'))
                np.save(out/(case+'_'+variant+'_'+str(trial)+'_native.npy'),e,allow_pickle=False)
                rows.append(r)
                with (out/'timings.jsonl').open('a') as stream:stream.write(json.dumps(r)+'\n')
                print('TIMING',json.dumps(r),flush=True)
        slomo.cleanup();atexit.unregister(slomo.cleanup);del slomo;gc.collect();torch.cuda.empty_cache()
    (out/'summary.json').write_text(json.dumps(dict(rows=rows,completed=True,real_data_used=False,core_commit=POINTER['release_commit'],v2e_commit='cac638444557a224ba6ee5c29260916f152c00bd',profile_sha256=hashlib.sha256((R/'sources/stage9_sensor_profile.json').read_bytes()).hexdigest()),indent=2)+'\n')

if __name__=='__main__':main()
