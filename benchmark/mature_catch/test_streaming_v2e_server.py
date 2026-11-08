"""Actual CUDA parity against archived official V2E file API on three real endpoints."""
from pathlib import Path
import argparse,atexit,hashlib,json,os,subprocess,time

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--v2e-source',type=Path,required=True);p.add_argument('--endpoint-hz',type=int,choices=[100,50,25],default=100);a=p.parse_args();a.output.mkdir(exist_ok=False)
    gpu=os.environ['EGL_DEVICE_ID'];uuid=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();os.environ['CUDA_VISIBLE_DEVICES']=uuid
    import sys;sys.path.insert(0,str(a.v2e_source))
    import h5py,numpy as np,torch
    from PIL import Image
    from v2ecore.slomo import SuperSloMo
    from v2ecore.emulator import EventEmulator
    from streaming_v2e import StreamingV2E
    torch.set_num_threads(2)
    from cadence_config import minimum_v2e_upsampling
    minimum=minimum_v2e_upsampling(a.endpoint_hz)
    fixture=Path('/home/zhaoyuhan/Eventbased_WAM/runs/robot_scenes/mature_catch/visual_fast_paired_20261108a/paired_evaluation/cases/s02_normal/v2e/migration_audit/sensor_episode_000/sparse_hdr_endpoints.h5')
    with h5py.File(fixture,'r') as h:
        indices=100+np.arange(3)*(100//a.endpoint_hz);gray=h['native_ldr_gray8'][indices];times=h['time_s'][indices]
    src=a.output/'input';dest=a.output/'official_frames';src.mkdir();dest.mkdir()
    for i,frame in enumerate(gray):np.save(src/f'{i:03d}.npy',frame)
    checkpoint=a.v2e_source/'input/SuperSloMo39.ckpt'
    native=SuperSloMo(model=str(checkpoint),auto_upsample=True,upsampling_factor=minimum,batch_size=1,video_path=None,preview=False)
    atexit.unregister(native.cleanup)
    started=time.monotonic();fractional,average=native.interpolate(str(src),str(dest),(640,480))
    official_frames=[np.asarray(Image.open(dest/f'{i}.png')).copy() for i in range(len(fractional))]
    labels=np.array([np.interp(x,[0.,1.,2.],times) for x in fractional])
    stream=StreamingV2E(a.v2e_source,checkpoint,minimum_upsampling=minimum)
    atexit.unregister(stream.slomo.cleanup)
    generated=[];stream_labels=[];interval_counts=[]
    for j in range(2):
        count=0
        for t,frame,u in stream.pair_frames(gray[j],gray[j+1]):
            generated.append(frame);stream_labels.append(times[j]+t*(times[j+1]-times[j]));count+=1
        interval_counts.append(count)
    assert len(generated)==len(official_frames)
    differences=[int(np.max(np.abs(x.astype(np.int16)-y.astype(np.int16)))) for x,y in zip(generated,official_frames)]
    assert max(differences)==0,differences
    assert np.max(np.abs(np.asarray(stream_labels)-labels))<1e-12
    kwargs=dict(pos_thres=.2,neg_thres=.2,sigma_thres=0.,cutoff_hz=0.,leak_rate_hz=0.,refractory_period_s=0.,shot_noise_rate_hz=0.,photoreceptor_noise=False,seed=23,device='cuda',output_width=640,output_height=480,hdr=False)
    reference=EventEmulator(**kwargs);chunks=[]
    for frame,t in zip(official_frames,labels):
        e=reference.generate_events(frame,float(t))
        if e is not None and len(e):chunks.append(e)
    expected=np.concatenate(chunks) if chunks else np.empty((0,4))
    stream.emulator=EventEmulator(**kwargs);stream.previous=None;stream.previous_time=None;chunks=[]
    for frame,t in zip(gray,times):
        e,info=stream.push(frame,float(t))
        if len(e):chunks.append(e)
    actual=np.concatenate(chunks) if chunks else np.empty((0,4))
    def canonical(x):return x[np.lexsort((x[:,3],x[:,2],x[:,1],x[:,0]))]
    assert actual.shape==expected.shape,(actual.shape,expected.shape)
    first=canonical(actual);second=canonical(expected)
    assert np.array_equal(first[:,1:],second[:,1:])
    assert not len(actual) or np.max(np.abs(first[:,0]-second[:,0]))<1e-12
    receipt=stream.receipt();result={'status':'PASS','endpoint_hz':a.endpoint_hz,'minimum_upsampling':minimum,'maximum_configured_subframe_dt_s':.001,'host':os.uname().nodename,'gpu_uuid':uuid,'adapter_sha256':hashlib.sha256(Path(__file__).with_name('streaming_v2e.py').read_bytes()).hexdigest(),'checkpoint_sha256':receipt['checkpoint_sha256'],'source_sha256':receipt['source_sha256'],'fixture':str(fixture),'fixture_scope':'three archived actual native uint8 renderer gray endpoints, strided to the requested acquisition cadence; adapter parity only','frame_times':times.tolist(),'source_fixture_sha256':hashlib.sha256(gray.tobytes()).hexdigest(),'interval_subframes':interval_counts,'compared_frames':len(generated),'maximum_uint8_frame_difference':max(differences),'compared_events':len(actual),'event_xy_polarity_equal':True,'maximum_event_time_difference_s':float(np.max(np.abs(first[:,0]-second[:,0]))) if len(actual) else 0.,'wall_s':time.monotonic()-started,'headless_compatibility':'unused video/GUI cleanup callbacks unregistered; algorithms unchanged'}
    np.savez_compressed(a.output/'paired_events.npz',official=expected,streaming=actual)
    (a.output/'V2E_PARITY_RECEIPT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
