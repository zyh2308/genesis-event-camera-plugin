"""Endpoint-only ablation of the frozen sparse reconstruction, not a new sensor.

The frozen plugin configures its original radiometry/crop and DvsEmulator. This
worker feeds only actual endpoint radiance to that same pixel model. It never
calls capture/interpolate/warp, renders, or filters already-generated events.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,pickle,sys,time
sys.dont_write_bytecode=True

PROFILE_SHA256='dbcf85fe3c86ae21972d4f5b79638018ca44f46a39b8cb1e14e8b6cff3a31a8a'
PACKET_KEYS={'time_s','resolution_wh','intrinsics','radiance'}


def forbidden_provider(*args,**kwargs):
 raise RuntimeError('Endpoint-only generation must not read geometry or invoke a rendering provider')


class EndpointOnlyGenerator:
 def __init__(self,profile,device=None):
  self.profile=profile;self.device=device or profile['plugin_kwargs']['device'];self.plugin=None
  self.initialized=False;self.dvs_update_calls=0;self.source_endpoints=0

 def _packet(self,packet):
  import numpy as np
  if set(packet)!=PACKET_KEYS:raise ValueError('Endpoint worker accepts actual radiance/time/camera calibration only')
  t=float(packet['time_s']);resolution=tuple(map(int,packet['resolution_wh']));K=np.asarray(packet['intrinsics'])
  if not np.isfinite(t) or len(resolution)!=2 or min(resolution)<=0:raise ValueError('Invalid endpoint clock/resolution')
  if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError('Invalid actual camera calibration')
  if self.initialized and (resolution!=self.resolution or not np.array_equal(K,self.intrinsics)):
   raise ValueError('Endpoint camera calibration changed within an episode')
  return t,resolution,K

 def reset(self,packet):
  import numpy as np
  from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
  from genesis_event_plugin.noise_model import DvsNoiseConfig
  t,self.resolution,self.intrinsics=self._packet(packet)
  kwargs=dict(self.profile['plugin_kwargs']);kwargs['device']=self.device
  if str(self.device)=='cpu':kwargs['warp_backend']='numpy'
  self.scene=SimpleNamespace(cur_t=t)
  camera=SimpleNamespace(res=self.resolution,intrinsics=self.intrinsics)
  self.plugin=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**self.profile['noise_config_kwargs']),
   frame_provider=forbidden_provider,motion_state_provider=forbidden_provider,**kwargs)
  self.plugin.attach(self.scene,camera)
  pixels=self.plugin._sensor_crop(self.plugin._radiance(None,provided=packet['radiance']))
  self.plugin.dvs_emulator.initialize(pixels,t)
  self.initialized=True;self.source_endpoints=1;self.dvs_update_calls=0
  self.response_config=asdict(self.plugin.dvs_emulator.cfg)
  self.response_config_sha256=hashlib.sha256(json.dumps(self.response_config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  return np.empty((0,4),dtype=np.float64)

 def generate(self,packet):
  if not self.initialized:raise RuntimeError('Endpoint worker requires a real reset endpoint first')
  t,_,_=self._packet(packet);self.scene.cur_t=t
  pixels=self.plugin._sensor_crop(self.plugin._radiance(None,provided=packet['radiance']))
  # Preserve the frozen pixel model's threshold-crossing timestamps unchanged.
  events=self.plugin.dvs_emulator.generate_events(pixels,t)
  self.source_endpoints+=1;self.dvs_update_calls+=1
  return events


def main():
 wire=sys.stdout.buffer;sys.stdout=sys.stderr
 import numpy as np
 import torch
 from event_wire import encode,decode
 import genesis_event_plugin.physics_plugin as loaded_plugin
 base=Path('/home/zhaoyuhan/Eventbased_WAM')
 profile_path=base/'frozen_baselines/robot-sparse-cuda-r2-20261102/event_sparse_reconstruction_20261102.json'
 if hashlib.sha256(profile_path.read_bytes()).hexdigest()!=PROFILE_SHA256:raise RuntimeError('Frozen response/reconstruction profile changed')
 profile=json.loads(profile_path.read_text());core=base/'runs/robot_sparse_cuda_acceptance_20261102c/core/genesis_event_plugin'
 for name,digest in profile['core_canonical_LF_sha256'].items():
  assert hashlib.sha256((core/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
 assert Path(loaded_plugin.__file__).resolve().parent==core.resolve()
 assert torch.cuda.is_available();torch.set_num_threads(2)
 generator=EndpointOnlyGenerator(profile,device='cuda:0')
 def reply(payload):pickle.dump(encode(payload),wire,protocol=5);wire.flush()
 reply(dict(ready=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
  profile_sha256=PROFILE_SHA256,core_files_verified=17,reconstruction_mode='endpoint_only',intermediate_frames=0,
  external_execution_adapter='No intermediate frame reconstruction; original frozen DVS endpoint updates',
  worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 while True:
  try:request=decode(pickle.load(sys.stdin.buffer))
  except EOFError:break
  if request['kind']=='close':break
  if request['kind'] not in ('reset','frame'):raise ValueError('Unregistered endpoint request kind')
  packet=request['packet'];t=float(packet['time_s']);before=time.monotonic()
  events=generator.reset(packet) if request['kind']=='reset' else generator.generate(packet)
  events=events.detach().cpu().numpy() if hasattr(events,'detach') else np.asarray(events)
  assert events.shape==(len(events),4) and (not len(events) or float(events[:,0].max())<=t+1e-8)
  reply(dict(events=events,packet_end_s=t,endpoint_reads=generator.source_endpoints,dense_HDR_reads=0,
   renderer_calls_in_reconstruction=0,wall_s=time.monotonic()-before,reconstruction_mode='endpoint_only',
   intermediate_frames=0,dvs_update_calls=generator.dvs_update_calls,dvs_response_config_sha256=generator.response_config_sha256))


if __name__=='__main__':main()
