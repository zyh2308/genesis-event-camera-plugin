"""Actual frozen math on CPU; unused HDF I/O is a hard-fail import stub.

No HDF recording, renderer, physics or GPU is tested. The frozen package imports
its recorder eagerly; every h5py.File call is forbidden rather than simulated.
"""
from pathlib import Path
import ast,hashlib,json,sys,tempfile,types,unittest
sys.dont_write_bytecode=True
def forbidden_hdf(*args,**kwargs):raise RuntimeError('HDF recording is outside this CPU interface test')
hdf_stub=types.ModuleType('h5py');hdf_stub.File=forbidden_hdf;sys.modules['h5py']=hdf_stub
HERE=Path(__file__).resolve().parent
CORE=HERE.parent/'ablation_inputs/frozen_core'
sys.path.insert(0,str(CORE))
import numpy as np
from endpoint_event_worker import EndpointOnlyGenerator,forbidden_provider
from prepare_source_overlay import prepare


class EndpointContracts(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.profile=json.loads((HERE.parent/'ablation_inputs/event_sparse_reconstruction_20261102.json').read_text())
  for name,digest in cls.profile['core_canonical_LF_sha256'].items():
   assert hashlib.sha256((CORE/'genesis_event_plugin'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name

 def packet(self,t,value):
  return dict(time_s=t,resolution_wh=[3,2],intrinsics=np.array([[100.,0.,1.],[0.,100.,1.],[0.,0.,1.]]),radiance=np.full((2,3,3),value,dtype=np.float32))

 def test_actual_frozen_response_and_timestamps_match_direct_endpoint_updates(self):
  from genesis_event_plugin.physics_plugin import GenesisPhysicsEventPlugin
  from genesis_event_plugin.noise_model import DvsNoiseConfig
  kwargs=dict(self.profile['plugin_kwargs']);kwargs.update(device='cpu',warp_backend='numpy')
  reference=GenesisPhysicsEventPlugin(noise_config=DvsNoiseConfig(**self.profile['noise_config_kwargs']),
   frame_provider=forbidden_provider,motion_state_provider=forbidden_provider,**kwargs)
  initial=self.packet(0.,1.);reference.attach(types.SimpleNamespace(cur_t=0.),types.SimpleNamespace(res=(3,2),intrinsics=initial['intrinsics']))
  initial_pixels=reference._sensor_crop(reference._radiance(None,provided=initial['radiance']))
  reference.dvs_emulator.initialize(initial_pixels,0.)
  worker=EndpointOnlyGenerator(self.profile,device='cpu');self.assertEqual(len(worker.reset(initial)),0)
  self.assertIsNone(worker.plugin.output_dir)
  from dataclasses import asdict
  self.assertEqual(worker.response_config,asdict(reference.dvs_emulator.cfg))
  worker.plugin.capture=forbidden_provider;worker.plugin.interpolator.interpolate=forbidden_provider
  got_all=[]
  for t,value in [(0.08,1.),(.16,float(np.exp(.8))),(.24,float(np.exp(.2)))]:
   packet=self.packet(t,value);before=packet['radiance'].copy()
   got=worker.generate(packet);want=reference.dvs_emulator.generate_events(reference._sensor_crop(reference._radiance(None,provided=packet['radiance'])),t)
   np.testing.assert_array_equal(got,want);np.testing.assert_array_equal(packet['radiance'],before);got_all.append(got)
  events=np.concatenate(got_all);self.assertTrue(len(events)>0)
  self.assertTrue(np.any((events[:,0]>.08)&(events[:,0]<.16)))
  self.assertEqual(worker.source_endpoints,4);self.assertEqual(worker.dvs_update_calls,3)

 def test_geometry_input_and_nonincreasing_time_are_rejected(self):
  worker=EndpointOnlyGenerator(self.profile,device='cpu');worker.reset(self.packet(0.,1.))
  bad=self.packet(.08,2.);bad['depth']=np.ones((2,3))
  with self.assertRaises(ValueError):worker.generate(bad)
  with self.assertRaises(ValueError):worker.generate(self.packet(0.,2.))

 def test_prepared_overlay_has_exact_three_source_files_and_unchanged_default_worker(self):
  source=Path('//wsl.localhost/Ubuntu-22.04/home/科研/Eventbased_WAM/repositories/gec-robot-catch-fair-frontend-20261108/benchmark/mature_catch')
  before={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in ('sparse_event_sensor.py','run_original.py')}
  with tempfile.TemporaryDirectory() as temp:
   output=Path(temp)/'overlay';receipt=prepare(source,output)
   self.assertEqual(set(receipt['overlay_sha256']),{'sparse_event_sensor.py','run_original.py','endpoint_event_worker.py'})
   tree=ast.parse((output/'sparse_event_sensor.py').read_text())
   expression=next(x.value for x in ast.walk(tree) if isinstance(x,ast.Assign) and any(isinstance(y,ast.Name) and y.id=='worker_name' for y in x.targets))
   compiled=compile(ast.Expression(expression),'worker_selection','eval')
   for method,mode,name in [('ours','full','event_worker.py'),('ours','endpoint_only','endpoint_event_worker.py'),('v2e','full','v2e_event_worker.py')]:
    self.assertEqual(eval(compiled,{'self':types.SimpleNamespace(generator_method=method,reconstruction_mode=mode)}),name)
   ast.parse((output/'run_original.py').read_text());self.assertFalse(receipt['core_modified'])
  for name,digest in before.items():self.assertEqual(hashlib.sha256((source/name).read_bytes()).hexdigest(),digest)


if __name__=='__main__':unittest.main(verbosity=2)
