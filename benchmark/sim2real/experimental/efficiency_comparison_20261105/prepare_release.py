"""Build static, isolated release sources from the verified frozen reference.

This preparation is not a release/freeze decision. GPU acceptance follows later.
"""
import ast,hashlib,json,shutil,textwrap
from pathlib import Path
P=Path('/home/zhaoyuhan/Eventbased_WAM');ROOT=Path(__file__).resolve().parent
OLD=P/'runs/simulator_gpu_resident_20261105b';T=P/'runs/simulator_efficiency_20261105a'
R=P/'runs/sim2real/20261103-stage9-direct-v1'
pointer=json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())
CORE=Path(pointer['core_path'])/'genesis_event_plugin'
profile=json.loads((R/'sources/stage9_sensor_profile.json').read_text())
target=ROOT/'release_candidate_static';package=target/'genesis_event_plugin';package.mkdir(parents=True,exist_ok=False)
for name,digest in profile['core_canonical_LF_sha256'].items():
    source=(CORE/name).read_text();assert hashlib.sha256(source.encode()).hexdigest()==digest
    (package/name).write_text(source)
for source in CORE.rglob('*.py'):
    if '__pycache__' in str(source) or source.parent==CORE:continue
    destination=package/source.relative_to(CORE);destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,destination)
(target/'stage9_sensor_profile.json').write_text(json.dumps(profile,indent=2)+'\n')

def segment(path,name,parent=None):
    source=path.read_text();tree=ast.parse(source);nodes=tree.body
    if parent:nodes=next(x for x in nodes if isinstance(x,ast.ClassDef) and x.name==parent).body
    node=next(x for x in nodes if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name==name)
    return textwrap.dedent(''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno]))

# Same exact coverage loop, with only a proved empty-work early exit.
geometry=(CORE/'torch_physics.py').read_text()
assert geometry.count('unresolved = next_unresolved')==1
geometry=geometry.replace('unresolved = next_unresolved','unresolved = next_unresolved\n            if not bool(torch.stack([item.any() for item in unresolved]).any()):\n                break')
(package/'gpu_geometry.py').write_text(geometry)

# Static method: no inspect/exec, no replacement of process-global functions.
pair=segment(CORE/'physics_interpolator.py','_interpolate_pair','PhysicsInterpolator')
pair=pair.replace('from .torch_physics import (','from .gpu_geometry import (')
old='self.last_intermediate_z = [z0.detach().cpu().numpy(), z1.detach().cpu().numpy()]'
assert pair.count(old)==1;pair=pair.replace(old,'self.last_intermediate_z = DeferredList([z0,z1])')
old='for key in torch_visibility:\n            self.last_visibility[key] = torch_visibility[key][0].detach().cpu().numpy()\n        self.last_visibility["intermediate_visibility"] = valid_t[0].detach().cpu().numpy().astype(bool)\n        self.last_visibility["intermediate_unknown"] = ~self.last_visibility["intermediate_visibility"]'
assert pair.count(old)==1;pair=pair.replace(old,'self.last_visibility = DeferredDiagnostics(self.last_visibility)\n        for key in torch_visibility:\n            self.last_visibility[key] = torch_visibility[key][0]')
start=pair.index('        frames = []');end=pair.index('\n\n    projections =',start)
pair=pair[:start]+"        return finish_geometry(self,warped_t,valid_t,alphas,t0,t1,sampling,gradient,_allow_approximate,return_torch,'exact')"+pair[end:]
original=(CORE/'physics_interpolator.py').read_text();tree=ast.parse(original)
imports='\n'.join(ast.get_source_segment(original,x) for x in tree.body if isinstance(x,(ast.Import,ast.ImportFrom)))
helpers='\n\n'.join(segment(OLD/'resident2.py',x) for x in ['DeviceRenderRequest','DeferredDiagnostics','DeferredList','finish_geometry'])
code='''"""Instance-local GPU interpolation; all reference numerical rules retained."""
import torch
from collections.abc import MutableMapping,Sequence
from .physics_interpolator import PhysicsInterpolator,RenderRequest,RenderRequiredError
'''+imports+'\n\n'+helpers+'''

class ResidentPhysicsInterpolator(PhysicsInterpolator):
    @classmethod
    def from_reference(cls,reference):
        result=cls.__new__(cls)
        result.__dict__.update(reference.__dict__)
        return result

    def interpolate_device(self,*args,**kwargs):
        kwargs.pop('return_torch',None)
        return super().interpolate(*args,return_torch=False,**kwargs)

    def interpolate(self,*args,**kwargs):
        requested=kwargs.pop('return_torch',False)
        frames,times=self.interpolate_device(*args,**kwargs)
        if requested:
            self.last_frames_torch=torch.stack(frames)[None]
            return self.last_frames_torch,times
        return list(torch.stack(frames).cpu().numpy()),times

'''+textwrap.indent(pair,'    ')
(package/'gpu_interpolator.py').write_text(code)

sensor=segment(OLD/'prior_candidates.py','fused')
sensor=sensor.replace('def fused(','def generate_device_sequence(',1)
sensor=sensor.replace('import frozen_sensor_fused as cuda','from .gpu_kernel import load_kernel\n    cuda=load_kernel()')
sensor=sensor.replace('if not supported(emu):','if not supported(emu):')
old='prepared=np.stack([emu._prepare_input(f) for f in frames])\n    stack=torch.as_tensor(prepared,device=emu.device,dtype=torch.float64)'
assert sensor.count(old)==1
sensor=sensor.replace(old,"assert torch.is_tensor(frames) and frames.is_cuda\n    if tuple(frames.shape[1:])!=(emu.height,emu.width):raise ValueError('sensor frame shape')\n    if not bool(torch.isfinite(frames).all()) or bool((frames<0).any()):raise ValueError('invalid radiance')\n    stack=frames.to(dtype=torch.float64)*float(emu.cfg.radiance_scale)")
sensor=sensor.replace('return [emu.generate_events(f,t) for f,t in zip(frames,ts)]','host=frames.detach().cpu().numpy()\n        return [emu.generate_events(f,t) for f,t in zip(host,ts)]')
sensor=sensor.replace('emu.new_frame=stack[-1];emu.log_new_frame=log_frames[-1]','emu.new_frame=stack[-1].clone();emu.log_new_frame=log_frames[-1].clone()')
(package/'gpu_sensor.py').write_text('"""FP64 sensor state machine, original RNG order and exact fallback."""\nimport math\nimport numpy as np\nimport torch\n\n'+segment(OLD/'prior_candidates.py','supported')+'\n\n'+sensor)

capture=segment(CORE/'physics_plugin.py','capture','GenesisPhysicsEventPlugin')
capture=capture.replace('        if self.dvs_emulator is None:', '        if self.dvs_emulator is None:',1)
lines=capture.splitlines();lines[1:1]=['    if not self.optimization_active:','        return super().capture(action)'];capture='\n'.join(lines)+'\n'
old='chunks = [\n        self.dvs_emulator.generate_events(frame, ts)\n        for frame, ts in zip(sensor_frames, timestamps)\n    ]'
assert capture.count(old)==1
capture=capture.replace(old,'chunks = self._generate_device_sequence(sensor_frames,timestamps)')
capture=capture.replace('self.interpolator.interpolate(','self.interpolator.interpolate_device(',1)
plugin='''"""Explicit optimized entry; unsupported configurations keep the reference path."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from .physics_plugin import GenesisPhysicsEventPlugin
from .noise_model import DvsNoiseConfig
from .gpu_interpolator import ResidentPhysicsInterpolator,DeferredDiagnostics
from .gpu_sensor import generate_device_sequence

class OptimizedGenesisPhysicsEventPlugin(GenesisPhysicsEventPlugin):
    optimization_active=False

    def attach(self,*args,**kwargs):
        result=super().attach(*args,**kwargs)
        profile=json.loads((Path(__file__).resolve().parent.parent/'stage9_sensor_profile.json').read_text())
        expected=asdict(DvsNoiseConfig(**profile['noise_config_kwargs']))
        reasons=[]
        try:same_profile=asdict(self.noise_config)==expected
        except (TypeError,ValueError):same_profile=False
        if not same_profile:reasons.append('sensor profile outside verified Stage9 configuration')
        if self.warp_backend!='torch' or not str(self.device).startswith('cuda'):reasons.append('requires CUDA Torch geometry')
        if self.warp_composite!='coverage_time_log_linear' or self.disocclusion_policy!='hold':reasons.append('unsupported geometry policy')
        if self.camera_margin is not None or self.intermediate_state_provider is not None:reasons.append('margin or state provider uses reference path')
        if tuple(self.cam.res) not in {(320,240),(346,260)}:reasons.append('resolution outside verified Stage9 shapes')
        self.optimization_active=not reasons
        self.optimization_status={'active':self.optimization_active,'reference_fallback_reasons':reasons}
        if self.optimization_active:self.interpolator=ResidentPhysicsInterpolator.from_reference(self.interpolator)
        return result

    def _generate_device_sequence(self,frames,times):
        value=generate_device_sequence(self.dvs_emulator,torch.stack(frames),times)
        self.fallbacks=getattr(self.dvs_emulator,'_candidate_fallbacks',0)
        return value

    def materialize_diagnostics(self):
        if isinstance(self.interpolator.last_visibility,DeferredDiagnostics):self.interpolator.last_visibility.materialize()
        list(self.interpolator.last_intermediate_z)

'''+textwrap.indent(capture,'    ')
(package/'optimized.py').write_text(plugin)
csrc=package/'csrc';csrc.mkdir()
for name in ['fused_sensor.cpp','fused_sensor.cu']:
    # Match the sources used by the previously accepted compiled binary.
    shutil.copyfile(T/name,csrc/name)
(package/'gpu_kernel.py').write_text('''"""Load a private extension, built without fast math or fused multiply-add."""
import os,threading
from pathlib import Path
import torch
_kernel=None
_lock=threading.Lock()
def load_kernel():
    global _kernel
    with _lock:
        if _kernel is None:
            from torch.utils.cpp_extension import load
            source=Path(__file__).resolve().parent/'csrc'
            build=os.environ.get('EVENTGENESIS_GPU_BUILD_DIR')
            if build:Path(build).mkdir(parents=True,exist_ok=True)
            toolkit=Path(os.environ['CUDA_HOME']) if os.environ.get('CUDA_HOME') else None
            includes=[str(toolkit/'targets/x86_64-linux/include')] if toolkit and (toolkit/'targets/x86_64-linux/include').exists() else []
            links=['-L'+str(toolkit/'targets/x86_64-linux/lib')] if includes else []
            _kernel=load(name='eventgenesis_precision_sensor',sources=[str(source/'fused_sensor.cpp'),str(source/'fused_sensor.cu')],
                extra_cflags=['-O3','-std=c++17'],extra_cuda_cflags=['-O3','--fmad=false','-std=c++17'],
                extra_include_paths=includes,extra_ldflags=links,build_directory=build,verbose=False)
        return _kernel
''')
for path in package.glob('*.py'):compile(path.read_text(),str(path),'exec')
manifest=dict(status='prepared_unaccepted',reference_commit=pointer['release_commit'],core_hashes=profile['core_canonical_LF_sha256'],
    files={str(x.relative_to(target)):hashlib.sha256(x.read_bytes()).hexdigest() for x in target.rglob('*') if x.is_file()},
    design='static instance-local methods; no runtime exec/inspect or global geometry replacement',formal_core_changed=False)
(target/'PREPARATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(prepared=True,files=len(manifest['files']),path=str(target))))
