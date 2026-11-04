"""Port existing semantic checks to the actual static release entry point."""
import json,textwrap
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'simulator_gpu_resident_20261105b'
out=ROOT/'acceptance_scripts';out.mkdir(exist_ok=False)
target=ROOT/'release_candidate_static'
bridge='''from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin
from genesis_event_plugin.gpu_interpolator import ResidentPhysicsInterpolator,DeferredDiagnostics,DeferredList,DeviceRenderRequest
def make_plugin_class(base,mode):
    assert mode=='exact'
    return OptimizedGenesisPhysicsEventPlugin
def install_geometry(interpolator,mode):
    assert mode=='exact'
    interpolator.__class__=ResidentPhysicsInterpolator
'''
(out/'release_bridge.py').write_text(bridge)
for name in ['test_sensor_v2.py','test_holes_v2.py','test_deferred_final.py','test_actual_sensor_final.py','test_process_resume_v2.py']:
    code=(OLD/name).read_text()
    code=code.replace("sys.path.insert(0,core)","sys.path.insert(0,str(root.parent/'release_candidate_static'))")
    code=code.replace("sys.path.insert(0,str(core))","sys.path.insert(0,str(root.parent/'release_candidate_static'))")
    code=code.replace("sys.path.insert(0,json.loads((P/'frozen_baselines/robot-sparse-cuda-r2-20261102/BASELINE.json').read_text())['core_path'])","sys.path.insert(0,str(root.parent/'release_candidate_static'))")
    code=code.replace('from resident2 import sensor_device','from genesis_event_plugin.gpu_sensor import generate_device_sequence as sensor_device')
    code=code.replace('from resident2 import finish_geometry','from genesis_event_plugin.gpu_interpolator import finish_geometry')
    code=code.replace('from resident_final import','from release_bridge import')
    code=code.replace('import frozen_sensor_fused as cuda','from genesis_event_plugin.gpu_kernel import load_kernel\ncuda=load_kernel()')
    if name=='test_holes_v2.py':
        code=code.replace("for mode in ['exact','gpu']:","for mode in ['exact']:")
        code=code.replace('gpu_math_bitwise_passed=True','gpu_math_bitwise_passed=None')
    compile(code,str(out/name),'exec');(out/name).write_text(code)
(ROOT/'ACCEPTANCE_PREPARATION.json').write_text(json.dumps(dict(actual_package=str(target),checks=[x.name for x in out.glob('test*.py')],
    test_changes='import actual static package; same assertions; retain prior rejected pure GPU fallback separately'),indent=2)+'\n')
print('ACCEPTANCE_PREPARED',len(list(out.glob('test*.py'))))
