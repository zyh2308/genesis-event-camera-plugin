"""Check task launch overrides without importing a simulator or touching policy inputs."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

source=ast.parse(Path(__file__).with_name('run_original.py').read_text())
main=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='main')
sampler=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='sample_pose')

def invoke(override,speed=None):
    np.random.seed(41)
    env=SimpleNamespace()
    def native():
        env.object_pos3d=np.random.uniform(-1,1,3)
        env.object_vel6d=np.array([.2,-1.2,2.3,0.,0.,0.])
    ns={'np':np,'env':env,'original_random_pose':native,'args':SimpleNamespace(horizontal_speed=speed,speed_scale=1.),'launch_override':override}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[sampler],type_ignores=[])),'sampler','exec'),ns)
    ns['sample_pose']()
    return env,np.random.random(8)

def test_native_draws_and_override_are_separate():
    native,after_native=invoke(None)
    config={'position':[.65,1.9,1.1],'velocity':[-2.2,-2.,2.6]}
    changed,after_changed=invoke(config)
    np.testing.assert_array_equal(after_native,after_changed)
    np.testing.assert_array_equal(changed.object_pos3d,config['position'])
    np.testing.assert_array_equal(changed.object_vel6d[:3],config['velocity'])
    np.testing.assert_array_equal(changed.object_vel6d[3:],native.object_vel6d[3:])

def test_original_speed_option_retained():
    native,_=invoke(None,2.4)
    assert abs(np.linalg.norm(native.object_vel6d[:2])-2.4)<1e-12
    assert native.object_vel6d[2]==2.3

def test_override_rejects_invalid_values():
    for config in [{'position':[1,2],'velocity':[1,2,3]},{'position':[1,2,3],'velocity':[float('nan'),2,3]}]:
        try:invoke(config)
        except AssertionError:pass
        else:raise AssertionError('Invalid launch accepted')

if __name__=='__main__':
    test_native_draws_and_override_are_separate();test_original_speed_option_retained();test_override_rejects_invalid_values();print('PASS: override RNG independence, native speed, invalid launch checks')
