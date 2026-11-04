"""Actual final package, continuous one-second conformance against saved frozen oracle."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=ROOT.parent/'simulator_gpu_resident_20261105b'
code=(old/'bench_pipeline_final.py').read_text()
code=code.replace('root=Path(__file__).resolve().parent',"root=Path(__file__).resolve().parent\nsys.path.insert(0,str(root.parent/'simulator_gpu_resident_20261105b'))")
code=code.replace("core=Path(pointer['core_path'])","core=root/'release_candidate_static'")
code=code.replace('from resident_final import make_plugin_class','from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin\n    def make_plugin_class(base,mode):return OptimizedGenesisPhysicsEventPlugin')
needle='plugin.attach(scene,cam).start_episode();assert len(plugin.capture())==0;'
assert code.count(needle)==1
code=code.replace(needle,'plugin.attach(scene,cam).start_episode();assert plugin.optimization_active,plugin.optimization_status;assert len(plugin.capture())==0;')
code=code.replace('for j in range(1,len(ts)):', 'for j in range(1,2 if trial==-1 else len(ts)):')
code=code.replace("cursor['i']==len(ts)-1","cursor['i']==(1 if trial==-1 else len(ts)-1)")
code=code.replace('warmup_trials=1,measured_trials=a.trials','warmup_trials=1,warmup_intervals=1,measured_trials=a.trials')
(ROOT/'bench_static_long.py').write_text(code);compile(code,str(ROOT/'bench_static_long.py'),'exec')
print('LONG_STATIC_PREPARED')
