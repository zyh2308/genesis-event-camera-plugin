"""Repeat the declared comparison using the actual integrated release entry."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'compare.py').read_text()
source=source.replace("CORE=Path(POINTER['core_path'])","CORE=ROOT/'release_candidate_static'")
source=source.replace('import argparse,contextlib,gc,','import argparse,atexit,contextlib,gc,')
source=source.replace('from resident_final import make_plugin_class','from genesis_event_plugin.optimized import OptimizedGenesisPhysicsEventPlugin\ndef make_plugin_class(base,mode):\n    assert mode==\'exact\'\n    return OptimizedGenesisPhysicsEventPlugin')
needle="                r['serialize_s']=write_events(e,out/(case+'_'+variant+'_'+str(trial)+'.h5'))"
assert source.count(needle)==1
source=source.replace(needle,needle+"\n                np.save(out/(case+'_'+variant+'_'+str(trial)+'_native.npy'),e,allow_pickle=False)")
source=source.replace("display_sha256=[digest(x) for x in self.images.values()])","display_sha256=[digest(x) for x in self.images.values()],intrinsics_sha256=digest(self.K),segmentation_ids_sha256=digest(self.ids))")
source=source.replace("    cursor={'i':0};scene=SimpleNamespace(cur_t=0.);cam=SimpleNamespace(res=(f.W,f.H),intrinsics=f.K)","    unrelated_allocated_bytes=torch.cuda.memory_allocated()\n    cursor={'i':0};scene=SimpleNamespace(cur_t=0.);cam=SimpleNamespace(res=(f.W,f.H),intrinsics=f.K)")
source=source.replace('peak_bytes=torch.cuda.max_memory_allocated(),substeps=sampling,','peak_bytes=torch.cuda.max_memory_allocated()-unrelated_allocated_bytes,worker_peak_bytes=torch.cuda.max_memory_allocated(),unrelated_allocated_bytes=unrelated_allocated_bytes,substeps=sampling,')
source=source.replace('emu.cleanup();del emu,images;gc.collect();return row,events','emu.cleanup();atexit.unregister(emu.cleanup);del emu,images;gc.collect();return row,events')
source=source.replace('        del slomo;gc.collect();torch.cuda.empty_cache()','        slomo.cleanup();atexit.unregister(slomo.cleanup);del slomo;gc.collect();torch.cuda.empty_cache()')
needle="    plugin.attach(scene,cam).start_episode();assert len(plugin.capture())==0;torch.cuda.synchronize()"
assert source.count(needle)==1
source=source.replace(needle,"    plugin.attach(scene,cam).start_episode()\n    if variant=='ours_resident':assert plugin.optimization_active,plugin.optimization_status\n    assert len(plugin.capture())==0;torch.cuda.synchronize()")
assert 'from resident_final import' not in source
(ROOT/'compare_static.py').write_text(source)
compile(source,str(ROOT/'compare_static.py'),'exec')
print('STATIC_BENCHMARK_PREPARED')
