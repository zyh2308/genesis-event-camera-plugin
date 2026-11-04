"""Diagnostic-only Python wall profile; excluded from performance comparisons."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'bench_pipeline.py').read_text()
source=source.replace('    torch.cuda.reset_peak_memory_stats();','    import cProfile\n    profiler=cProfile.Profile();profiler.enable()\n    torch.cuda.reset_peak_memory_stats();')
source=source.replace('    events=np.concatenate(chunks)', '    profiler.disable();profiler.dump_stats(str(out/("trial_"+str(trial)+".prof")))\n    events=np.concatenate(chunks)')
source+='\nimport pstats\ns=pstats.Stats(str(out/"trial_0.prof"));s.sort_stats("cumulative");s.print_stats(45)\n'
sys.argv=[str(root/'bench_pipeline.py'),'--variant','baseline','--case','s2_00','--trials','1','--out','cpu_profile']
exec(compile(source,str(root/'bench_pipeline.py'),'exec'),{'__name__':'__main__','__file__':str(root/'bench_pipeline.py')})
