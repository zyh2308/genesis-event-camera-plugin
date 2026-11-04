"""One-interval warmup then a fresh-state one-second continuous replay."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
variant=sys.argv[1]
source=(root/'bench_pipeline_final.py').read_text()
source=source.replace("if cursor['i']==len(ts)-1:","if cursor['i']==len(ts)-1 or trial==-1:")
source=source.replace('for j in range(1,len(ts)):', 'for j in range(1,min(2,len(ts)) if trial==-1 else len(ts)):')
sys.argv=[str(root/'bench_pipeline_final.py'),'--variant',variant,'--case','s3_00','--trials','1','--intervals','100','--out','long_resident_final']
exec(compile(source,str(root/'bench_pipeline_final.py'),'exec'),{'__file__':str(root/'bench_pipeline_final.py'),'__name__':'__main__'})
