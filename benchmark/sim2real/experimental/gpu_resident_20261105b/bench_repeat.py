"""Preserve the first long trace; rerun the untouched reference independently."""
import sys
from pathlib import Path
root=Path(__file__).resolve().parent
source=(root/'bench_long.py').read_text().replace("'long_'+variant","'long_baseline_repeat'")
sys.argv=[str(root/'bench_long.py'),'baseline']
exec(compile(source,str(root/'bench_long.py'),'exec'),{'__file__':str(root/'bench_long.py'),'__name__':'__main__'})
