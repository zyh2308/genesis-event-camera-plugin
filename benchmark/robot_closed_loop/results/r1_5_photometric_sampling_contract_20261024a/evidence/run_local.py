from pathlib import Path
import subprocess,json,time
HERE=Path(__file__).resolve().parent
start=time.perf_counter()
cmd=['wsl.exe','-d','Ubuntu-22.04','--','env','PYTHONDONTWRITEBYTECODE=1','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMBA_NUM_THREADS=1','python3','-u','-']
r=subprocess.run(cmd,input=(HERE/'diagnose_cached_regions.py').read_text(),text=True,capture_output=True,encoding='utf-8',timeout=180)
(HERE/'cached_regions.stderr').write_text(r.stderr,encoding='utf-8')
(HERE/'cached_regions.stdout').write_text(r.stdout,encoding='utf-8')
if r.returncode: print(r.stderr[-5000:]);raise SystemExit(r.returncode)
j=json.loads(r.stdout);j['local_runner_wall_s']=time.perf_counter()-start
(HERE/'cached_regions.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
print('rows',len(j['rows']),'wall_s',j['local_runner_wall_s'])
for x in j['rows']:
    if x['K_center_offset_px']==0 and 'interior_supported' in x:
        print(x['episode'],x['time_s'],x['alpha'],x['candidate'],'support',round(x['supported_fraction'],4),'edge',x['contour_band_2px_supported'],'interior',x['interior_supported'])
