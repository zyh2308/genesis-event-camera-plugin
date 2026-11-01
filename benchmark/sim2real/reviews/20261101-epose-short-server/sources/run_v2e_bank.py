import os,sys,json,subprocess,time
from pathlib import Path
import h5py,numpy as np
R=Path(os.environ['EPOSE_SHORT_RUN']);P=Path('/home/zhaoyuhan/Eventbased_WAM');source=P/'external/sources/v2e-cac6384445/v2e-pinned';bank=R/'bank';out=bank/'v2e_events';out.mkdir(exist_ok=True)
rows=[]
for name in [f'train_{i:02d}' for i in range(4)]+[f'validation_{i:02d}' for i in range(2)]:
 folder=bank/name;target=out/f'{name}.h5'
 cmd=[sys.executable,str(source/'v2e.py'),'--input',str(folder/'v2e_input'),'--input_frame_rate','100','--auto_timestamp_resolution','False','--timestamp_resolution','0.000025','--slomo_model',str(P/'external/sources/v2e-cac6384445/input/SuperSloMo39.ckpt'),'--output_folder',str(folder/'v2e_output'),'--unique_output_folder','False','--skip_video_output','--no_preview','--dvs_h5',str(target),'--dvs_aedat2','None','--dvs_aedat4','None','--dvs_text','None','--pos_thres','0.2','--neg_thres','0.2','--sigma_thres','0.03','--cutoff_hz','300','--leak_rate_hz','0.01','--leak_jitter_fraction','0.1','--noise_rate_cov_decades','0.1','--shot_noise_rate_hz','0.001','--refractory_period','0.0005','--dvs_emulator_seed','42']
 (folder/'v2e_command.json').write_text(json.dumps(cmd,indent=2));t=time.monotonic()
 with (folder/'v2e.stdout.log').open('w') as a,(folder/'v2e.stderr.log').open('w') as b:code=subprocess.run(cmd,cwd=source,stdout=a,stderr=b,timeout=180).returncode
 assert code==0,(name,code)
 with h5py.File(target) as h:e=np.asarray(h['events'])
 assert len(e)>0 and e.shape[1]==4 and e[:,0].max()<=40001 and e[:,1].max()<346 and e[:,2].max()<260
 counts=[int(((e[:,0]>j*10000)&(e[:,0]<=(j+1)*10000)).sum()) for j in range(4)];assert min(counts)>0,(name,counts)
 row={'rollout_id':name,'events':len(e),'counts_by_10ms':counts,'elapsed_s':time.monotonic()-t};rows.append(row);(bank/'v2e_manifest.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
