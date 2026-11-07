from pathlib import Path
p=Path(__file__).parent
s=(p/'prepare_pose_train_radiance.py').read_text()
s=s.replace('np.linspace(0,95,8,dtype=int)','np.linspace(0,95,96,dtype=int)')
s=s.replace('eight uniformly spaced ordinal positions in the sorted 96 training trajectories','all 96 original training trajectories, no validation or real trajectories')
assert 'np.linspace(0,95,8,' not in s
(p/'prepare_pose_full_train_radiance.py').write_text(s)
r=(p/'run_pose_light_study.py').read_text()
a=r.index('    specs = [');b=r.index('    completed = []',a)
r=r[:a]+"    specs=[('radiance_train','prepare_pose_full_train_radiance.py',['--campaign',str(campaign),'--out',str(root/'radiance_train')])]\n"+r[b:]
r=r.replace('deadline=time.monotonic()+3*3600','deadline=time.monotonic()+18*3600')
r=r.replace("purpose='paired synthetic pose robustness'","purpose='paired original train-bank illumination coverage; data only'")
r=r.replace("'Own study phase exceeded bounded runtime'","'Own data-preparation phase exceeded 18-hour runtime bound'")
(p/'run_pose_full_light_preparation.py').write_text(r)
for name,body in [('prepare',s),('bounded runner',r)]:compile(body,name,'exec')
print('Full original train-bank data preparation scripts compiled')
