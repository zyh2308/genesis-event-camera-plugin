"""44 real episodes; shared baseline cells remain explicitly linked to originals."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

CONDITIONS=['normal100','normal50','normal25','backlit50'];NAMES=['100张/秒','50张/秒','25张/秒','逆光，50张/秒'];METHODS=['ours','v2e','init_only','state_reference']
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':11})

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);data=json.loads(a.data.read_text(encoding='utf-8'));assert data['complete'] and not data['violations'];records=data['records'];assert len(records)==44
 lookup={(x['case_id'],x['mode']):x for x in records};rows=[]
 def confirmed(r):return r['verified_catch']
 for r in sorted(records,key=lambda x:(x['seed'],CONDITIONS.index(x['condition']),METHODS.index(x['mode']))):
  result=r['result'];initial=result.get('initial',{});pos=initial.get('position',[None]*3);vel=initial.get('velocity',[None]*3);onset=r.get('assumed_grasp_first_time');before=r.get('precontact_last_certainly_collision_free_endpoint_s');premature=bool(onset is not None and (result.get('first_touch') is None or (before is not None and onset<before-1e-9)))
  outcome='完成接取' if confirmed(r) else ('初始化失败' if (r.get('initialization') or {}).get('valid') is False else ('提前认为抓住，随后失败' if premature else ('未碰到球' if not result.get('hand_contact_steps',0) else '接触后未留住')))
  def m(key,stat):return (r.get(key) or {}).get(stat)
  rows.append({'dataset':'cadence_new_seed','case_id':r['case_id'],'seed':r['seed'],'condition':r['condition'],'input_image_hz':r['endpoint_hz'],'method':r['mode'],'raw_time_limit_success_flag':result['success'],'terminated':result.get('terminated',False),'terminal_flag_conflict':r['terminal_flag_conflict'],'verified_catch':confirmed(r),'outcome':outcome,'actual_hand_contact_steps':result.get('hand_contact_steps',0),'final_step_contact':r.get('final_hand_contact'),'first_contact_step_end_s':result.get('first_touch'),'terminal_s':result['sim_time'],'initial_x_m':pos[0],'initial_y_m':pos[1],'initial_z_m':pos[2],'initial_vx_m_s':vel[0],'initial_vy_m_s':vel[1],'initial_vz_m_s':vel[2],'matched_physical_initial_state':r.get('matched_initial_state'),'raw_events':r.get('raw_event_count',0),'precontact_polls':r.get('flight_polls',0),'available_precontact_polls':r.get('available_flight_polls',0),'fresh_precontact_polls':r.get('fresh_flight_polls',0),'pixel_error_median':m('all_precontact_state_pixel_error','median'),'pixel_error_p95':m('all_precontact_state_pixel_error','p95'),'position_error_median_mm':m('all_precontact_state_position_error_mm','median'),'position_error_p95_mm':m('all_precontact_state_position_error_mm','p95'),'velocity_error_median_m_s':m('all_precontact_state_velocity_error_m_s','median'),'velocity_error_p95_m_s':m('all_precontact_state_velocity_error_m_s','p95'),'assumed_grasp_first_s':onset,'definitely_assumed_before_actual_contact_interval':premature,'server_output':r['output']})
 for name,encoding in [('逐条44回合结果.csv','utf-8-sig'),('cadence_episode_results.csv','utf-8')]:
  with (a.output/name).open('w',encoding=encoding,newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 points=data['perception_rows']
 if points:
  for x in points:x['input_image_hz']=lookup[(x['case_id'],x['method'])]['endpoint_hz']
  with (a.output/'逐查询状态误差.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
 summary={'actual_episodes':44,'independent_seed_blocks':4,'rates_hz':[100,50,25],'four_new_seeds':sorted({r['seed'] for r in records}),'by_condition':{},'shared_baselines':'one state-reference per seed; one no-event run per seed/lighting; explicit reuse, not extra independent trials'};table=[]
 for condition in CONDITIONS:
  summary['by_condition'][condition]={}
  for method in METHODS:
   candidates=[r for r in records if r['condition']==condition and r['mode']==method]
   reused=False
   if not candidates:
    reused=True;source_condition='normal100' if method=='state_reference' or condition!='backlit50' else 'backlit50';candidates=[r for r in records if r['condition']==source_condition and r['mode']==method]
   assert len(candidates)==4
   summary['by_condition'][condition][method]={'successes':sum(confirmed(r) for r in candidates),'n':4,'baseline_reused':reused,'actual_source_cases':[r['case_id'] for r in candidates]}
   for record in candidates:table.append({'display_condition':condition,'method':method,'seed':record['seed'],'verified_catch':confirmed(record),'actual_source_case':record['case_id'],'baseline_reused':reused,'server_output':record['output']})
 with (a.output/'对照表_含明确复用的基线.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
 (a.output/'CADENCE_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 fig,ax=plt.subplots(figsize=(11,5.8));x=np.arange(4);colors=['#218c87','#c17c48'];bars=[]
 for j,method in enumerate(['ours','v2e']):bars+=list(ax.bar(x+(j-.5)*.32,[summary['by_condition'][c][method]['successes'] for c in CONDITIONS],.30,color=colors[j],label='Ours' if j==0 else 'V2E'))
 for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+.1,f'{int(b.get_height())}/4',ha='center',fontsize=14)
 ref=summary['by_condition']['normal100']['state_reference']['successes'];ax.axhline(ref,color='#5b6c87',ls='--',lw=1.6,label=f'原任务含噪状态参考 {ref}/4（共用）')
 baseline=[summary['by_condition'][c]['init_only']['successes'] for c in CONDITIONS];ax.plot(x,baseline,color='#97a7ae',marker='x',ls=':',label='停止后续事件（按光照复用）');ax.set_xticks(x,NAMES);ax.set_ylim(-.25,5.2);ax.set_yticks(range(5));ax.set_ylabel('完成接取的回合数');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.legend(loc='upper center',frameon=False,ncol=2,fontsize=10);ax.set_title('同一物理来球，逐步减少输入图像｜Ours / V2E 配对闭环',fontsize=16,pad=16)
 fig.text(.04,.025,'4个全新seed，44个实际回合。7°机位、估计器、原权重和冻结仿真器不变；V2E内部采样精度维持1ms。',fontsize=10);fig.tight_layout(rect=[0,.07,1,1]);fig.savefig(a.output/'01_稀疏输入闭环结果.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(16,5.8),gridspec_kw={'width_ratios':[2.7,1.1]});seeds=summary['four_new_seeds'];matrix=np.array([[confirmed(next(r for r in records if r['seed']==seed and r['condition']==c and r['mode']==m)) for c in CONDITIONS for m in ['ours','v2e']] for seed in seeds],dtype=int);base=np.array([[confirmed(next(r for r in records if r['seed']==seed and r['condition']==c and r['mode']==m)) for c,m in [('normal100','state_reference'),('normal100','init_only'),('backlit50','init_only')]]for seed in seeds],dtype=int)
 for ax,mat in zip(axs,[matrix,base]):
  ax.imshow(mat,cmap=ListedColormap(['#f4dce0','#cceae1']),aspect='auto',vmin=0,vmax=1)
  for i in range(4):
   for j in range(mat.shape[1]):ax.text(j,i,'成功' if mat[i,j] else '失败',ha='center',va='center',fontsize=10)
  ax.set_yticks(range(4),[str(s) for s in seeds]);ax.spines[['left','bottom']].set_visible(False)
 axs[0].set_xticks(range(8),[label+'\n'+m for label in ['100Hz','50Hz','25Hz','逆光50Hz']for m in ['Ours','V2E']]);axs[1].set_xticks(range(3),['含噪\n状态参考','停止事件\n普通','停止事件\n逆光']);axs[0].set_title('32条实际事件回合');axs[1].set_title('12条实际基线回合');fig.suptitle('全部44条结果｜每一条成功和失败均保留',fontsize=17);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.output/'02_全部44回合.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(2,2,figsize=(12,8),sharey=True)
 for ax,c,label in zip(axs.flat,CONDITIONS,NAMES):
  for j,method in enumerate(['ours','v2e']):
   selected=sorted([r for r in rows if r['condition']==c and r['method']==method],key=lambda r:r['seed'])
   for k,row in enumerate(selected):
    if row['pixel_error_median'] is None:continue
    x=j+(k-1.5)*.08;median=max(.01,row['pixel_error_median']);high=max(.01,row['pixel_error_p95']);ax.plot([x,x],[median,high],color=colors[j],alpha=.4);ax.scatter(x,median,color=colors[j],s=35);ax.scatter(x,high,facecolors='none',edgecolors=colors[j],s=35);ax.text(x,median*1.10,str(row['seed'])[-2:],ha='center',fontsize=8)
  ax.set_yscale('log');ax.set_xticks([0,1],['Ours','V2E']);ax.set_title(label);ax.set_ylabel('接触前像素误差（px）');ax.grid(axis='y',alpha=.15)
 fig.suptitle('每条回合：实心中位数 / 空心P95｜包含成功与失败',fontsize=16);fig.text(.04,.012,'误差纳入接触前全部可用控制状态，包括预测及提前保持假设；状态缺失与事件年龄在CSV中另计。',fontsize=10);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(a.output/'03_逐条感知误差.png',dpi=180);plt.close(fig);print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
