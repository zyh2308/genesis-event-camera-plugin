"""Create complete per-episode results and paired plots from archived physics."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':11})
METHODS=['ours','v2e','init_only','state_reference']
LABELS={'ours':'Ours','v2e':'V2E','init_only':'仅首帧，无后续事件','state_reference':'真值状态参考'}
CONDITIONS=['normal','visual_fast','backlit'];CN=['普通','高速','高速＋逆光']
COLORS=['#218c87','#c17c48','#99a8b3','#5a6d91']

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 data=json.loads(a.data.read_text(encoding='utf-8'));assert data['complete'] and not data['violations'],data.get('violations')
 records=data['records'];assert len(records)==72;lookup={(r['case_id'],r['mode']):r for r in records}
 rows=[]
 for r in sorted(records,key=lambda r:(r['case_id'],METHODS.index(r['mode']))):
  result=r['result'];initial=result.get('initial',{});v=initial.get('velocity',[None]*3);pos=initial.get('position',[None]*3);available=r.get('available_flight_polls',0);polls=r.get('flight_polls',0)
  def m(name,stat):return (r.get(name) or {}).get(stat)
  rows.append(dict(case_id=r['case_id'],seed=r['seed'],condition=r['condition'],method=r['mode'],success=r['verified_catch'],raw_time_limit_success_flag=result['success'],terminated=result.get('terminated',False),terminal_flag_conflict=r['terminal_flag_conflict'],actual_contact_steps=result.get('hand_contact_steps',0),outcome='完成接取' if r['verified_catch'] else ('未接触' if not result.get('hand_contact_steps',0) else '接触后未留住'),terminal_s=result['sim_time'],initial_x_m=pos[0],initial_y_m=pos[1],initial_z_m=pos[2],initial_vx_m_s=v[0],initial_vy_m_s=v[1],initial_vz_m_s=v[2],matched_initial_state=r.get('matched_initial_state'),initialization_valid=(r.get('initialization') or {}).get('valid'),events=r.get('raw_event_count',0),precontact_polls=polls,available_precontact_polls=available,fresh_precontact_polls=r.get('fresh_flight_polls'),pixel_error_median=m('all_precontact_state_pixel_error','median'),pixel_error_p95=m('all_precontact_state_pixel_error','p95'),position_error_median_mm=m('all_precontact_state_position_error_mm','median'),position_error_p95_mm=m('all_precontact_state_position_error_mm','p95'),velocity_error_median_m_s=m('all_precontact_state_velocity_error_m_s','median'),velocity_error_p95_m_s=m('all_precontact_state_velocity_error_m_s','p95'),assumed_grasp_first_s=r.get('assumed_grasp_first_time'),actual_first_contact_s=result.get('first_touch'),server_output=r['output']))
 for row in rows:
  record=lookup[(row['case_id'],row['method'])]
  row['contact_in_final_control_step']=record.get('final_hand_contact')
  row['original_time_limit_flag']=row['raw_time_limit_success_flag']
 # Include final physical contact explicitly, alongside the unchanged native score.
 for name,encoding in [('逐条结果.csv','utf-8-sig'),('episode_results.csv','utf-8')]:
  with (a.output/name).open('w',encoding=encoding,newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 points=data['perception_rows']
 if points:
  with (a.output/'逐查询误差.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
 summary={'scope':'6 new seed blocks x3 conditions x4 methods; development paired evaluation','by_condition':{},'all_episodes':72,'all_cases_kept':True}
 for c in CONDITIONS:
  summary['by_condition'][c]={}
  for method in METHODS:
   selected=[r for r in rows if r['condition']==c and r['method']==method];summary['by_condition'][c][method]={'successes':sum(r['success'] for r in selected),'contacts':sum(r['actual_contact_steps']>0 for r in selected),'n':len(selected)}
 for method in ['v2e','init_only']:
  pairs=[(lookup[(r['case_id'],'ours')]['verified_catch'],r['verified_catch']) for r in records if r['mode']==method]
  summary['ours_vs_'+method]={'both_success':sum(a and b for a,b in pairs),'ours_only':sum(a and not b for a,b in pairs),'other_only':sum(not a and b for a,b in pairs),'both_fail':sum(not a and not b for a,b in pairs)}
 (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
 fig,axs=plt.subplots(1,3,figsize=(15,5.4),sharey=True)
 for ax,c,label in zip(axs,CONDITIONS,CN):
  counts=[summary['by_condition'][c][m]['successes'] for m in METHODS];bars=ax.bar(range(4),counts,color=COLORS,width=.68)
  for b,n in zip(bars,counts):ax.text(b.get_x()+b.get_width()/2,n+.12,str(n)+'/6',ha='center',fontsize=14)
  ax.set_xticks(range(4),['Ours','V2E','首帧后\n停止事件','真值参考']);ax.set_ylim(0,7);ax.set_yticks(range(7));ax.set_title(label);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
 axs[0].set_ylabel('完成接取的回合数');fig.suptitle('固定机位、原权重与冻结仿真器｜配对闭环验证',fontsize=17)
 fig.text(.05,.02,'6个新seed，三种条件配对；真值参考用于判断物理任务余量。每个预定回合均计入，包括参考失败。',fontsize=10);fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(a.output/'01_配对接取结果.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(16,6))
 from matplotlib.colors import ListedColormap
 for ax,c,label in zip(axs,CONDITIONS,CN):
  matrix=np.array([[lookup[(f's{i:02d}_{c}',m)]['verified_catch'] for m in METHODS] for i in range(1,7)],dtype=int);ax.imshow(matrix,cmap=ListedColormap(['#f4dce0','#cceae1']),aspect='auto',vmin=0,vmax=1)
  for i in range(6):
   for j in range(4):ax.text(j,i,'成功' if matrix[i,j] else '失败',ha='center',va='center',fontsize=10)
  ax.set_xticks(range(4),['Ours','V2E','首帧后\n停止事件','真值参考']);ax.set_yticks(range(6),[str(2026110807+i) for i in range(6)]);ax.set_title(label);ax.spines[['left','bottom']].set_visible(False)
 fig.suptitle('全部72条回合｜不筛除失败',fontsize=17);fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(a.output/'02_全部逐条结果.png',dpi=180);plt.close(fig)
 fig,axs=plt.subplots(1,3,figsize=(15,5.5),sharey=True)
 for ax,c,label in zip(axs,CONDITIONS,CN):
  for j,method in enumerate(METHODS[:3]):
   selected=sorted([r for r in rows if r['condition']==c and r['method']==method],key=lambda r:r['seed'])
   for k,row in enumerate(selected):
    median=row['pixel_error_median'];p95=row['pixel_error_p95']
    if median is None:continue
    x=j+(k-2.5)*.055;ax.plot([x,x],[max(.01,median),max(.01,p95)],color=COLORS[j],alpha=.4);ax.scatter(x,max(.01,median),s=32,color=COLORS[j]);ax.scatter(x,max(.01,p95),s=32,facecolors='none',edgecolors=COLORS[j]);ax.text(x,max(.01,median)*1.09,str(row['seed'])[-2:],fontsize=7,ha='center')
  ax.set_yscale('log');ax.set_xticks(range(3),['Ours','V2E','首帧后停止事件']);ax.set_title(label);ax.grid(axis='y',alpha=.15)
 axs[0].set_ylabel('接触前位置投影误差（px，越低越好）');fig.suptitle('每条回合的感知误差｜实心：中位数；空心：P95',fontsize=17)
 fig.text(.04,.015,'数字为seed末两位。纳入实际接触前全部可用控制状态，包括预测和提前持握假设；状态缺失另计，不以成功回合筛选。',fontsize=9);fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(a.output/'03_逐条像素误差.png',dpi=180);plt.close(fig)
 print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
