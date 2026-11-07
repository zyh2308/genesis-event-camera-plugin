import argparse,csv,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties

FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','font.size':11})

def metric(record,key,stat='median'):
    return (record.get(key) or {}).get(stat)

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.data.read_text(encoding='utf-8'));assert data['complete'] and not data['violations'],data.get('violations');lookup={(r['case_id'],r['mode']):r for r in data['records']};conditions=['normal','fast','backlit'];names={'normal':'普通','fast':'高速','backlit':'逆光'};rows=[];summary={}
    for condition in conditions:
        er=[r for r in data['records'] if r['mode']=='event' and r['condition']==condition];gr=[lookup[(r['case_id'],'state_reference')] for r in er]
        summary[condition]=dict(n=len(er),event_successes=sum(r['result']['success'] for r in er),state_reference_successes=sum(r['result']['success'] for r in gr),event_contacts=sum(r['result'].get('hand_contact_steps',0)>0 for r in er),paired=dict(both_success=sum(e['result']['success'] and g['result']['success'] for e,g in zip(er,gr)),event_only=sum(e['result']['success'] and not g['result']['success'] for e,g in zip(er,gr)),reference_only=sum(not e['result']['success'] and g['result']['success'] for e,g in zip(er,gr)),neither=sum(not e['result']['success'] and not g['result']['success'] for e,g in zip(er,gr))))
    for case_id in sorted({r['case_id'] for r in data['records']}):
        e=lookup[(case_id,'event')];g=lookup[(case_id,'state_reference')];r=e['result'];initial=r.get('initial',g['result']['initial']);v=initial['velocity'];xyz=initial['position'];camera=e.get('camera_fov') or {};n=camera.get('samples',0)
        category='完成接取' if r['success'] else ('初始化失败' if not e['initialization']['valid'] else ('未接触' if not r.get('hand_contact_steps',0) else '接触后未留住'))
        row=dict(case_id=case_id,seed=e['seed'],condition=names[e['condition']],lighting=e['lighting'],horizontal_speed_m_s=e['horizontal_speed_m_s'],total_initial_speed_m_s=float(np.linalg.norm(v[:3])),initial_x_m=xyz[0],initial_y_m=xyz[1],initial_z_m=xyz[2],initial_values_source='actual event episode' if 'initial' in r else 'matched-seed state reference; event initialization did not advance physics',gravity_z=initial['gravity'][2],release_time_s=initial['release_time_s'],event_success=r['success'],event_outcome=category,actual_hand_contact_steps=r.get('hand_contact_steps',0),first_hand_contact_s=r.get('first_touch'),terminal_s=r['sim_time'],final_ee_distance_m=r.get('final_ee_distance'),state_reference_success=g['result']['success'],state_reference_contact_steps=g['result']['hand_contact_steps'],initial_state_exact_match=e.get('matched_initial_state'),initialization_valid=e['initialization']['valid'],raw_event_count=e.get('raw_event_count',0),precontact_polls=e.get('flight_polls',0),available_precontact_polls=e.get('available_flight_polls',0),fresh_precontact_polls=e.get('fresh_flight_polls',0),pixel_error_median=metric(e,'all_precontact_state_pixel_error'),pixel_error_p95=metric(e,'all_precontact_state_pixel_error','p95'),position_error_median_mm=metric(e,'all_precontact_state_position_error_mm'),position_error_p95_mm=metric(e,'all_precontact_state_position_error_mm','p95'),velocity_error_median_m_s=metric(e,'all_precontact_state_velocity_error_m_s'),velocity_error_p95_m_s=metric(e,'all_precontact_state_velocity_error_m_s','p95'),assumed_grasp_first_time=e.get('assumed_grasp_first_time'),motion_onset_delay_s=e.get('motion_onset_delay_from_release_s'),max_base_displacement_m=r.get('max_base_displacement_m'),fov_samples=n,camera7_center_in_frame_fraction=camera.get('center_inside',0)/n if n else None,camera0_projected_center_fraction_same_trajectory=camera.get('zero_pitch_center_inside_same_robot_trajectory',0)/n if n else None)
        rows.append(row)
    for enc,name in [('utf-8-sig','逐条结果.csv'),('utf-8','episode_results.csv')]:
        with (a.output/name).open('w',encoding=enc,newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (a.output/'逐查询状态误差.csv').open('w',encoding='utf-8-sig',newline='') as f:
        points=data['perception_rows'];w=csv.DictWriter(f,fieldnames=list(points[0]));w.writeheader();w.writerows(points)
    total=dict(episodes=18,independent_seed_blocks=6,event_successes=sum(x['event_success'] for x in rows),reference_successes=sum(x['state_reference_success'] for x in rows),raw_events=sum(x['raw_event_count'] for x in rows),by_condition=summary)
    (a.output/'SUMMARY.json').write_text(json.dumps(total,indent=2,ensure_ascii=False),encoding='utf-8')
    fig=plt.figure(figsize=(15,7.4));grid=fig.add_gridspec(1,2,width_ratios=[1.18,1],wspace=.40);ax=fig.add_subplot(grid[0]);matrix=np.array([[lookup[(f's{i:02d}_{c}','event')]['result']['success'] for c in conditions] for i in range(1,7)],dtype=int);ax.imshow(matrix,cmap=ListedColormap(['#f4dce0','#cceae1']),vmin=0,vmax=1,aspect='auto')
    for i in range(6):
        for j,c in enumerate(conditions):
            e=lookup[(f's{i+1:02d}_{c}','event')];r=e['result'];label=('成功' if r['success'] else '失败')+f" | 接触{r.get('hand_contact_steps',0)}步\n水平 {e['horizontal_speed_m_s']:.1f} m/s";ax.text(j,i,label,ha='center',va='center',fontsize=11,color='#155c51' if r['success'] else '#8c3944')
    ax.set_xticks(range(3),[names[c] for c in conditions]);ax.set_yticks(range(6),[str(2026110801+i) for i in range(6)]);ax.set_ylabel('新 episode seed');ax.set_title('全部18条事件闭环：保留每一条结果',pad=16);ax.spines[['left','bottom']].set_visible(False)
    bx=fig.add_subplot(grid[1]);x=np.arange(3);ev=[summary[c]['event_successes'] for c in conditions];gt=[summary[c]['state_reference_successes'] for c in conditions];bars=bx.bar(x-.18,ev,.34,color='#258b85',label='事件状态＋原策略');bars2=bx.bar(x+.18,gt,.34,color='#758b9c',label='真值含噪状态＋原策略参考')
    for bar in list(bars)+list(bars2):bx.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.12,f'{int(bar.get_height())}/6',ha='center',fontsize=13)
    bx.set_xticks(x,[names[c] for c in conditions]);bx.set_ylim(0,7.3);bx.set_yticks(range(7));bx.set_ylabel('完成接取的回合数');bx.set_title('同来球、同权重的状态参考',pad=16);bx.legend(loc='upper center',fontsize=10,frameon=False);bx.grid(axis='y',alpha=.15);bx.set_axisbelow(True)
    fig.suptitle('7°固定机位泛化验证｜原估计器、原权重、冻结稀疏事件仿真器',fontsize=17,y=.98);fig.text(.05,.025,'每种条件6个新seed；高速与逆光配对。成功沿用原任务评分，并检查真实手部接触。参考失败不等于任务绝对不可接。',fontsize=10,color='#455965');fig.subplots_adjust(bottom=.12,top=.87);fig.savefig(a.output/'03_泛化结果总览.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.5));colors={'normal':'#218c87','fast':'#cc793c','backlit':'#7568ae'}
    for j,c in enumerate(conditions):
        selected=[r for r in rows if r['condition']==names[c]]
        for i,r in enumerate(selected):
            if r['pixel_error_median'] is not None:
                ax.scatter(j+(i-2.5)*.045,max(.01,r['pixel_error_median']),color=colors[c],s=42)
                ax.scatter(j+(i-2.5)*.045,max(.01,r['pixel_error_p95']),edgecolors=colors[c],facecolors='none',s=42)
                ax.plot([j+(i-2.5)*.045]*2,[max(.01,r['pixel_error_median']),max(.01,r['pixel_error_p95'])],color=colors[c],alpha=.3)
                ax.text(j+(i-2.5)*.045,max(.01,r['pixel_error_median'])*1.08,str(r['seed'])[-2:],fontsize=8,ha='center')
    ax.set_yscale('log');ax.set_xticks(range(3),[names[c] for c in conditions]);ax.set_ylabel('接触前控制状态像素误差（px）');ax.set_title('每条回合：实心中位数 / 空心P95；数字为seed末两位');ax.grid(axis='y',alpha=.2)
    for r in rows:
        if r['camera7_center_in_frame_fraction'] is None:continue
        bx.plot([0,1],[100*r['camera0_projected_center_fraction_same_trajectory'],100*r['camera7_center_in_frame_fraction']],color=colors[{v:k for k,v in names.items()}[r['condition']]],alpha=.45,marker='o',markersize=4)
    bx.set_xticks([0,1],['0°几何投影','实际7°']);bx.set_xlim(-.2,1.2);bx.set_ylim(-2,103);bx.set_ylabel('采样到的飞行片段中，目标中心在画面内（%）');bx.set_title('同一归档物理轨迹的视野诊断');bx.grid(axis='y',alpha=.2)
    fig.text(.03,.015,'误差涵盖真实接触前全部可用控制状态，包括静态预测和提前持握假设；缺失状态另计可用率。0°仅改变投影姿态，不是0°闭环实测，也不检验遮挡。',fontsize=10);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(a.output/'04_定位与视野诊断.png',dpi=180);plt.close(fig)
    print(json.dumps(total,ensure_ascii=False))

if __name__=='__main__':main()
