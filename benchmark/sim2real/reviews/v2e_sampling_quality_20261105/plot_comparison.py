"""Scientific comparison plots and tables from immutable measured results."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
font=Path('/mnt/c/Windows/Fonts/msyh.ttc')
if font.exists():
    matplotlib.font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

def table(name,rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if rows:
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

v2e=json.loads((ROOT/'light/v2e_sensor_full_v2/metrics.json').read_text())
summary=[]
for kind,folder in [('step','ours_common_step_v2'),('smooth','ours_common_smooth_v2')]:
    ours=json.loads((ROOT/'light'/folder/'metrics.json').read_text())
    for method,data in [('EventGenesis',ours),('V2E', [x for x in v2e if x['batch']==kind])]:
        for n in sorted({x['n'] for x in data}):
            rows=[x for x in data if x['n']==n];moving=[x for x in rows if x['sign']]
            times=[t*1e6 for x in moving for t in x['metrics']['nearest_same_polarity_errors_s'] if t is not None]
            signed=[(x['actual_events'][0][0]-x['analytic_events'][0][0])*1e6 for x in moving if len(x['actual_events'])==len(x['analytic_events'])==1]
            summary.append(dict(signal=kind,method=method,n_per_10ms=n,dt_us=10000/n,
                worst_event_abs_error_us=max(times),pooled_p95_event_abs_error_us=float(np.percentile(times,95)),
                min_signed_event_error_us=min(signed),max_signed_event_error_us=max(signed),
                worst_state_RMSE_over_C=max(x['state_rmse_over_C'] for x in rows),
                worst_state_p99_over_C=max(x['state_p99_over_C'] for x in rows),
                minimum_case_F1=min(x['metrics']['f1'] for x in moving),
                case_count=len(rows),quality_pass_count=sum(x['qualified_cuda_passing'] for x in rows),
                reference_event_count=sum(x['metrics']['reference_count'] for x in moving),
                actual_event_count=sum(x['metrics']['actual_count'] for x in moving)))
table('sensor_comparison',summary)
fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
for a,kind,title in zip(ax,['step','smooth'],['亮度突变','平滑亮度变化']):
    for method,color in [('EventGenesis','C0'),('V2E','C1')]:
        rs=[x for x in summary if x['signal']==kind and x['method']==method]
        a.plot([x['n_per_10ms'] for x in rs],[x['worst_event_abs_error_us'] for x in rs],'o-',color=color,label=method)
    a.axhline(20,color='C3',ls='--',lw=1,label='本项目诊断预算：20 微秒')
    a.set(xscale='log',yscale='log',xlabel='每 10 毫秒采样步数',ylabel='最坏事件绝对时间误差（微秒）',title=title)
    ticks=[32,63,126,189,252,377,754,1508];a.set_xticks(ticks);a.set_xticklabels(ticks,rotation=40);a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('同光学信号、同采样网格；分别相对各自连续传感器模型评分',fontsize=12)
for ext in ['png','pdf']:fig.savefig(OUT/('sensor_time_comparison.'+ext),dpi=190)
plt.close(fig)

path=ROOT/'light/geometry_common_quantized/common_quantized_geometry.json'
if path.exists():
    geo=json.loads(path.read_text());rows=[]
    for x in geo['rows']:
        rows.append(dict(method=x['method'],mode=x['mode'],n=x.get('N'),
                         all_PSNR_dB=x['strata']['all']['PSNR_dB'],
                         all_RMSE=x['strata']['all']['RMSE'],
                         active_RMSE=x['strata']['active']['RMSE'],
                         boundary_RMSE=x['strata']['boundary']['RMSE'],
                         geometry_wall_s=x.get('geometry_wall_s')))
    table('geometry_display_comparison',rows)
    picked=[r for r in rows if r['method']=='display_endpoint_linear' or r['method']=='EventGenesis' and r['n']==377 or r['method']=='V2E_SuperSloMo' and r['mode'].startswith(('auto_min2','fixed377'))]
    labels=['端点线性插值' if r['method']=='display_endpoint_linear' else 'EventGenesis\n377 步' if r['method']=='EventGenesis' else 'V2E\n原生自动' if r['mode'].startswith('auto_min2') else 'V2E\n377 步' for r in picked]
    colors=['0.6' if r['method']=='display_endpoint_linear' else 'C0' if r['method']=='EventGenesis' else 'C1' for r in picked]
    fig,ax=plt.subplots(1,2,figsize=(9.8,3.9),layout='constrained')
    for a,key,title in [(ax[0],'all_PSNR_dB','全图 PSNR（越高越好）'),(ax[1],'active_RMSE','亮度变化区 RMSE（越低越好）')]:
        values=[r[key] for r in picked];bars=a.bar(labels,values,color=colors,width=.6)
        a.set(title=title);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
        a.bar_label(bars,labels=[f'{v:.2f}' if key=='all_PSNR_dB' else f'{v:.4f}' for v in values],padding=3,fontsize=9)
        a.set_ylim(0,max(values)*1.16)
    fig.suptitle('同一 0.1 秒片段、统一显示映射与最终量化；仅作诊断',fontsize=12)
    for ext in ['png','pdf']:fig.savefig(OUT/('geometry_display_comparison.'+ext),dpi=190)
    plt.close(fig)
print('PLOTS',OUT)
