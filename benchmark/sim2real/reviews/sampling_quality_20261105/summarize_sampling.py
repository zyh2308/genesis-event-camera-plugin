"""Create auditable tables and scientific figures from measured sampling tests."""
from pathlib import Path
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
OUT.mkdir(exist_ok=True)
font = Path('/mnt/c/Windows/Fonts/msyh.ttc')
if font.exists():
    prop = FontProperties(fname=str(font))
    matplotlib.font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = prop.get_name()
plt.rcParams.update({'font.size':10, 'axes.spines.top':False,
                     'axes.spines.right':False, 'pdf.fonttype':42})

def save_table(name, rows):
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if rows:
        with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analytic(source, prefix, title):
    if not source.exists(): return
    data=json.loads(source.read_text())
    rows=[]
    for n in sorted({x['n'] for x in data}):
        protected=[x for x in data if x['n']==n and x['protected']]
        moving=[x for x in protected if x['metrics']['reference_count']]
        sensitivity=[x for x in data if x['n']==n and not x['protected']]
        times=[v for x in moving for v in x['metrics']['nearest_same_polarity_errors_s'] if v is not None]
        receipt=source.parent/'protected'/f'n{n:04d}'/'receipt.json'
        if not receipt.exists(): receipt=source.parent/'protected'/f'N{n}'/'receipt.json'
        found=list((source.parent/'protected').glob(f'*/receipt.json'))
        timings={json.loads(p.read_text())['n_per_10ms']:json.loads(p.read_text())['single_call_sensor_wall_s'] for p in found}
        rows.append(dict(n_per_10ms=n,dt_us=10000/n,protected_cases=len(protected),
            quality_pass=sum(x['passes_registered_pilot_budgets'] for x in protected),
            qualified_cuda_pass=sum(x['qualified_cuda_passing'] for x in protected),
            worst_state_RMSE_over_C=max(x['state_rmse_over_C'] for x in protected),
            worst_state_p99_over_C=max(x['state_p99_over_C'] for x in protected),
            worst_event_abs_error_us=max(times)*1e6 if times else None,
            pooled_event_abs_error_p95_us=float(np.percentile(times,95))*1e6 if times else None,
            lowest_case_F1=min(x['metrics']['f1'] for x in moving),
            complete_reference_success=sum(x['metrics']['complete_reference_success'] for x in moving),
            reference_cases=len(moving),sensitivity_cases=len(sensitivity),
            sensitivity_quality_pass=sum(x['passes_registered_pilot_budgets'] for x in sensitivity),
            sensitivity_cuda_qualified_pass=sum(x['qualified_cuda_passing'] for x in sensitivity),
            single_call_sensor_s=timings.get(n)))
    save_table(prefix+'_summary',rows)
    x=[r['n_per_10ms'] for r in rows]
    fig,ax=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    ax[0].plot(x,[r['worst_state_RMSE_over_C'] for r in rows],'o-',label='最差 RMSE / C')
    ax[0].plot(x,[r['worst_state_p99_over_C'] for r in rows],'s-',label='最差 p99 / C')
    ax[0].axhline(.025,color='C0',ls=':',lw=1)
    ax[0].axhline(.05,color='C1',ls=':',lw=1)
    ax[0].set(yscale='log',ylabel='亮度响应误差 / 阈值 C'); ax[0].legend(fontsize=8)
    ax[1].plot(x,[r['worst_event_abs_error_us'] for r in rows],'o-',label='最差事件时间误差')
    ax[1].axhline(20,color='C3',ls='--',lw=1,label='本次登记预算：20 微秒')
    ax[1].set(yscale='log',ylabel='绝对时间误差（微秒）'); ax[1].legend(fontsize=8)
    ax[2].plot(x,[r['qualified_cuda_pass'] for r in rows],'o-',label='同时满足全部预算')
    ax[2].plot(x,[r['complete_reference_success']+r['protected_cases']-r['reference_cases'] for r in rows],'s--',label='50 微秒内匹配全部事件')
    ax[2].set(ylabel='通过的测试项数',ylim=(-1,max(r['protected_cases'] for r in rows)+3)); ax[2].legend(fontsize=8)
    for a in ax:
        a.set(xscale='log',xlabel='每 10 毫秒采样步数')
        a.set_xticks(x);a.set_xticklabels(x,rotation=40);a.grid(alpha=.2)
    fig.suptitle(title+'（无噪声传感器诊断；预算为本项目设定）',fontsize=12)
    for ext in ('png','pdf'): fig.savefig(OUT/(prefix+'_convergence.'+ext),dpi=190)
    plt.close(fig)

analytic(ROOT/'light/analytic_full/metrics.json','step','亮度突变：解析参考')
analytic(ROOT/'light/smooth_full/metrics.json','smooth','平滑亮度变化：独立积分参考')

geo=ROOT/'light/geometry_independent_v2/geometry_quality.json'
if geo.exists():
    g=json.loads(geo.read_text())
    rows=[dict(n=a['N'],geometry_s_per_100ms=a['geometry_wall_s'],
               **{k+'_RMSE_over_C':a['strata'][k]['RMSE_over_C'] for k in ['all','active','boundary','occlusion','thin','background']})
          for a in g['aggregates']]
    save_table('geometry_summary',rows)
    fig,ax=plt.subplots(1,2,figsize=(9,3.5),layout='constrained')
    for key,label in [('all','全图'),('active','运动 / 亮度变化区'),('boundary','深度 / 物体边界')]:
        ax[0].plot([r['n'] for r in rows],[r[key+'_RMSE_over_C'] for r in rows],'o-',label=label)
    ax[0].set(ylabel='真实中间帧的亮度误差 RMSE / C',ylim=(0,None));ax[0].legend(fontsize=8)
    ax[1].plot([r['n'] for r in rows],[r['geometry_s_per_100ms'] for r in rows],'o-')
    ax[1].set(ylabel='生成 100 毫秒片段的几何耗时（秒）',ylim=(0,None))
    for a in ax:
        a.set(xlabel='每 10 毫秒采样步数');a.grid(alpha=.2)
    fig.suptitle('同一新场景片段：增加采样未消除几何误差（独立 1 毫秒渲染参考）',fontsize=11)
    for ext in ('png','pdf'):fig.savefig(OUT/('geometry_floor.'+ext),dpi=190)
    plt.close(fig)
print('FIGURES',str(OUT))
