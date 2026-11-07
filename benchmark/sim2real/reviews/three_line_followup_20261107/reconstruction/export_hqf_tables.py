"""Export every frozen HQF run, seed variance and the pre-specified EG/V2E contrast."""
import csv,hashlib,json,statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).parent
OUT=ROOT/'hqf_tables';OUT.mkdir(exist_ok=True)
d=json.loads((ROOT/'hqf_summary.json').read_text())
METHODS=['eventgenesis','v2e','vid2e','film_esim','rife_esim']
LABELS=['EventGenesis','V2E','Vid2E','FILM+ESIM','RIFE+ESIM']
assert d['real_scores_computed'] and d['checkpoint_selection_allowed'] is False and d['manager_completion']['exit_code']==0
assert len(d['per_arm'])==15
sequences=sorted(next(iter(d['per_arm'].values()))['per_sequence'])
assert len(sequences)==14
def values(v):
    assert not v['psnr']['mean_is_positive_infinity']
    return dict(lpips_alex=v['lpips_alex_mean'],ssim=v['ssim_mean'],psnr_db=v['psnr']['mean_db'],mse=v['mse_mean'])
def csvfile(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
rows=[];seed_rows=[];summary=[]
for method in METHODS:
    for seed in range(3):
        arm=d['per_arm'][f'{method}_seed{seed}']
        assert arm['sequence_count']==14 and arm['total_scored_frames']==15498 and sorted(arm['per_sequence'])==sequences
        result=values(arm['equal_sequence_mean']);seed_rows.append(dict(method=method,seed=seed,**result))
        for seq,s in arm['per_sequence'].items():
            rows.append(dict(method=method,seed=seed,sequence=seq,scored_frames=s['scored_frame_count'],**values(s['metrics'])))
        for metric,mean in result.items():
            actual=statistics.mean(r[metric] for r in rows if r['method']==method and r['seed']==seed)
            assert abs(actual-mean)<1e-12,'Source sequence aggregation mismatch'
    r=dict(method=method,n_seeds=3,n_sequences=14,frames_per_seed=15498)
    for metric in ['lpips_alex','ssim','psnr_db','mse']:
        v=[a[metric] for a in seed_rows if a['method']==method]
        r[metric+'_mean']=statistics.mean(v);r[metric+'_sample_sd']=statistics.stdev(v)
    summary.append(r)
csvfile('hqf14_all_seeds_per_sequence.csv',rows)
csvfile('hqf14_per_seed.csv',seed_rows)
csvfile('hqf14_mean_sample_sd.csv',summary)
# Shared seed and sequence draws preserve the paired comparison. Three seeds
# provide limited inference: report uncertainty, not a p-value or a win claim.
delta=np.array([[values(d['per_arm'][f'eventgenesis_seed{s}']['per_sequence'][q]['metrics'])['lpips_alex']-
                  values(d['per_arm'][f'v2e_seed{s}']['per_sequence'][q]['metrics'])['lpips_alex']
                for q in sequences] for s in range(3)])
rng=np.random.default_rng(20261107)
boot=np.array([delta[np.ix_(rng.integers(0,3,3),rng.integers(0,14,14))].mean() for _ in range(20000)])
contrast=dict(metric='LPIPS-Alex',contrast='EventGenesis minus V2E; lower is better',estimate=float(delta.mean()),
    percentile_95_interval=np.quantile(boot,[.025,.975]).tolist(),draws=20000,seed=20261107,
    method='paired crossed bootstrap: sample seeds and sequence identities independently, share draws across methods',
    limitation='Only three seeds; all seed2 runs look degenerate and require synthetic diagnosis. No exclusion, selection or superiority claim.')
(OUT/'primary_lpips_paired_uncertainty.json').write_text(json.dumps(contrast,indent=2)+'\n')
lines=['# HQF14：首轮完整真实重建评分','',
    '本表保留5种生成方式、全部3个seed、每个seed全部14条序列；每组15,498帧，共232,470帧次。先在序列内按帧平均，再对14条序列等权平均，最后对三个seed计算均值与样本标准差。权重只按合成验证选定；没有据真实评分替换权重。','',
    '**这是完成运行的首轮原始结果，尚未通过输出稳定性核验。** 五种方法的seed2均出现近似相同的差分数，疑似输出范围/模型退化，需要在合成验证中诊断。全部seed保留，不能只报seed0/1，不能据此证明EventGenesis优于V2E。','',
    '| 训练事件来源 | LPIPS-Alex ↓ | SSIM ↑ | PSNR/dB ↑ | MSE ↓ |','|---|---:|---:|---:|---:|']
for label,r in zip(LABELS,summary,strict=True):
    row=[f"{r[k+'_mean']:.4f} ± {r[k+'_sample_sd']:.4f}" for k in ['lpips_alex','ssim','psnr_db','mse']]
    lines.append('| '+label+' | '+' | '.join(row)+' |')
lines+=['',f"预先指定的主指标对比：EventGenesis − V2E 的 LPIPS 差值为 {contrast['estimate']:.4f}；正值表示EventGenesis较差。配对跨seed/序列bootstrap区间为 [{contrast['percentile_95_interval'][0]:.4f}, {contrast['percentile_95_interval'][1]:.4f}]。只有3个seed，区间仅作不确定性描述。",'',
    '所有逐seed和逐序列数值在CSV中。图中同时显示每个seed，误差条为样本标准差，不是置信区间。LPIPS-Alex为主指标，SSIM/PSNR/MSE为辅助描述，不做事后择指标的优势宣称。','',
    '冻结原始记录路径与SHA见`hqf_summary.json`的`server_files`。真实木块E-POSE与本表无关；本次没有更改任何真实测试协议或仿真器核心。']
(OUT/'README.md').write_text('\n'.join(lines)+'\n')
latex=['% Raw first full run; all seeds retained; stability audit pending.',
       r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & LPIPS$\downarrow$ & SSIM$\uparrow$ & PSNR$\uparrow$ & MSE$\downarrow$ \\',r'\midrule']
for label,r in zip(LABELS,summary,strict=True):
    cells=[f"${r[k+'_mean']:.4f}\\pm {r[k+'_sample_sd']:.4f}$" for k in ['lpips_alex','ssim','psnr_db','mse']]
    latex.append(label+' & '+' & '.join(cells)+r' \\')
latex += [r'\bottomrule',r'\end{tabular}']
(OUT/'hqf14_table.tex').write_text('\n'.join(latex)+'\n')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
fontpath=Path('/mnt/c/Windows/Fonts/msyh.ttc')
font=FontProperties(fname=str(fontpath)) if fontpath.exists() else FontProperties()
fig,axes=plt.subplots(2,2,figsize=(11.4,7.1),constrained_layout=True)
colors=['#277DA8','#3A9679','#C66A35']
for ax,key,label in zip(axes.flat,['lpips_alex','ssim','psnr_db','mse'],['LPIPS-Alex ↓','SSIM ↑','PSNR / dB ↑','MSE ↓'],strict=True):
    x=np.arange(5)
    means=[r[key+'_mean'] for r in summary];sd=[r[key+'_sample_sd'] for r in summary]
    ax.errorbar(x,means,yerr=sd,fmt='s',color='#273746',capsize=4,markersize=5,label='均值 ± 标准差',zorder=2)
    for seed,color in enumerate(colors):
        ys=[next(r[key] for r in seed_rows if r['method']==m and r['seed']==seed) for m in METHODS]
        ax.scatter(x+(seed-1)*.075,ys,s=33,color=color,label='seed '+str(seed),zorder=3)
    ax.set_xticks(x,LABELS,rotation=15,ha='right');ax.set_title(label,fontproperties=font,fontsize=12)
    ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=4,prop=font,frameon=False)
fig.suptitle('真实 HQF 重建：完整三 seed 初轮结果\n全部 seed 保留；seed 2 退化待核查',fontproperties=font,fontsize=14)
fig.savefig(OUT/'hqf14_all_seed_metrics.png',dpi=200)
fig.savefig(OUT/'hqf14_all_seed_metrics.pdf')
print(json.dumps(dict(status='exported_all_15_runs',primary_contrast=contrast,summary=summary),indent=2))
