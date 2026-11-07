"""Export all14 sequence results; historical comparisons retain budget differences."""
import csv,datetime,hashlib,json,pathlib,statistics,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
p=pathlib.Path(__file__).parent
repo=pathlib.Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105')
out=repo/'benchmark/sim2real/reviews/hqf_repair_pair_20261108';out.mkdir(exist_ok=False)
for f in ['sources','evidence','tables']: (out/f).mkdir()
d=json.loads((p/'hqf_result_summary.json').read_text());new=d['score'];old=json.loads((repo/'benchmark/sim2real/reviews/three_line_followup_20261107/reconstruction/hqf_summary.json').read_text())
assert d['completion']['status']=='complete' and d['completion']['exit_code']==0 and d['exit']['status']=='released_complete'
assert new['real_scores_computed'] and new['checkpoint_selection_allowed'] is False
assert all(r['checkpoint_updates']==4800 for r in d['protocol']['runs']) and len(d['protocol']['runs'])==2
assert set(new['per_arm'])=={'eventgenesis_seed0','v2e_seed0'}
for name,digest in d['source_manifest'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest,name
for name,rec in d['diagnostic_sources'].items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==rec['sha256'],name
(out/'evidence/hqf_result_summary.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
for name in ['build_pair_evaluation.py','hqf_repair_pair_bridge.py','hqf_repair_runtime_preflight.py','run_hqf_pair_reserved.py','evaluation_support.py','bridge_preservation_checks.json','checkpoint_binding_preflight.json','source_manifest.json','audit_prediction_range.py','finish_hqf_evaluation.py','export_hqf_repair_report.py']:
    shutil.copyfile(p/name,out/'sources'/name)
def values(v):
    assert not v['psnr']['mean_is_positive_infinity']
    return dict(lpips_alex=v['lpips_alex_mean'],ssim=v['ssim_mean'],psnr_db=v['psnr']['mean_db'],mse=v['mse_mean'])
rows=[];per=[];summary={}
for method in ['eventgenesis','v2e']:
    rid=method+'_seed0';n=new['per_arm'][rid];o=old['per_arm'][rid]
    assert old['checkpoint_bindings'][rid]['sha256']==d['historical_selection'][method]['sha256']
    assert n['sequence_count']==14 and n['total_scored_frames']==15498 and set(n['per_sequence'])==set(o['per_sequence'])
    nv=values(n['equal_sequence_mean']);ov=values(o['equal_sequence_mean']);delta={k:nv[k]-ov[k] for k in nv}
    summary[method]=dict(old=ov,new=nv,delta=delta,old_checkpoint_updates=d['historical_selection'][method]['optimizer_update'],new_checkpoint_updates=4800,
        lpips_improved_sequences=sum(n['per_sequence'][s]['metrics']['lpips_alex_mean']<o['per_sequence'][s]['metrics']['lpips_alex_mean'] for s in n['per_sequence']))
    for version,v,updates in [('previous_linear',ov,summary[method]['old_checkpoint_updates']),('repaired_sigmoid',nv,4800)]:
        rows.append(dict(method=method,seed=0,version=version,checkpoint_updates=updates,sequences=14,scored_frames=15498,**v))
    for seq in n['per_sequence']:
        na=n['per_sequence'][seq];oa=o['per_sequence'][seq];assert na['scored_frame_count']==oa['scored_frame_count']
        for version,source in [('previous_linear',oa),('repaired_sigmoid',na)]:
            per.append(dict(method=method,seed=0,version=version,sequence=seq,scored_frames=source['scored_frame_count'],**values(source['metrics'])))
    for k in nv:assert abs(statistics.mean(values(x['metrics'])[k] for x in n['per_sequence'].values())-nv[k])<1e-12
def csvfile(name,rs):
    with (out/'tables'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),lineterminator='\n');w.writeheader();w.writerows(rs)
csvfile('hqf14_seed0_repair_comparison.csv',rows);csvfile('hqf14_all_sequence_comparison.csv',per)
(out/'tables/comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
fig,axes=plt.subplots(2,2,figsize=(9,6.8),constrained_layout=True)
for ax,key,title in zip(axes.flat,['lpips_alex','ssim','psnr_db','mse'],['LPIPS-Alex ↓','SSIM ↑','PSNR / dB ↑','MSE ↓']):
    x=np.arange(2);a=[summary[m]['old'][key] for m in summary];b=[summary[m]['new'][key] for m in summary]
    ax.bar(x-.17,a,width=.32,color='#A9B4BF',label='修复前');ax.bar(x+.17,b,width=.32,color='#258A9A',label='修复后')
    precision=5 if key=='mse' else (3 if key=='psnr_db' else 4)
    for xx,v in zip(np.r_[x-.17,x+.17],a+b):ax.text(xx,v,f'{v:.{precision}f}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(x,['EventGenesis','V2E']);ax.set_title(title);ax.set_ylim(0,max(a+b)*1.22 if key!='ssim' else 1)
    ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('真实 HQF：修复前后参考（seed 0）\n修复后均为4,800步；旧权重为12,000 / 11,400步',fontproperties=font,fontsize=12)
axes[0,0].legend(prop=font,frameon=False)
fig.savefig(out/'tables/hqf14_repair_comparison.png',dpi=180);fig.savefig(out/'tables/hqf14_repair_comparison.pdf')
stamp=datetime.datetime.fromisoformat(d['completion']['finished_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M JST')
lines=['# 重建修复后的真实HQF等步数配对开发测试','',f'完成时间：{stamp}。本轮没有训练更新；光流和位姿作业未改动。','',
    '两组均使用既存的seed0、第80轮/4,800步synthetic_best.pt，权重在接触本轮HQF结果前固定。每组全部14条序列、15,498帧，总计28次序列推理和30,996次帧评分，原事件窗口、anchor排除、循环状态、native裁剪和float32指标协议保持不变。','',
    '| 方法 | 配置/检查点步数 | LPIPS-Alex ↓ | SSIM ↑ | PSNR/dB ↑ | MSE ↓ |','|---|---|---:|---:|---:|---:|']
for method,label in [('eventgenesis','EventGenesis'),('v2e','V2E')]:
    s=summary[method]
    for version,display,updates in [('old','修复前',s['old_checkpoint_updates']),('new','修复后',4800)]:
        v=s[version];lines.append(f"| {label} | {display} / {updates:,} | {v['lpips_alex']:.5f} | {v['ssim']:.5f} | {v['psnr_db']:.3f} | {v['mse']:.5f} |")
lines+=['','**本轮结论：两个当前检查点未出现近黑或平坦输出，但两组真实HQF的LPIPS-Alex均未超过各自旧seed0结果。稳定性改善不等于真实重建质量改善。** 旧权重训练更久，现阶段不能把差距单独归因于sigmoid约束或训练不足。','',
    '## 怎样理解前后变化','']
for method,label in [('eventgenesis','EventGenesis'),('v2e','V2E')]:
    s=summary[method];delta=s['delta'];pct=100*(s['old']['lpips_alex']-s['new']['lpips_alex'])/s['old']['lpips_alex']
    direction='降低' if pct>=0 else '升高'
    lines.append(f"- {label}：LPIPS变化{delta['lpips_alex']:+.5f}（{direction}{abs(pct):.1f}%），SSIM变化{delta['ssim']:+.5f}，PSNR变化{delta['psnr_db']:+.3f}dB，MSE变化{delta['mse']:+.5f}；LPIPS在{s['lpips_improved_sequences']}/14条序列改善。")
lines+=['','**前后表是历史参考，不是严格等预算修复消融。** 修复前权重分别为12,000和11,400步，修复后均为4,800步，训练配方的输出激活也不同；不能把全部差值归因于单个机制。当前两组修复后检查点之间预算相同，但只有一个seed，不构成多seed生成器优劣结论。','',
    '本次权重来自seed0，不直接证明seed2在真实域已恢复；seed2修复证据目前仍来自此前合成短训。HQF此前已查看，本轮明确属于既见数据开发测试，不能称盲测，也没有使用分数换权重、挑序列、修改亮度后处理或启动新训练。','',
    '指标先在各序列内按帧平均，再对14条序列等权平均。LPIPS-Alex使用原锁定PerceptualSimilarity v0.1实现与本地AlexNet权重；SSIM/PSNR/MSE计算函数及score_pair循环均未修改。没有GT亮度拟合、直方图匹配或图像输入网络。','',
    '[逐序列CSV](tables/hqf14_all_sequence_comparison.csv) · [汇总CSV](tables/hqf14_seed0_repair_comparison.csv) · [完整轻量证据](evidence/hqf_result_summary.json)','',
    '![修复前后指标](tables/hqf14_repair_comparison.png)','',
    '服务器保留完整float预测、原始逐帧分数、日志、协议和SHA；仓库只纳入代码、轻量证据与表格。原15组完整结果和本轮预检失败记录均未覆盖。','']
if d.get('prediction_range_audit'):
    lines+=['## 真实输出退化核对','']
    for rid,r in d['prediction_range_audit']['per_arm'].items():
        v=r['equal_sequence_mean'];lines.append(f"- {rid}：全部{r['frames']:,}帧范围及有限性通过；等序列近黑比例{v['near_black_frame_fraction']:.2%}、低对比度平坦帧比例{v['flat_frame_fraction']:.2%}，平均帧内标准差{v['mean_spatial_std']:.5f}。")
    lines+=['','“平坦帧”阈值沿用合成诊断的帧内标准差<0.001；近黑要求同时平均亮度<0.01。只作结果诊断，没有把真实标签用于训练或选点。','']
(out/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
status=repo/'benchmark/sim2real/sim2real_status.md';s=status.read_text();s='<!-- hqf-repair-20261108 -->\n> '+stamp+'：两组seed0、第4,800步HQF真实开发测试已完成，未新增训练；全部14序列、每组15,498帧，GPU预约已释放。前后结果保留旧权重预算差异，不作为严格修复因果量或多seed排名。详见[真实测试结果](reviews/hqf_repair_pair_20261108/README.md)。\n\n'+s;status.write_text(s)
index={str(f.relative_to(out)):dict(size_bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in sorted(out.rglob('*')) if f.is_file()}
(out/'artifact_index.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=index),indent=2)+'\n')
print(json.dumps(dict(report=str(out/'README.md'),summary=summary),indent=2))
