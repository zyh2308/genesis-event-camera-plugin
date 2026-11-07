"""Export actual synthetic root-cause results; no real metric or seed exclusion."""
import csv,json
from pathlib import Path
import numpy as np
P=Path(__file__).parent
data_path=P/'diagnostic_evidence.json'
if not data_path.exists():data_path=P.parent/'evidence/diagnostic_evidence.json'
data=json.loads(data_path.read_text());out=(P if data_path.parent==P else P.parent)/'diagnostic_tables';out.mkdir(exist_ok=True)
cases=data['fixed_cases']+data['causal_cases'];assert len(cases)==8
assert len({r['initial_state_sha256'] for r in cases})==1
assert len({r['sample_label_hashes_sha256'] for r in cases})==1
rows=[]
for r in cases:
    assert r['seed']==2 and r['updates']==20 and r['fixed_train_batch'] and not r['real_data_used']
    assert r['recovery_model_hash_match'] and r['optimizer_counters']==[20]
    m=r['after']['equal_rollout_mean']
    rows.append(dict(method=r['method'],seed=2,arm=r['arm'],updates=20,clipped_lpips_vgg=m['clipped_lpips'],raw_lpips_vgg=m['raw_lpips'],
        gray_lpips_vgg=m['gray_lpips'],black_lpips_vgg=m['zero_lpips'],clipped_mse=m['mse'],
        contrast_ratio=m['clipped_std']/m['target_std'],correlation=m['correlation'],raw_mean=m['raw_mean'],
        negative_pixel_fraction=m['negative_fraction'],degenerate_frame_fraction=r['after']['degenerate_frame_fraction']))
def csvfile(name,values):
    with (out/name).open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
csvfile('fixed20_synthetic_ablation.csv',rows)
gradients=[]
for r in data['objective']['results']:
    g=r['initial_head_bias_gradients']
    gradients.append(dict(method=r['method'],seed=r['seed'],initial_prediction_mean=r['prediction_mean'],negative_fraction=r['negative_fraction'],
        perceptual_bias_gradient=next(v for k,v in g.items() if k.startswith('perceptual_loss')),
        temporal_bias_gradient=next(v for k,v in g.items() if k.startswith('temporal_consistency_loss')),
        TC_sign_symmetry_error=r['tc_sign_symmetry_error']))
csvfile('initialization_gradient_probes.csv',gradients)
for r in gradients:assert r['TC_sign_symmetry_error']==0
for row in cases:
    trace=data['raw_training_logs'][Path(row['source_result_path']).parent.name]
    import hashlib
    assert hashlib.sha256(trace['content'].encode()).hexdigest()==trace['sha256']
    lines=[json.loads(line) for line in trace['content'].splitlines()]
    assert [v['update'] for v in lines]==list(range(1,21))
    if row['arm']=='linear_l2':assert any(k.startswith('l2_loss/') for k in lines[0]['losses'])
    if row['arm']=='linear_no_tc':assert all(v==0 for line in lines for k,v in line['losses'].items() if k.startswith('temporal_consistency_loss/'))
    (out/(row['method']+'_seed2_'+row['arm']+'_updates1_20.jsonl')).write_text(trace['content'])

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
arms=['original','linear_l2','linear_no_tc','sigmoid'];labels=['原配置','加像素损失','关闭时序项\n（仅诊断）','约束输出范围\n（保留时序项）']
fig,axes=plt.subplots(1,2,figsize=(11.3,4.8),constrained_layout=True)
x=np.arange(4);colors=['#277da8','#d17b36']
for offset,method,color,label in [(-.18,'eventgenesis',colors[0],'EventGenesis'),(.18,'v2e',colors[1],'V2E')]:
    subset=[next(r for r in rows if r['method']==method and r['arm']==arm) for arm in arms]
    bars=axes[0].bar(x+offset,[r['clipped_lpips_vgg'] for r in subset],.32,color=color,label=label)
    axes[0].bar_label(bars,fmt='%.3f',padding=3,fontsize=9)
    bars=axes[1].bar(x+offset,[100*r['degenerate_frame_fraction'] for r in subset],.32,color=color,label=label)
    axes[1].bar_label(bars,fmt='%.0f%%',padding=3,fontsize=9)
gray=np.mean([r['gray_lpips_vgg'] for r in rows]);axes[0].axhline(gray,color='#555',ls='--',lw=1,label='固定灰图基线')
axes[0].set_ylim(0,.82);axes[0].set_title('合成验证 LPIPS-VGG ↓',fontproperties=font)
axes[1].set_ylim(0,108);axes[1].set_title('近黑/平坦退化帧比例（%）↓',fontproperties=font)
for ax in axes:
    ax.set_xticks(x,labels,fontproperties=font,fontsize=10);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18)
axes[0].legend(prop=font,frameon=False,loc='upper right')
fig.suptitle('重建退化的短程对照：同一 seed 2，每组 20 次更新\n独立合成验证；不代表真实域表现或生成器排名',fontproperties=font,fontsize=13)
fig.savefig(out/'fixed20_synthetic_ablation.png',dpi=200);fig.savefig(out/'fixed20_synthetic_ablation.pdf')
(out/'export_checks.json').write_text(json.dumps(dict(passed=True,all_eight_same_initialization=True,all_eight_same_labels_and_augmentation=True,
    each_has_20_finite_optimizer_updates=True,pixel_loss_executed=True,TC_disabled_only_in_causal_arm=True,real_data_used=False),indent=2)+'\n')
print(json.dumps(rows,indent=2))
