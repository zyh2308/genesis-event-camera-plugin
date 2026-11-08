"""Reproduce two synthetic-only progress figures from verified context_analysis.json.

Run: python3 plot_context.py
No remote access, inference, training, or source-data mutation.
"""
from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'context_figures'
SOURCE = ROOT / 'context_analysis.json'


def setup_style():
    candidates = [Path('/mnt/c/Windows/Fonts/msyh.ttc'),
                  Path('C:/Windows/Fonts/msyh.ttc')]
    font = next((p for p in candidates if p.exists()), None)
    if font is not None:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        'axes.unicode_minus': False, 'font.size': 11,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#505050', 'axes.linewidth': 0.8,
        'xtick.color': '#333333', 'ytick.color': '#333333',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white',
    })
    return str(font) if font else 'matplotlib default'


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), dpi=190)
    fig.savefig(OUT / (name + '.pdf'))
    plt.close(fig)


def flow_figure(data):
    methods = ['eventgenesis', 'v2e', 'vid2e', 'film_esim', 'rife_esim']
    labels = ['EventGenesis', 'V2E', 'Vid2E', 'FILM + ESIM', 'RIFE + ESIM']
    stats = [data['flow']['methods'][m]['best_epe_px_per_30ms'] for m in methods]
    means = np.array([x['mean'] for x in stats])
    sd = np.array([x['sample_sd'] for x in stats])
    assert all(x['n'] == 3 for x in stats)
    zero = data['flow']['methods']['eventgenesis']['zero_flow_epe_px_per_30ms']['mean']
    fig, ax = plt.subplots(figsize=(10.2, 6.2))
    fig.subplots_adjust(left=.105, right=.97, top=.85, bottom=.23)
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=sd, width=.58, color=['#277E80', '#DE8E50', '#819BB5', '#A096B5', '#929BA2'],
           edgecolor='white', linewidth=.8, capsize=5,
           error_kw={'elinewidth': 1.2, 'capthick': 1.2, 'ecolor': '#303030'})
    for i, stat in enumerate(stats):
        ax.scatter(i + np.array([-.095, 0., .095]), stat['values'], s=21,
                   facecolor='white', edgecolor='#333333', linewidth=.8, zorder=4)
        ax.text(i, means[i] + sd[i] + .012, f'{means[i]:.4f}',
                ha='center', va='bottom', fontsize=11)
    ax.axhline(zero, linestyle=(0, (5, 3)), color='#555555', linewidth=1.2)
    ax.text(-.42, zero+.012, f'零光流参考：{zero:.4f}', color='#444444', fontsize=11)
    ax.set_xticks(x, labels)
    ax.set_ylabel('合成验证 EPE（像素 / 30 ms，越低越好）')
    ax.set_ylim(0, .55)
    ax.set_axisbelow(True)
    ax.grid(axis='y', alpha=.18, linewidth=.7)
    fig.suptitle('光流：五种生成器的配对训练已完成', fontsize=16, y=.96)
    fig.text(.105, .885, '每组 45 epoch / 8,910 次更新；检查点按合成验证预定规则选取', fontsize=11, color='#555555')
    fig.text(.105, .105, '柱：3 个种子的均值；误差条：样本标准差；圆点：单个种子。', fontsize=10, color='#555555')
    fig.text(.105, .06, '这是合成验证结果，不是 MVSEC 真实测试；不作统计显著性或真实泛化结论。', fontsize=10, color='#555555')
    save(fig, 'flow_synthetic_best_3seed')


def pose_figure(data):
    methods = [('eventgenesis', 'EventGenesis'), ('v2e', 'V2E')]
    conditions = ['clean', 'half', 'eighth']
    condition_labels = ['正常光', '1/2 光照', '1/8 光照']
    arms = [('clean_control', '原光照对照', '#9DA3AA'),
            ('physical_supervised', '物理光照覆盖', '#5D86B0'),
            ('paired_consistency', '覆盖 + 配对一致性', '#277E80')]
    metrics = [('depth_mae_m', '深度平均绝对误差（cm，越低越好）'),
               ('success', '完整位姿成功率（%，越高越好）')]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.2))
    fig.subplots_adjust(left=.085, right=.975, top=.795, bottom=.17, wspace=.22, hspace=.40)
    x = np.arange(3)
    for row, (method, title) in enumerate(methods):
        for col, (metric, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            ymax = 0.
            for ai, (arm, label, color) in enumerate(arms):
                stats = [data['pose']['methods'][method][arm][c][metric] for c in conditions]
                assert all(s['n'] == 3 for s in stats)
                means = np.array([s['mean'] for s in stats]) * 100.
                sd = np.array([s['sample_sd'] for s in stats]) * 100.
                positions = x + (ai - 1) * .25
                ax.bar(positions, means, width=.23, yerr=sd, color=color,
                       edgecolor='white', linewidth=.5, capsize=3,
                       error_kw={'elinewidth': .95, 'capthick': .95}, label=label)
                ymax = max(ymax, float(np.max(means + sd)))
                for xp, mean, err in zip(positions, means, sd):
                    ax.annotate(f'{mean:.1f}', (xp, mean+err), xytext=(0, 4),
                                textcoords='offset points', ha='center', va='bottom', fontsize=8.8)
            # Full zero origin; common success-rate scale across both generators.
            limit = math.ceil(ymax * 1.22 / 5.) * 5.
            ax.set_ylim(0, max(5., limit))
            ax.set_xticks(x, condition_labels)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, loc='left', pad=12, fontsize=13)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.18, linewidth=.7)
    # Identical metric scales across the two generator rows allow fair reading.
    for col in range(2):
        limit = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
        for row in range(2):axes[row, col].set_ylim(0, limit)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .888),
               ncol=3, frameon=False, fontsize=11)
    fig.suptitle('位姿：暗光有所改善，其他光照仍存在退步', fontsize=16, y=.976)
    fig.text(.085, .924, '同一条件内比较三分支；更新预算相同，均为 3 个种子的均值 ± 样本标准差', fontsize=11, color='#555555')
    fig.text(.085, .095, '1/2 与 1/8 为物理辐亮度倍数，来自 4 条重光照验证轨迹；不是标定照度（lux）。', fontsize=9.5, color='#555555')
    fig.text(.085, .06, '正常光为原 96 条轨迹，每条 4 个候选窗并应用可见性规则；各条件轨迹不同，不能直接解释为纯光照因果。', fontsize=9.5, color='#555555')
    fig.text(.085, .025, '门槛仅通过 14/30 项，未做本轮真实 E-POSE 评分；图为合成开发诊断，误差条不表示统计显著性。', fontsize=9.5, color='#555555')
    save(fig, 'pose_synthetic_light_3seed')


def main():
    OUT.mkdir(exist_ok=True)
    data = json.loads(SOURCE.read_text())
    font = setup_style()
    flow_figure(data)
    pose_figure(data)
    manifest = {'source': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'font': font, 'snapshot_utc': data['snapshot_utc'],
                'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.suffix in ('.png', '.pdf')}}
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':main()
