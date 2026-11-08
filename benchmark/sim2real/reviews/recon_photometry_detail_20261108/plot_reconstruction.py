"""Plot the completed single-seed synthetic reconstruction loss comparison.

Run: python3 plot_reconstruction.py
Reads immutable analysis CSV/JSON only; starts no training or scoring.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT/'results_analysis'
OUT = ANALYSIS/'figures'


def main():
    fonts = [Path('/mnt/c/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/msyh.ttc')]
    font = next((p for p in fonts if p.exists()), None)
    if font:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10.5,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#505050', 'axes.linewidth': .8,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    source = ANALYSIS/'all200_primary.csv'
    gates_path = ANALYSIS/'adoption_gates.json'
    gates = json.loads(gates_path.read_text())
    with source.open(newline='', encoding='utf-8-sig') as stream:
        rows = {(r['method'], r['arm']): r for r in csv.DictReader(stream)}
    arms = ['control', 'photometric', 'detail']
    methods = [('eventgenesis', 'EventGenesis'), ('v2e', 'V2E')]
    for method, _ in methods:
        for arm in arms:
            row = rows[method, arm]
            assert (int(row['seed']), int(row['optimizer_update']), int(row['rollouts']), int(row['frames_per_rollout'])) == (0, 5100, 8, 200)
    assert gates['training_evidence_passed'] and gates['pairing_passed'] and gates['no_real_scoring']
    assert not gates['any_candidate_passed_both']
    assert all(gates['methods']['eventgenesis'][arm]['passed'] for arm in arms[1:])
    assert all(not gates['methods']['v2e'][arm]['passed'] for arm in arms[1:])
    for arm in arms[1:]:
        failed = {x['metric'] for x in gates['methods']['v2e'][arm]['checks'] if not x['passed']}
        assert failed == {'lpips_vgg', 'edge_mae'}

    metrics = [('mse', '光度误差 MSE（越低越好）', 4),
               ('lpips_vgg', '感知距离 LPIPS-VGG（越低越好）', 4),
               ('edge_mae', '逐像素梯度 MAE（越低越好）', 5)]
    colors = ['#9DA3AA', '#5D86B0', '#277E80']
    arm_labels = ['同期对照', '+ L1', '+ L1 + 边缘']
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.5))
    fig.subplots_adjust(left=.065, right=.98, top=.79, bottom=.19, wspace=.33, hspace=.40)
    for ri, (method, title) in enumerate(methods):
        for ci, (metric, label, decimals) in enumerate(metrics):
            ax = axes[ri, ci]
            values = [float(rows[method, arm][metric]) for arm in arms]
            ax.bar(range(3), values, color=colors, width=.60, edgecolor='white', linewidth=.7)
            for i, value in enumerate(values):
                ax.annotate(f'{value:.{decimals}f}', (i, value), xytext=(0, 5),
                            textcoords='offset points', ha='center', va='bottom', fontsize=10)
            ax.set_title(title, loc='left', pad=11, fontsize=12)
            ax.set_ylabel(label, fontsize=9.5)
            ax.set_xticks(range(3), arm_labels)
            ax.tick_params(axis='x', labelsize=9.5)
            ax.set_axisbelow(True)
            ax.grid(axis='y', alpha=.18, linewidth=.7)
            ax.set_ylim(0, max(values)*1.23)
    # Identical metric scales across generators; retain zero-origin axes.
    for ci in range(3):
        limit = max(axes[0, ci].get_ylim()[1], axes[1, ci].get_ylim()[1])
        for ri in range(2):axes[ri, ci].set_ylim(0, limit)
    fig.suptitle('重建：光度改善，感知与细节尚未同时保住', fontsize=16, y=.97)
    fig.text(.065, .914, '单个 seed 0；三分支均从各自共同检查点 4800 续训 300 步，比较相同总步数 5100', fontsize=10.5, color='#555555')
    fig.text(.065, .872, '8 条独立合成 rollout，每条完整 200 帧并等权平均；原 LPIPS + TC 保留，不混入源检查点作为主要对照', fontsize=10.5, color='#555555')
    fig.text(.065, .112, 'EventGenesis：两个候选均通过 9/9 项主门槛；V2E：光度改善，但 LPIPS-VGG 与梯度 MAE 退步。', fontsize=10, color='#444444')
    fig.text(.065, .072, '两个候选均未通过“两生成器同时满足”的预登记门槛：本轮不采用，不启动长训。', fontsize=10, color='#444444')
    fig.text(.065, .032, '这是单 seed 的合成开发诊断，不是多 seed 论文最终结果；没有本轮真实评分，不作显著性或真实泛化结论。', fontsize=9.5, color='#555555')
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT/'reconstruction_photometry_detail.png', dpi=190, facecolor='white')
    fig.savefig(OUT/'reconstruction_photometry_detail.pdf', facecolor='white')
    plt.close(fig)
    manifest = {'source_csv': source.name, 'source_csv_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'gates_sha256': hashlib.sha256(gates_path.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'font': str(font), 'plot_contract': 'single seed0, all200 primary; 8 equal-weight rollouts; 3 arms at5100; no before mixed',
                'outputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*') if p.suffix in ('.png', '.pdf')}}
    (OUT/'reconstruction_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':main()
