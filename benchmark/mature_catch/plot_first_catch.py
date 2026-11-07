"""Plot observed task outcomes; a development case, not success-rate statistics."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True, help='Portable plot_data.json')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--font', default='C:/Windows/Fonts/msyh.ttc')
args = parser.parse_args()
data = json.loads(args.data.read_text(encoding='utf-8'))
font = FontProperties(fname=args.font)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={'width_ratios': [1.7, 1]})
colors = ['#167d86', '#9a6970']
names = ['hybrid_normal_v9', 'initial_frame_only_normal_v9']
labels = ['初始图像＋事件更新', '仅初始图像，随后不更新']
for name, label, color in zip(names, labels, colors):
    rows = data[name]['trace_plot']
    axes[0].plot([x['t'] for x in rows], [x['distance_m'] for x in rows],
                 color=color, linewidth=2.3, label=label)
axes[0].axvspan(1.40, 2.52, color=colors[0], alpha=.10)
axes[0].annotate('真实手部接触持续至任务结束', xy=(1.95, .13), xytext=(1.23, .68),
                 fontproperties=font, fontsize=10, color=colors[0],
                 arrowprops=dict(arrowstyle='->', color=colors[0]))
axes[0].set(xlim=(0, 2.6), ylim=(0, 2.5))
axes[0].set_xlabel('模拟时间（秒）', fontproperties=font)
axes[0].set_ylabel('手与球的真实距离（米）', fontproperties=font)
axes[0].set_title('同一来球，固定原接取权重', fontproperties=font, fontsize=12)
axes[0].legend(prop=font, frameon=False, loc='upper right')
axes[0].grid(alpha=.18)
for i, (name, label, color) in enumerate(zip(names, labels, colors)):
    rows = data[name]['trace_plot']
    axes[1].plot([x['t'] for x in rows], [1-i]*len(rows), color='#d9dedf', linewidth=9,
                 solid_capstyle='butt')
    contact = [x['t'] for x in rows if x['hand_contact']]
    if contact:
        axes[1].plot(contact, [1-i]*len(contact), color=color, linewidth=9,
                     solid_capstyle='butt')
    outcome = data[name]['result']
    axes[1].scatter([outcome['sim_time']], [1-i], color=color,
                    marker='o' if outcome['success'] else 'x', s=85, zorder=3)
    axes[1].text(.02, 1-i+.17, label, fontproperties=font, fontsize=10)
    axes[1].text(.02, 1-i-.25, '接住，保持约 1.12 秒' if outcome['success'] else '无接触，提前结束',
                 fontproperties=font, fontsize=10, color=color)
axes[1].set(xlim=(0, 2.6), ylim=(-.5, 1.5), yticks=[])
axes[1].set_xlabel('模拟时间（秒）；彩色粗线为手部接触', fontproperties=font, fontsize=9)
axes[1].set_title('接触与任务结束记录', fontproperties=font, fontsize=12)
for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('Genesis 事件跟踪接取：首个完整成功回合', fontproperties=font, fontsize=16, y=.99)
fig.text(.5, .015, '已知开发来球，水平速度 1.7 m/s；单回合诊断，不能解释为总体成功率或 RGB 视频基线比较。',
         ha='center', fontproperties=font, fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .06, 1, .94))
dest = args.output
dest.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(dest, dpi=180, facecolor='white')
print(dest)
