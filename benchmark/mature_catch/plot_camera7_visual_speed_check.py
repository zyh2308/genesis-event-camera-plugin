from pathlib import Path
import argparse,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

r=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=r/'received_delivery/GENERALIZATION_PORTABLE_G3.json');parser.add_argument('--output',type=Path,default=Path('C:/Users/zhaoyuhan/Desktop/7度机位泛化验证_20261108'));args=parser.parse_args();d=args.output;d.mkdir(parents=True,exist_ok=True)
records=json.loads(args.data.read_text())['records']
lookup={x['case_id']:x for x in records if x['mode']=='event'}
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':12})
rows=[]
for i in range(1,7):
    n=lookup[f's{i:02d}_normal'];f=lookup[f's{i:02d}_fast'];ns=n['camera_fov']['image_speed_px_s']['median'];fs=f['camera_fov']['image_speed_px_s']['median']
    rows.append(dict(seed=n['seed'],normal_horizontal_speed_m_s=n['horizontal_speed_m_s'],fast_horizontal_speed_m_s=f['horizontal_speed_m_s'],normal_image_speed_median_px_s=ns,fast_image_speed_median_px_s=fs,horizontal_speed_ratio=f['horizontal_speed_m_s']/n['horizontal_speed_m_s'],image_speed_ratio=fs/ns))
fig,axes=plt.subplots(1,2,figsize=(12.7,5),gridspec_kw={'wspace':.30})
x=np.arange(6)
for j,row in enumerate(rows):axes[0].plot([j-.12,j+.12],[row['normal_image_speed_median_px_s'],row['fast_image_speed_median_px_s']],color='#b9c4ca',lw=1.4,zorder=1)
axes[0].scatter(x-.12,[z['normal_image_speed_median_px_s'] for z in rows],color='#238b83',label='普通',s=45,zorder=2)
axes[0].scatter(x+.12,[z['fast_image_speed_median_px_s'] for z in rows],color='#c9833e',label='高速',s=45,zorder=2)
axes[0].set_ylim(0,560);axes[0].set_ylabel('球心画面运动中位数（px/s）');axes[0].set_title('物理速度增加，画面运动没有同步增加');axes[0].legend(frameon=False,ncol=2)
axes[1].bar(x-.18,[z['horizontal_speed_ratio'] for z in rows],width=.34,color='#738b9c',label='水平初速：高速/普通')
axes[1].bar(x+.18,[z['image_speed_ratio'] for z in rows],width=.34,color='#238b83',label='成像运动中位数：高速/普通')
axes[1].axhline(1,color='#a44b4b',ls='--',lw=1);axes[1].set_ylim(0,1.9);axes[1].set_ylabel('相对普通组的比值');axes[1].set_title('逐seed配对核查');axes[1].legend(frameon=False,fontsize=10)
for ax in axes:ax.set_xticks(x,[f'{i:02d}' for i in range(1,7)]);ax.set_xlabel('新seed末两位');ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
fig.subplots_adjust(left=.08,right=.98,bottom=.20,top=.85)
fig.suptitle('当前“高速”条件的视觉难度核查｜固定7°机位与原策略',fontsize=16,y=.98)
fig.text(.025,.045,'相邻10ms真实端点、球心均在画面内；包含机器人自身运动。成像中位速度变化 −3.1%至+6.0%，不代表尾部速度或其他难度不变。',fontsize=10,color='#425766')
fig.savefig(d/'06_高速条件有效性核查.png',dpi=170);plt.close(fig)
with (d/'高速条件有效性核查.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'paired_seeds':len(rows),'image_speed_ratio_range':[min(z['image_speed_ratio'] for z in rows),max(z['image_speed_ratio'] for z in rows)]}))
