from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
r=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=r/'INPUT_MOTION_DIAGNOSTIC.json');p.add_argument('--output',type=Path,default=r/'delivery_local');a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True);data=json.loads(a.data.read_text());font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9.8,5.2));rates=[100,50,25];colors=['#758b9c','#3b9d97','#c17c48'];medians=[]
for i,hz in enumerate(rates):
 values=[x['early_frame_displacement_px']['median'] for x in data['records'] if x['input_hz']==hz and x['early_frame_displacement_px']['median'] is not None];median=float(np.median(values));medians.append(median);ax.bar(i,median,width=.6,color=colors[i]);ax.scatter([i+(j-(len(values)-1)/2)*.05 for j in range(len(values))],values,color='#203c50',s=25,zorder=3);ax.text(i,max(values)+.35,f'{median:.2f} px',ha='center',fontsize=15)
ax.set_xticks(range(3),['100张/秒\n间隔10 ms','50张/秒\n间隔20 ms','25张/秒\n间隔40 ms']);ax.set_ylabel('每两张输入图之间的目标位移（像素）');ax.set_ylim(0,max(medians)*1.38);ax.set_title('同一条物理轨迹｜让输入更稀疏，检验原重建路径',fontsize=16,pad=18);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
fig.text(.04,.035,'同6条归档真值轨迹、同相机路径；点为各回合的前期窗口中位数。球速、控制频率与冻结仿真器均不变。',fontsize=9);fig.tight_layout(rect=[0,.10,1,1]);fig.savefig(out/'00_稀疏输入难度.png',dpi=180);plt.close(fig)
print(json.dumps({'median_displacement_px':dict(zip(rates,medians)),'25_vs100_ratio':medians[2]/medians[0]}))
