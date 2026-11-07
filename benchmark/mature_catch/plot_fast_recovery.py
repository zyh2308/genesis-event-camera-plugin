from pathlib import Path
import json,argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msyh.ttc'));args=p.parse_args();data=json.loads(args.data.read_text(encoding='utf-8'))
font=FontProperties(fname=str(args.font));plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':11,'font.size':10,'axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(12.8,4.8),dpi=180);names=['rgbd_fast_v1','rgbd_fast_full_contour_v3_20261108','rgbd_fast_flight_fit_v3b_20261108','rgbd_fast_camera7_v4_20261108'];labels=['原机位','边缘轮廓过滤','轨迹拟合','相机上仰 7°'];colors=['#7E8892','#4A83B5','#9B78AF','#167D86'];handles=[]
for n,label,c in zip(names,labels,colors):
 rows=data['cases'][n]['paired_rows'];times=[a['time_s'] for a in rows]
 for ax,key in zip(axes,['position_error_mm','velocity_error_m_s']):
  line,=ax.plot(times,[a[key] for a in rows],label=label,color=c,linewidth=2.2,marker='o',markersize=3)
  if ax is axes[0]:handles.append(line)
for ax,ylabel in zip(axes,['三维位置误差（mm）','速度误差（m/s）']):
 ax.set_xlabel('控制时刻（s）',fontproperties=font);ax.set_ylabel(ylabel,fontproperties=font);ax.grid(axis='y',alpha=.18);ax.set_xlim(.63,1.29);ax.set_ylim(bottom=0)
 for gap in data['cases']['rgbd_fast_v1']['detection_gaps']:ax.axvspan(gap['start'],gap['end'],color='#C5CED6',alpha=.20,zorder=0)
fig.suptitle('高速来球 2.2 m/s：视野与事件状态估计',fontproperties=font,fontsize=17,y=.98)
fig.text(.5,.895,'相同初始来球、原权重与冻结事件核心；17 个共同控制时刻',fontproperties=font,ha='center',fontsize=10.5,color='#425666')
fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.875),ncol=4,frameon=False,prop=font)
fig.text(.5,.085,'阴影带：旧机位目标观测间断。各组观测由各自真实闭环产生，非同输入离线回放。',fontproperties=font,ha='center',fontsize=9.5,color='#52616E')
fig.text(.5,.035,'物理结果：前三组未留住球；相机上仰 7° 完成接取，保持至 2.52 s。',fontproperties=font,ha='center',fontsize=10,color='#167D86')
fig.subplots_adjust(top=.745,bottom=.23,left=.075,right=.98,wspace=.25);args.output.parent.mkdir(exist_ok=True);fig.savefig(args.output,facecolor='white');plt.close(fig)
print(args.output)
