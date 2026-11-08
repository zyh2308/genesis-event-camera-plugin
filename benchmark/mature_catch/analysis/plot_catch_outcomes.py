from pathlib import Path
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=list(csv.DictReader(a.results.open(encoding='utf-8')));assert len(rows)==44
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'font.size':12});groups=[];labels=[];details=[]
for condition,name in [('normal100','100张/秒'),('normal50','50张/秒'),('normal25','25张/秒'),('backlit50','逆光50张/秒')]:
 for method in ['ours','v2e']:
  selected=[r for r in rows if r['condition']==condition and r['method']==method];assert len(selected)==4;catch=sum(r['verified_catch']=='True' for r in selected);contact=sum(int(r['actual_hand_contact_steps'])>0 for r in selected);groups.append([4-contact,contact-catch,catch]);labels.append(name+'｜'+('Ours' if method=='ours' else 'V2E'));details.append({'condition':condition,'method':method,'no_hand_contact':4-contact,'contact_but_not_retained':contact-catch,'retained':catch,'n':4})
fig,ax=plt.subplots(figsize=(11,7));values=np.array(groups);left=np.zeros(8)
for column,(name,color) in enumerate([('没有手部接触','#c3ccd2'),('接触后未留住','#d8a16b'),('最终留住球','#26978c')]):
 ax.barh(range(8),values[:,column],left=left,color=color,label=name,height=.67)
 for y,width,x in zip(range(8),values[:,column],left):
  if width:ax.text(x+width/2,y,str(width),ha='center',va='center',color='#203c50',fontsize=14)
 left+=values[:,column]
ax.set_yticks(range(8),labels);ax.invert_yaxis();ax.set_xlim(0,4);ax.set_xticks(range(5));ax.set_xlabel('回合数（每条4个全新seed）');ax.set_title('从碰到球，到最终留住球｜全部事件闭环回合',fontsize=17,pad=18);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=3,frameon=False);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(a.output/'04_接触与留球.png',dpi=180);plt.close(fig);(a.output/'CONTACT_RETENTION_SUMMARY.json').write_text(json.dumps(details,indent=2),encoding='utf-8');print('Contact/retention stages plotted from all32 actual event episodes')
