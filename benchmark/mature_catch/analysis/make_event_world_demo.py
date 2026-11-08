"""Show actual archived task states alongside only the event packets then available."""
from pathlib import Path
import argparse,json
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--annotation',type=Path,required=True);a=p.parse_args();r=Path(__file__).resolve().parent;out=r/'delivery_local';data=json.loads(a.data.read_text(encoding='utf-8'));sel=json.loads(a.selection.read_text(encoding='utf-8'));annotation=json.loads(a.annotation.read_text(encoding='utf-8'));records={(x['case_id'],x['mode']):x for x in data['records']};case=sel['case_id'];hz=records[(case,'ours')]['endpoint_hz'];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);tiny=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
def project(point,pos,target,fov):
 forward=np.asarray(target)-pos;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);d=np.asarray(point)-pos;z=float(d@forward)
 if z<=0:return None
 f=240/np.tan(np.deg2rad(fov/2));return (320+f*(d@right)/z,240+f*(d@down)/z,f*.04/z)
clips={};events={}
for method in ['ours','v2e']:
 with imageio.get_reader(r/'received_media'/(a.prefix+'_'+method+'.mp4')) as reader:clips[method]=[Image.fromarray(f).resize((960,360)) for f in reader]
 with imageio.get_reader(r/'received_media'/(a.prefix+'_'+method+'_events.mp4')) as reader:events[method]=[Image.fromarray(f) for f in reader]
 assert len(clips[method])==len(events[method])==len(annotation['records'][method])
frames=[]
for i in range(max(len(x) for x in clips.values())):
 image=Image.new('RGB',(1280,888),'white');draw=ImageDraw.Draw(image);draw.text((16,10),f'{hz}张/秒输入｜原权重事件接取：全景、手部近景、已到达事件',font=font,fill='#203c50')
 for j,method in enumerate(['ours','v2e']):
  k=min(i,len(clips[method])-1);record=records[(case,method)];row=annotation['records'][method][k];y=82+j*394;image.paste(clips[method][k],(0,y));event=events[method][k];image.paste(event,(960,y+40));color='#167d6d' if record['verified_catch'] else '#ad4349';draw.text((15,y-30),method.upper()+('：最终留住球' if record['verified_catch'] else '：未完成接取')+f"｜真实接触 {record['result'].get('hand_contact_steps',0)} 个控制步",font=small,fill=color)
  hand=np.asarray(row['hand_position'])
  for column,(pos,target,fov) in enumerate([(np.array([2.6,2.4,1.7]),np.array([0.,.6,.85]),48),(hand+[.55,.55,.25],hand+[0.,.03,0.],50)]):
   q=project(row['object_position'],pos,target,fov)
   if q and 0<=q[0]<640 and 0<=q[1]<480:
    cx=480*column+.75*q[0];cy=y+.75*q[1];radius=max(5,.75*q[2]);draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#2487e8',width=2)
  draw.rectangle((8,y+6,510,y+33),fill='white');draw.text((15,y+8),f"{row['t']:.2f}s 模拟时间"+('｜已结束，此后定格' if i>=len(clips[method]) else ''),font=small,fill='#203c50');draw.text((975,y+8),'已到达事件 / 实际控制估计',font=tiny,fill='#203c50')
 draw.text((15,858),'蓝圈：离线真实球标注；绿圈：实际估计。首帧RGB-D后仅用事件与自身状态。红/蓝：事件极性。',font=small,fill='#203c50');frames.append(image)
path=out/(a.prefix+'_demo.mp4')
with imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
 for frame in frames:writer.append_data(np.asarray(frame))
preview=[frame.resize((960,666)) for frame in frames[::2]];palette=preview[len(preview)//2].quantize(colors=128);indexed=[x.quantize(palette=palette) for x in preview];indexed[0].save(path.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2);frames[min(35,len(frames)-1)].save(out/(a.prefix+'_preview.png'));(out/(a.prefix+'_provenance.json')).write_text(json.dumps({'case':case,'input_hz':hz,'sources':{m:records[(case,m)]['output'] for m in ['ours','v2e']},'selection_rule':sel.get('selection'),'event_availability':'actual packet available_s; no future packets','scene_alignment':'event input before command; scene state after that command, not claimed simultaneous','blue_marker':'offline true position; not controller input','frozen_algorithms':True},indent=2));print(path)
