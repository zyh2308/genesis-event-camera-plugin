from pathlib import Path
import argparse,json
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--case',default='s01_normal');p.add_argument('--prefix',default='normal');p.add_argument('--output',default='normal_ours_vs_v2e_world_and_hand');a=p.parse_args();r=Path(__file__).resolve().parent;output=r/'delivery_local';output.mkdir(exist_ok=True)
 data=json.loads(a.data.read_text(encoding='utf-8'));records={(x['case_id'],x['mode']):x for x in data['records']};methods=['ours','v2e'];clips=[]
 annotation_path=r/'received_media'/(a.prefix+'_world_annotations.json');annotation=json.loads(annotation_path.read_text()) if annotation_path.exists() else None
 def project(point,pos,lookat,fov):
  forward=np.asarray(lookat)-pos;forward=forward/np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right=right/np.linalg.norm(right);down=np.cross(forward,right);delta=np.asarray(point)-pos;z=float(delta@forward)
  if z<=0:return None
  focal=240/np.tan(np.deg2rad(fov/2));return (320+focal*(delta@right)/z,240+focal*(delta@down)/z,focal*.04/z)
 for method in methods:
  with imageio.get_reader(r/'received_media'/(a.prefix+'_'+method+'.mp4')) as reader:clips.append([Image.fromarray(x).resize((960,360)) for x in reader])
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19);frames=[]
 for i in range(max(map(len,clips))):
  canvas=Image.new('RGB',(960,878),'white');draw=ImageDraw.Draw(canvas);draw.text((15,10),'同一来球｜Ours 与 V2E：实际任务全景＋手部近景',font=font,fill='#203c50')
  for j,(method,images) in enumerate(zip(methods,clips)):
   record=records[(a.case,method)];result=record['result'];y=76+390*j;canvas.paste(images[min(i,len(images)-1)],(0,y));color='#167d6d' if record.get('verified_catch',result['success'] and not result.get('terminated',False)) else '#ad4349';t=min((i+1)/25,result['sim_time']);ended=i>=len(images)
   if annotation:
    row=annotation['records'][method][min(i,len(annotation['records'][method])-1)];t=row['t'];hand=np.array(row['hand_position'])
    for column,(pos,lookat,fov) in enumerate([(np.array([2.6,2.4,1.7]),np.array([0.,.6,.85]),48),(hand+[.55,.55,.25],hand+[0.,.03,0.],50)]):
     q=project(row['object_position'],pos,lookat,fov)
     if q and 0<=q[0]<640 and 0<=q[1]<480:
      cx=480*column+.75*q[0];cy=y+.75*q[1];radius=max(5,.75*q[2]);draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#2487e8',width=2)
   text=method.upper()+('：完成接取' if record.get('verified_catch',result['success'] and not result.get('terminated',False)) else '：未完成接取')+f" ｜真实接触 {result.get('hand_contact_steps',0)} 个控制步"
   draw.text((15,y-31),text,font=small,fill=color);draw.rectangle((8,y+7,510,y+36),fill='white');draw.text((15,y+9),f'{t:.2f} s'+(' ｜回合已结束，此后画面定格' if ended else ' ｜模拟时间原速'),font=small,fill='#203c50')
  draw.text((15,840),'蓝圈：离线标注的真实球位置｜首帧RGB-D后使用事件和自身状态' if annotation else '首帧RGB-D后使用事件和自身状态｜原权重、估计器、冻结仿真器不变',font=small,fill='#203c50');frames.append(canvas)
 path=output/(a.output+'.mp4')
 with imageio.get_writer(path,fps=25,codec='libx264',quality=7,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
  for im in frames:writer.append_data(np.asarray(im))
 preview=[im.resize((720,659)) for im in frames[::2]];palette=preview[len(preview)//2].quantize(colors=96);indexed=[im.quantize(palette=palette) for im in preview];indexed[0].save(path.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2)
 frames[min(35,len(frames)-1)].save(output/(a.output+'_preview.png'))
 (output/(a.output+'_provenance.json')).write_text(json.dumps({'case':a.case,'source_episodes':[records[(a.case,m)]['output'] for m in methods],'scope':'archived actual Genesis states rendered with spectator cameras; no new physics; complete native recordings retained','methods':methods,'frames':len(frames),'fps':25,'end_freezes_explicit':True,'blue_circle':'offline scoring-only true ball center projection, including when occluded; never supplied to controller' if annotation else None},indent=2))
 print(path)

if __name__=='__main__':main()
