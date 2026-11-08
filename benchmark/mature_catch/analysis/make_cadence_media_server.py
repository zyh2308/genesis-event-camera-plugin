"""All actual native Genesis outcomes, with explicitly shared baseline panels."""
import argparse,json
from pathlib import Path
import imageio.v2 as imageio,numpy as np
from PIL import Image,ImageDraw,ImageFont

METHODS=['ours','v2e','init_only','state_reference'];LABELS={'ours':'Ours','v2e':'V2E','init_only':'No later events','state_reference':'State reference'}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12);large=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)

def load(record):
 p=Path(record['output']);video=p/'episode_000/genesis.mp4'
 if video.exists():
  with imageio.get_reader(video) as reader:images=[Image.fromarray(x).resize((320,240)) for x in reader]
 else:images=[Image.open(p/'migration_audit/sensor_episode_000/initial_rgb.png').convert('RGB').resize((320,240))]
 trace=json.loads((p/'episode_000/trace.json').read_text()) if (p/'episode_000/trace.json').exists() else []
 return images,trace,str(video)

def mark_ball(draw,position,x,y):
 pos=np.array([3.1,3.2,2.]);target=np.array([0.,.8,.9]);forward=target-pos;forward/=np.linalg.norm(forward);right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);delta=np.asarray(position)-pos;z=float(delta@forward)
 if z<=0:return
 focal=240/np.tan(np.deg2rad(24));u=320+focal*(delta@right)/z;v=240+focal*(delta@down)/z
 if 0<=u<640 and 0<=v<480:
  cx=x+.5*u;cy=y+.5*v;radius=max(3,.5*focal*.04/z);draw.ellipse((cx-radius,cy-radius,cx+radius,cy+radius),outline='#2487e8',width=2)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();spec=json.loads((a.root/'EVALUATION_PROTOCOL.json').read_text());data=json.loads((a.root/'PAIRED_PORTABLE.json').read_text());assert data['complete'] and not data['violations'];assert len(data['records'])==44
 lookup={(x['case_id'],x['mode']):x for x in data['records']};out=a.root/'delivery';out.mkdir(exist_ok=True);manifest=[];master=out/'all44_cadence_comparisons.mp4'
 with imageio.get_writer(master,fps=25,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as whole:
  for case in spec['cases']:
   row=[]
   for method in METHODS:
    source_case=case['reference_case_id'] if method=='state_reference' else (case['initialization_only_case_id'] if method=='init_only' else case['case_id']);row.append((lookup[(source_case,method)],source_case!=case['case_id']))
   loaded=[load(record) for record,reused in row];movie=out/(case['case_id']+'_four_panels.mp4');preview=[]
   with imageio.get_writer(movie,fps=25,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
    for i in range(max(64,max(len(x[0]) for x in loaded))):
     image=Image.new('RGB',(1280,354),'white');draw=ImageDraw.Draw(image);draw.text((10,7),f"Seed {case['seed']} | input {case['endpoint_hz']} Hz | {case['lighting']} | same physical launch",font=large,fill='#203c50')
     for j,((record,reused),(frames,trace,path)) in enumerate(zip(row,loaded)):
      x=j*320;result=record['result'];color='#167d6d' if record['verified_catch'] else '#ad4349';draw.text((x+7,39),LABELS[record['mode']]+' | '+('CATCH' if record['verified_catch'] else 'FAIL')+(' *' if reused else ''),font=font,fill=color);image.paste(frames[min(i,len(frames)-1)],(x,68));t=(0 if i==0 else trace[min(i-1,len(trace)-1)]['t']) if trace else 0
      position=(result.get('initial') or {}).get('position') if i==0 else (trace[min(i-1,len(trace)-1)]['object_position'] if trace else None)
      if position is not None:mark_ball(draw,position,x,68)
      draw.rectangle((x+3,72,x+317,92),fill='white');draw.text((x+6,74),f'{t:.2f}s'+(' | ended / frozen' if i>=len(frames) else ''),font=small,fill='#203c50')
      if result.get('steps')==0:draw.text((x+7,200),'No rollout: initialization failed',font=small,fill='white')
     draw.text((10,317),'44 actual episodes, 4 new seeds. * Baselines reused explicitly. Blue circles: offline true ball annotation.',font=small,fill='#203c50');draw.text((10,335),'All movies25fps; acquisition100/50/25Hz. Reference uses normal light; lighting is not a state-policy input.',font=small,fill='#203c50');writer.append_data(np.asarray(image));whole.append_data(np.asarray(image))
     if case['case_id']=='c01_normal25' and i%2==0:preview.append(image.resize((960,266)))
   if preview:
    palette=preview[len(preview)//2].quantize(colors=96);indexed=[x.quantize(palette=palette) for x in preview];indexed[0].save(out/'cadence25_first_seed.gif',save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2);preview[len(preview)//2].save(out/'cadence_preview.png')
   manifest.append({'case':case['case_id'],'endpoint_hz':case['endpoint_hz'],'source_episodes':[record['output'] for record,reused in row],'baseline_reused':[reused for record,reused in row],'outcomes':[record['verified_catch'] for record,reused in row],'source_native_recordings':[x[2] for x in loaded],'annotation':'blue true ball projection for display only, including occluded points; never provided to actor'})
 (out/'MEDIA_PROVENANCE.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'actual_episodes':44,'panels_with_reuse':64,'cases':16,'movie':str(master)}))

if __name__=='__main__':main()
