"""Compose native Genesis recordings. No new physics or synthetic actions."""
import argparse,hashlib,json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

METHODS=['ours','v2e','init_only','state_reference']
LABELS={'ours':'Ours','v2e':'V2E','init_only':'First RGB-D only','state_reference':'State reference'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,13);large=ImageFont.truetype(FONT,21)

def load(record):
 p=Path(record['output']);video=p/'episode_000/genesis.mp4'
 if video.exists():
  with imageio.get_reader(video) as reader:frames=[Image.fromarray(x).resize((320,240)) for x in reader]
 else:frames=[Image.open(p/'migration_audit/sensor_episode_000/initial_rgb.png').convert('RGB').resize((320,240))]
 trace=json.loads((p/'episode_000/trace.json').read_text()) if (p/'episode_000/trace.json').exists() else []
 return frames,trace,str(video)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();data=json.loads((a.root/'PAIRED_PORTABLE.json').read_text());assert data['complete'] and not data['violations']
 output=a.root/'delivery';output.mkdir(exist_ok=True);records={(r['case_id'],r['mode']):r for r in data['records']};manifest=[]
 conditions=['normal','visual_fast','backlit'];per_case=[];allpath=output/'all_72_native_view_comparisons.mp4'
 with imageio.get_writer(allpath,fps=25,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as allwriter:
  for seed in range(1,7):
   for condition in conditions:
    case=f's{seed:02d}_{condition}';row=[records[(case,m)] for m in METHODS];loaded=[load(r) for r in row];frames=[]
    for i in range(max(64,max(len(x[0]) for x in loaded))):
     canvas=Image.new('RGB',(1280,344),'white');draw=ImageDraw.Draw(canvas);draw.text((10,8),f"{case} | seed {row[0]['seed']} | fixed 7 deg camera, original policy",font=large,fill='#203c50')
     for j,(r,(images,trace,path)) in enumerate(zip(row,loaded)):
      x=j*320;success=r['verified_catch'];color='#167d6d' if success else '#ad4349'
      draw.text((x+8,39),LABELS[r['mode']]+' | '+('CATCH' if success else 'FAIL'),font=font,fill=color);canvas.paste(images[min(i,len(images)-1)],(x,68))
      if trace:t=0. if i==0 else trace[min(i-1,len(trace)-1)]['t']
      else:t=0.
      note=f'{t:.2f} simulated s'+(' | ended / frozen' if i>=len(images) else '')
      if r['result'].get('steps')==0:note='No rollout: RGB-D initialization failed'
      draw.rectangle((x+3,72,x+317,92),fill='white');draw.text((x+6,74),note,font=small,fill='#203c50')
     draw.text((10,319),'One RGB-D initialization, then events + proprioception. All predefined successes and failures retained.',font=small,fill='#203c50');allwriter.append_data(np.asarray(canvas));frames.append(canvas)
    perpath=output/(case+'_four_methods.mp4')
    with imageio.get_writer(perpath,fps=25,codec='libx264',quality=6,macro_block_size=None,ffmpeg_params=['-threads','2']) as writer:
     for frame in frames:writer.append_data(np.asarray(frame))
    manifest.append({'case':case,'source_native_videos':[x[2] for x in loaded],'source_outcomes':[r['verified_catch'] for r in row],'frames':len(frames),'fps':25,'time_scope':'simulated time; actual end frames explicitly frozen; original raw recordings retained'})
    per_case.append((case,row,frames))
    # Retain only one short preview in memory; all case recordings remain on disk.
    if condition=='visual_fast' and seed==1:
     preview=frames[::2];palette=preview[min(16,len(preview)-1)].resize((960,258)).quantize(colors=96);indexed=[im.resize((960,258)).quantize(palette=palette) for im in preview];indexed[0].save(output/'visual_fast_first_seed.gif',save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2);frames[min(30,len(frames)-1)].save(output/'visual_fast_preview.png')
    per_case.clear()
 (output/'MEDIA_PROVENANCE.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({'episodes':72,'cases':18,'output':str(output),'all_movie_bytes':allpath.stat().st_size}))

if __name__=='__main__':main()
