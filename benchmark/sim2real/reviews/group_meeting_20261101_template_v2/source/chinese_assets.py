import json,subprocess,csv,shutil
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OLD=ROOT/'outputs/Sim2Real_20261101';OUT=ROOT/'outputs/Sim2Real_20261101_模板精简版';OUT.mkdir(exist_ok=True)
FONT='/mnt/c/Windows/Fonts/msyh.ttc';BOLD='/mnt/c/Windows/Fonts/msyhbd.ttc'
RED='#A00000';BLUE='#3676A5';ORANGE='#D47819';INK='#22313D';GRAY='#647887'
FONTS={}
def label(im,s,x,y,n=24,c=INK,bold=False):
 key=(n,bold)
 if key not in FONTS:FONTS[key]=ImageFont.truetype(BOLD if bold else FONT,n)
 ImageDraw.Draw(im).text((x,y),s,font=FONTS[key],fill=c)
def paste(im,a,x,y,w,h):
 if not isinstance(a,Image.Image):a=Image.fromarray(a)
 im.paste(a.resize((w,h),Image.Resampling.LANCZOS),(x,y))
def rec_frame(a,j):
 im=Image.new('RGB',(1280,720),'white');label(im,'真实事件重建',42,20,30,RED,True)
 for c,s in enumerate(['真实事件','参考图像','EventGenesis','V2E']):
  x=24+c*312;label(im,s,x,88,25,RED if c==2 else BLUE if c==3 else INK)
  paste(im,a[148:370,x:x+296],x,132,296,222);paste(im,a[420:642,x:x+296],x,416,296,222)
 label(im,'局部放大',24,374,22,GRAY);label(im,'HQF｜片段一' if j<240 else 'HQF｜片段二',24,675,21,GRAY);return im
def flow_frame(a,j):
 im=Image.new('RGB',(1280,720),'white');label(im,'合成场景与事件' if j<180 else '光流预测对照',32,20,30,RED,True)
 if j<180:
  for c,s in enumerate(['仿真场景','EventGenesis 事件','V2E 事件']):
   x=32+c*416;label(im,s,x,140,26,RED if c==1 else BLUE if c==2 else INK);paste(im,a[190:478,x:x+384],x,190,384,288)
  label(im,'4 个端点｜30 ms｜慢放',32,630,23,GRAY)
 else:
  for c,s in enumerate(['光流真值','EventGenesis','V2E']):
   x=32+c*416;label(im,s,x,92,26,RED if c==1 else BLUE if c==2 else INK);paste(im,a[150:438,x:x+384],x,138,384,288)
  for c,s in enumerate(['场景','误差','误差']):
   x=32+c*416;label(im,s,x,452,23);paste(im,a[500:668,x:x+224],x,492,224,168)
  label(im,'合成留出测试｜统一色标｜误差 0—8 像素',32,684,19,GRAY)
 return im
def pose_frame(a,j):
 im=Image.new('RGB',(1280,720),'white');label(im,'物体位姿',42,20,30,RED,True)
 for x,s in [(50,'真实事件'),(680,'位姿真值')]:
  label(im,s,x,116,28,BLUE if x==680 else INK);paste(im,a[174:587,x:x+550],x,170,550,413)
 label(im,'E-POSE｜真值示例，非模型预测',50,644,24,GRAY);return im
for name,fn,poster_frame,poster in [('01_reconstruction_HQF.mp4',rec_frame,0,'reconstruction_poster.png'),('02_flow_ReplicaCAD.mp4',flow_frame,210,'flow_result_poster.png'),('03_pose_EPOSE_task.mp4',pose_frame,50,'pose_poster.png')]:
 cap=cv2.VideoCapture(str(OLD/name));fps=cap.get(cv2.CAP_PROP_FPS)
 process=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r',str(fps),'-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/name)],stdin=subprocess.PIPE)
 j=0
 while True:
  ok,a=cap.read()
  if not ok:break
  im=fn(cv2.cvtColor(a,cv2.COLOR_BGR2RGB),j);process.stdin.write(np.asarray(im).tobytes())
  if j==poster_frame:im.save(OUT/poster)
  if name.startswith('02') and j==30:im.save(OUT/'flow_scene_poster.png')
  j+=1
 process.stdin.close();assert process.wait()==0;cap.release();print(name,j,flush=True)
# Evidence-only crops for editable native labels in the deck.
for name in ['reconstruction_boxes_poster.png','reconstruction_poster_pillar_1_poster.png']:
 a=Image.open(OLD/name)
 for i in range(4):
  x=24+312*i;a.crop((x,148,x+296,370)).save(OUT/f'{name.split("_poster")[0]}_panel{i}.png')
  if name.startswith('reconstruction_boxes'):a.crop((x,420,x+296,642)).save(OUT/f'reconstruction_crop_panel{i}.png')
a=Image.open(OLD/'flow_result_poster.png')
for i in range(3):
 x=32+416*i;a.crop((x,150,x+384,438)).save(OUT/f'flow_panel{i}.png');a.crop((x,500,x+224,668)).save(OUT/f'flow_small_panel{i}.png')
a=Image.open(OLD/'pose_real_task_poster.png')
for i,x in enumerate([50,680]):a.crop((x,174,x+550,587)).save(OUT/f'pose_panel{i}.png')
for n in ['results.json','result_tables.csv','reconstruction_all14_sequences.csv','SOURCE_AUDIT.md','组会讲稿.md']:shutil.copyfile(OLD/n,OUT/n)
print('Chinese assets complete',flush=True)
