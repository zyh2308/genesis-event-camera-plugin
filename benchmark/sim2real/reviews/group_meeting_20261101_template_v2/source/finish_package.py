import json,hashlib,zipfile,shutil,subprocess,csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
import numpy as np
ROOT=Path(__file__).resolve().parents[2];BUILD=Path(__file__).parent;OUT=ROOT/'outputs/Sim2Real_20261101_模板精简版'
font_manager.fontManager.addfont('/mnt/c/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font_manager.FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc').get_name(),'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
red='#A00000';blue='#3676A5';gray='#9AA8B2'
d=json.loads((OUT/'results.json').read_text())
for name,cats,series,ylab,maximum,scope in [
 ('figure_01_reconstruction',['LPIPS','SSIM','PSNR','MSE'],[('EventGenesis',[13,9,1,1],red)],'优于 V2E 的序列数',14.8,'HQF · 14 序列 · 单种子'),
 ('figure_02_flow',['全有效区','事件活跃区'],[('EventGenesis',d['flow']['eventgenesis'],red),('V2E',d['flow']['v2e'],blue),('零流',d['flow']['zero'],gray)],'EPE（像素 / 30 ms）↓',4.2,'合成留出诊断 · 单种子'),
 ('figure_03_pose',['EventGenesis','V2E'],[('角点误差',[d['pose']['eventgenesis'],d['pose']['v2e']],[red,blue])],'角点误差（像素）↓',40,'通用立方体 · 合成预实验')]:
 fig,ax=plt.subplots(figsize=(9,5),layout='constrained');x=np.arange(len(cats));bw=.64/len(series)
 for j,(label,vals,col) in enumerate(series):
  bars=ax.bar(x+(j-(len(series)-1)/2)*bw,vals,bw,color=col,label=label)
  ax.bar_label(bars,fmt='%.0f' if name.endswith('reconstruction') else '%.3f' if name.endswith('flow') else '%.2f',padding=5)
 ax.set_xticks(x,cats);ax.set_ylabel(ylab);ax.set_ylim(0,maximum);ax.set_title(scope,loc='left',pad=16)
 if len(series)>1:ax.legend(frameon=False,ncol=3,loc='upper center')
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig)
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EventGenesis 实验展示</title><style>body{font-family:"Microsoft YaHei",sans-serif;color:#263746;max-width:1200px;margin:32px auto;padding:0 24px;background:white}h1,h2{color:#a00000}h1{font-size:30px;border-bottom:2px solid #b7c9d5;padding-bottom:20px}section{margin:42px 0}video{width:100%;max-height:75vh;background:#fff}p{color:#617787}a{color:#3676a5}img{max-width:850px;width:100%}</style><h1>EventGenesis：公共真实数据验证</h1><a href="EventGenesis_组会汇报_模板精简版.pptx">可编辑 PPT</a>　<a href="result_tables.csv">结果表</a>　<a href="SOURCE_AUDIT.md">数据来源</a>
<section><h2>01 图像重建</h2><video controls poster="reconstruction_poster.png" preload="metadata" src="01_reconstruction_HQF.mp4"></video><p>HQF 真实测试</p><img src="figure_01_reconstruction.png" alt="重建指标对照"></section>
<section><h2>02 事件光流</h2><video controls poster="flow_scene_poster.png" preload="metadata" src="02_flow_ReplicaCAD.mp4"></video><p>合成留出诊断，尚无 MVSEC 成绩</p><img src="figure_02_flow.png" alt="光流误差"></section>
<section><h2>03 物体位姿</h2><video controls poster="pose_poster.png" preload="metadata" src="03_pose_EPOSE_task.mp4"></video><p>E-POSE 真值示例，非模型预测</p><img src="figure_03_pose.png" alt="合成位姿预实验"></section><script>document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(o=>{if(o!==v)o.pause()})));</script></html>'''
(OUT/'index.html').write_text(html,encoding='utf8')
(OUT/'README.txt').write_text('模板精简版：沿用用户提供PPT的实际母版、版式和页眉元素。\n打开index.html播放中文标注视频；PPT表格/图表可编辑。\n结果未变；重建是真实测试，光流是合成诊断，位姿视频是真值示例。\n视频为同目录独立MP4，PPT右下角“播放视频”可打开。\n',encoding='utf8')
old=(OUT/'SOURCE_AUDIT.md').read_text();(OUT/'SOURCE_AUDIT.md').write_text('# 模板精简版说明\n\n采用2026-09-14文件的实际母版、版式、页眉线条及编号。其旧实验内容不采用。精简正文与图内文字，保留模型/数据集/指标专名。数值、预测和结论边界未改变。重建图改为四项指标的逐序列胜出数量（13/9/1/1），详细原始数据仍见CSV。\n\n以下保留原始科学来源审阅记录；其中旧版文件名与旧版展示方式以本版README为准。\n\n'+old,encoding='utf8')
src=OUT/'source';src.mkdir(exist_ok=True)
for name in ['prepare_template.py','chinese_assets.py','build_simplified.mjs','finish_package.py']:shutil.copyfile(BUILD/name,src/name)
shutil.copyfile(BUILD/'template_reference.json',src/'template_reference.json')
qa={}
for v in OUT.glob('*.mp4'):
 result=subprocess.run(['ffmpeg','-v','error','-i',str(v),'-f','null','-'],capture_output=True,text=True);assert result.returncode==0 and not result.stderr,result.stderr
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(v)]));s=probe['streams'][0];qa[v.name]={'frames':int(s['nb_frames']),'dimensions':[s['width'],s['height']],'decoded':True}
assert qa['01_reconstruction_HQF.mp4']['frames']==480 and qa['02_flow_ReplicaCAD.mp4']['frames']==360 and qa['03_pose_EPOSE_task.mp4']['frames']==200
(BUILD/'media_qa.json').write_text(json.dumps(qa,indent=2))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'SHA256SUMS.txt').write_text('\n'.join(f'{sha(p)}  {p.relative_to(OUT).as_posix()}' for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt')+'\n')
dest=ROOT/'outputs/EventGenesis_组会材料_模板精简版.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in OUT.rglob('*'):
  if p.is_file():z.write(p,Path(OUT.name)/p.relative_to(OUT))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None
print(json.dumps({'zip':str(dest),'bytes':dest.stat().st_size,'sha256':sha(dest),'videos':qa},ensure_ascii=False))
