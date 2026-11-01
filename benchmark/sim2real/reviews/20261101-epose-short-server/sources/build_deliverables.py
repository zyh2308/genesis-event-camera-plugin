"""Create a factual report and visual evidence from the frozen real pilot."""
import json, hashlib, csv, subprocess, argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,default=ROOT/'outputs/EPOSE_服务器短训初测_20261101')
parser.add_argument('--evidence',type=Path)
args=parser.parse_args();OUT=args.output.resolve(); E=args.evidence.resolve() if args.evidence else OUT/'evidence'
summary=json.loads((E/'real_eval/summary.json').read_text())
training=json.loads((E/'training/result.json').read_text())['results']
manifest=json.loads((E/'evidence_manifest.json').read_text())
for name,record in manifest['small_artifacts'].items():
 p=E/name; assert p.stat().st_size==record['bytes']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],name
assert training[0]['initial_state_sha256']==training[1]['initial_state_sha256']
diagnostics={}
for row in training:
 steps=row['training_detail']; assert len(steps)==496
 assert all(np.isfinite([r['loss_total'],r['gradient_norm']]).all() and r['gradient_norm']>0 for r in steps)
 assert row['checkpoint_restore_hash_equal']
 val=row['validation_detail']
 diagnostics[row['generator']]={'updates':len(steps),'confidence_updates':sum(r['confidence_supervision_included'] for r in steps),'argmax_correct_cell_count':sum(r['confidence_argmax_cell_yx']==r['gt_centroid_cell_yx'] for r in val),'validation_count':len(val),'ordinary_eval_epe_network_px':row['validation_keypoint_mean_epe_px'],'GT_cell_diagnostic_only_epe_network_px':float(np.mean([r['gt_centroid_cell_decode_keypoint_mean_epe_px'] for r in val])),'first_epoch_mean_loss':float(np.mean([r['loss_total'] for r in steps[:16]])),'final_epoch_mean_loss':float(np.mean([r['loss_total'] for r in steps[-16:]]))}
(OUT/'audit_summary.json').write_text(json.dumps({'verified_artifacts':len(manifest['small_artifacts']),'same_initialization':True,'synthetic_diagnostics':diagnostics},indent=2))
with (OUT/'EPOSE_初测指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['生成器','训练更新','真实窗口','角点误差_px_346x260','ADD-S均值_cm_有效PnP','ADD-S成功率_阈值2.374cm','有效PnP数','平移误差_cm_有效PnP','原始旋转误差_deg_有效PnP'])
 for a,label in [('eventgenesis','EventGenesis'),('v2e','V2E')]:
  r=summary['results'][a];w.writerow([label,496,200,r['corner_epe_all_samples_px'],100*r['adds_mean_over_valid_pnp_m'],r['adds_success_0_1_d_fraction_all_samples'],r['pnp_valid_count'],100*r['translation_mean_over_valid_pnp_m'],r['rotation_raw_mean_over_valid_pnp_deg']])

fontroot=Path('/mnt/c/Windows/Fonts')
fonts={s:ImageFont.truetype(str(fontroot/'msyh.ttc'),s) for s in [17,19,21,23,25,28,34,38]}
C={'red':'#A00000','blue':'#3676A5','ink':'#263746','muted':'#617787','pale':'#E7F1F6','green':'#27805F'}
def tx(draw,xy,text,size=23,color=None):draw.text(xy,text,font=fonts[size],fill=color or C['ink'])
im=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(im)
tx(d,(44,32),'E-POSE：服务器短训初测',38,C['red'])
tx(d,(44,96),'匹配木块 CAD · 每组 496 次更新 · 真实片段 200 个事件窗',25,C['blue'])
cols=[44,305,580,865]; widths=[240,250,260,370]
heads=['生成器','角点误差 ↓','ADD-S 均值 ↓','位姿成功率 ↑']
for x,w,h in zip(cols,widths,heads):d.rectangle((x,170,x+w,227),fill=C['pale']);tx(d,(x+14,184),h,25,C['blue'])
for j,(arm,label) in enumerate([('eventgenesis','EventGenesis'),('v2e','V2E')]):
 r=summary['results'][arm];y=249+j*100
 vals=[label,f"{r['corner_epe_all_samples_px']:.2f} px",f"{100*r['adds_mean_over_valid_pnp_m']:.2f} cm",'0 / 200（0%）']
 for x,v in zip(cols,vals):tx(d,(x+14,y),v,28)
 d.line((44,y+67,1235,y+67),fill='#BED1DD',width=1)
tx(d,(44,465),'成功阈值：ADD-S ≤ 2.374 cm；PnP 失败计为不成功。',23)
tx(d,(44,507),'ADD-S 均值仅统计有效 PnP：EventGenesis 200 / 200，V2E 196 / 200。',21,C['muted'])
tx(d,(44,547),'仅 1 个物体、1 条真实序列、1 个 seed；不是 E-POSE 全集结果。',21,C['muted'])
d.rectangle((44,625,1236,681),fill=C['red']);tx(d,(65,638),'已有真实初测分数，但两组均未形成可靠的位姿估计。',25,'white')
im.save(OUT/'EPOSE_初测结果表.png')

pred={a:json.loads((E/f'real_eval/{a}_predictions.json').read_text()) for a in ['eventgenesis','v2e']}
edges=[(i,i^b) for i in range(8) for b in [1,2,4] if i<(i^b)]
def box(draw,points,colour):
 p=np.asarray(points,dtype=float)[1:]*1.65
 # Explicit clipping is display-only; scoring uses original unmodified predictions.
 for i,j in edges:
  x1,y1=p[i];x2,y2=p[j]
  if max(abs(x1),abs(y1),abs(x2),abs(y2))<1e7:draw.line((x1,y1,x2,y2),fill=colour,width=3)
def panel(raw,row,prediction=False):
 p=raw.resize((571,429));dr=ImageDraw.Draw(p);box(dr,row['gt_uv_ideal'],C['green'])
 if prediction and row['pnp_valid']:box(dr,row['reprojected_estimated_uv'],C['red'] if row is current_rows[0] else C['blue'])
 return p
frames=OUT/'demo_frames';frames.mkdir(exist_ok=True)
for i in range(200):
 current_rows=[pred['eventgenesis'][i],pred['v2e'][i]];n=current_rows[0]['index']
 raw=Image.open(E/f'real_eval/raw_undistorted/{n:04d}.png').convert('RGB')
 canvas=Image.new('RGB',(1280,720),'white');dr=ImageDraw.Draw(canvas)
 tx(dr,(40,22),'E-POSE 真实片段：短训模型的实际预测',34,C['red'])
 tx(dr,(40,81),'绿色：真实位姿    红色：EventGenesis    蓝色：V2E',23)
 for j,(row,label) in enumerate(zip(current_rows,['EventGenesis','V2E'])):
  x=40+630*j;canvas.paste(panel(raw,row,True),(x,174))
  tx(dr,(x,128),label,28,C['red'] if j==0 else C['blue'])
  message=f"角点误差 {row['corner_epe_px']:.1f} px"
  message+=f" · ADD-S {row['adds_m']*100:.1f} cm" if row['pnp_valid'] else ' · PnP 失败'
  tx(dr,(x,612),message,21)
 tx(dr,(40,659),f'固定采样窗 {i+1:03d} / 200（MAT {n:04d}） · 每组 496 次更新 · 未达到可靠位姿精度',19,C['muted'])
 canvas.save(frames/f'{i:04d}.png')
 if n==153:canvas.save(OUT/'EPOSE_真实预测对比.png')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%04d.png'),'-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'EPOSE_真实预测对比.mp4')],check=True)

report='''# E-POSE 匹配木块：服务器短训与真实片段初测

本次已经完成合成训练到真实测试，但两组在固定真实片段上的位姿成功率均为 0%。这是失败的学习效果初测，不是可用位姿能力或生成器优势的证明。不得用短训结果替代完整实验结论。

## 实际执行

- g1，RTX 3090，物理 GPU 1；GPU 已释放。PyTorch 2.8.0+cu126。
- 真实木块 CAD 原始坐标，转换 OBJ 只为 Genesis 读取，不缩放或重新居中。
- 4 条合成训练轨迹、2 条独立合成验证轨迹；每条 40 ms，4 个 10 ms 窗口，即每组 16 个训练窗、8 个验证窗。
- 相同随机初始化，seed 0；无外部预训练权重。共享物理轨迹、相机和标签。
- SingleShotPose，完整事件图 ON/OFF/total 三通道，416×416；每组 31 个短 epoch、496 次 SGD 更新，其中 256 次包括置信度监督。固定最终 checkpoint。
- 优化更新计时 EventGenesis 14.963 秒、V2E 14.412 秒；这不包含数据生成、V2E 配对生成、初始化、核验和整理时间。配对训练进程共约 51 秒。从 GPU 预约到完整流程结束约 26 分钟，前期环境核查和后续核验、整理另计。
- 不使用真实数据训练、调参、选模型；真实评分前冻结协议和权重哈希。没有 GT crop/mask/box 或 GT 选点。
- 真实测试预先固定 block/r_01_gl，MAT 0003、0006……0600，共 200 个原始约 10 ms 事件窗；事件去畸变，固定 EPNP。GT 仅用于评分。

## 真实结果

| 生成器 | 角点误差 ↓（346×260 原始像素域） | ADD-S 均值 ↓ | ADD-S≤0.1d 成功率 ↑ | 有效 PnP |
|---|---:|---:|---:|---:|
| EventGenesis | 116.32 px | 67.22 cm | 0/200（0%） | 200/200 |
| V2E | 83.10 px | 38.88 cm | 0/200（0%） | 196/200 |

CAD 精确直径 d=23.7367 cm，成功阈值 2.3737 cm。ADD-S 均值只对有效 PnP 求均值，两个有效集合不同；PnP 失败仍计入 200 个窗口的成功率分母。ADD、平移和原始旋转误差另见 CSV/summary.json。原始旋转误差没有做对称性等价处理。

## 为什么不能据此比较仿真器

两组还没有学会可靠地定位物体。8 个合成验证窗中，置信度最大值选择的网格均未命中物体中心网格。普通推理角点误差为 EventGenesis 375.54 / V2E 497.50 网络像素（416×416 域）。因此问题在进入真实域之前就已存在，不能将真实失败直接归为某个事件生成器的域差异。

训练损失下降且 496 次更新均有有限非零梯度，说明发生了训练；但训练损失下降不等于可用的检测能力。把 GT 中心网格强行告诉模型时的误差更小，只是离线定位诊断，不作为正式分数或真实预测使用。训练集极小、位姿/背景覆盖有限且从随机初始化开始，是明显限制；仅凭本次记录不能确定唯一根因。

当前真实结果中 V2E 的数值误差较小，但两组成功数都是零，且只有一个物体、一条序列、一个 seed，200 个相邻采样窗并非 200 条独立序列。不能据此宣称 V2E 或 EventGenesis 的一般优劣。

## 核验与边界

- 六条合成轨迹的真实网格顶点投影与渲染轮廓最大偏差约 1.003 px；虚拟 AABB 超出实际网格轮廓是预期几何区别。
- 27 个 simulator core 源文件与 frozen e79e656ac00f7847b98db9e12bf43b093f3afff1 一致。
- 两组初始模型哈希相同；训练日志、模型状态变化及保存恢复核验通过。
- 独立重算全部 200 个窗口的角点误差和成功计数；各组抽样 20 个有效 PnP 重算 ADD/ADD-S，差异为 0。
- 抽查 5 个真实窗口与原始 HDF5 的事件坐标、极性和事件数完全一致；抽查 MAT SHA 和逐点 GT 投影一致。未使用已知有误的官方 keypointprojections 字段。
- 仅为重建几何定义下的初测，不声称复现官方 E-POSE 完整排行榜协议。

## 组会可以这样说

“位姿线补上了匹配真实木块的合成训练和真实 E-POSE 初测。受时间限制，本次每组只做 496 次更新，测试一个固定片段的 200 个事件窗。目前两组均未达到位姿成功阈值，合成留出诊断也显示置信度定位没有学好，因此本轮暂不能用来评价生成器的真实泛化优势。下一步先在更充分的合成留出集上通过物体定位和位姿精度门槛，再开展统一预算的真实对照。”

## 后续执行边界

不要根据这 200 个真实窗调参。先在合成训练/留出数据上补充分布、置信度学习及短程收敛诊断；普通 eval、置信度选点必须优先过关。下一轮更改要另立 run/config，与本次冻结结果区分；已评分片段应登记为已见，最终确认使用预先保留的其他序列。训练只按合成验证决定，不挑选有利真实帧。

服务器大产物：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-epose-wood-short-research-v1/`。本地 evidence/ 包含协议、逐步训练日志、逐窗预测、独立核验和哈希索引；checkpoint 和完整合成 bank 留在服务器。
'''
(OUT/'EPOSE_短训结果说明.md').write_text(report,encoding='utf-8')
print(json.dumps({'verified_files':len(manifest['small_artifacts']),'diagnostics':diagnostics,'output':str(OUT)},ensure_ascii=False,indent=2))
