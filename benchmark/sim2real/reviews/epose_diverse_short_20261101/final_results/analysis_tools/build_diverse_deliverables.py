import json,csv,hashlib,subprocess,argparse
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);O=ap.parse_args().output.resolve();E=O/'evidence';F=E/'final';S=json.loads((F/'real_eval/summary.json').read_text());T=json.loads((F/'training/result.json').read_text());B=json.loads((F/'bank/manifest.json').read_text());cfg=json.loads((F/'protocol.json').read_text());results=S['results'];bc=B['protocol'];scene_ids=['train_00','train_04','train_08'];ntrain=bc['train_rollouts'];nval=bc['validation_rollouts'];nt=cfg['train_windows'];nv=cfg['validation_windows']
assert json.loads((F/'real_eval/independent_verification.json').read_text())['passed'];assert json.loads((F/'independent_synthetic_verification.json').read_text())['passed']
manifest=json.loads((E/'manifest.json').read_text())
for name,m in manifest['files'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==m['sha256'],name
C={'red':'#A00000','blue':'#3676A5','ink':'#263746','muted':'#617787','pale':'#E7F1F6','green':'#27805F'};fonts={n:ImageFont.truetype('/mnt/c/Windows/Fonts/msyh.ttc',n) for n in [18,19,20,21,22,23,24,25,26,27,28,30,32,34,36,38]}
def text(d,x,y,s,n=23,c=None):d.text((x,y),str(s),font=fonts[n],fill=c or C['ink'])
def number(v,scale=1):return '—' if v is None else (f'{v*scale:.2e}' if abs(v*scale)>=10000 else f'{v*scale:.2f}')
def canvas(title,subtitle):
 im=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(im);text(d,44,26,title,36,C['red']);text(d,44,87,subtitle,23,C['blue']);return im,d
im,d=canvas('合成场景：真实木纹、共享相机轨迹','同一 CAD、物理轨迹与标签，仅替换事件生成器')
for j,name in enumerate(scene_ids):
 x=44+j*410;src=Image.open(F/f'bank/preview/{name}.png').convert('RGB').resize((390,293));im.paste(src,(x,142));text(d,x+8,447,f'训练场景 {j+1}',25)
text(d,44,515,f'{ntrain} 条训练轨迹 + {nval} 条独立验证轨迹',27);text(d,44,559,f'先预热 40 ms，再取 10 ms 事件窗；{nt} / {nv} 个训练 / 验证窗。',23)
d.rectangle((44,640,1236,688),fill=C['red']);text(d,60,651,'公开 YCB 木块纹理；几何文件与 E-POSE CAD 的 SHA256 完全一致。',22,'white');im.save(O/'合成场景_扩充后.png')
im,d=canvas('共享物理场景，两种事件输出','固定选取三条轨迹的 60–70 ms 窗口，无生成器专属筛选')
for j,name in enumerate(scene_ids):
 x=44+j*410;text(d,x+8,129,f'场景 {j+1}',25)
 for k,(arm,label,col) in enumerate([('eventgenesis','EventGenesis',C['red']),('v2e','V2E',C['blue'])]):
  img=Image.open(F/f'input_previews/{name}_{arm}.png').convert('RGB').resize((300,225));im.paste(img,(x+38,180+k*255));text(d,x+38,155+k*255,label,20,col)
im.save(O/'合成事件_配对样例.png')
im,d=canvas('独立合成验证：是否找到物体',f'相同 {nval} 条留出轨迹、{nv} 个窗口；完整事件图输入')
xs=[44,316,650,960];ws=[260,322,298,276]
for x,w,h in zip(xs,ws,['生成器','中心误差 ↓','角点误差 ↓','选对网格 ↑']):d.rectangle((x,154,x+w,211),fill=C['pale']);text(d,x+12,170,h,24,C['blue'])
for j,r in enumerate(T['results']):
 y=237+j*114;before=r['history'][0];v=r['validation'];values=[r['generator'].replace('eventgenesis','EventGenesis').replace('v2e','V2E'),f"{before['mean_centroid_epe_px']:.1f} → {v['mean_centroid_epe_px']:.1f} px",f"{before['mean_corner_epe_px']:.1f} → {v['mean_corner_epe_px']:.1f} px",f"{v['correct_cell_count']} / {v['count']}"]
 for x,s in zip(xs,values):text(d,x+12,y,s,26)
 d.line((44,y+72,1235,y+72),fill='#BED1DD')
text(d,44,493,'前后均在同一新验证集统计；误差在 416×416 网络像素域。',22);text(d,44,538,f"每组追加 {cfg['optimizer_updates_per_arm']:,} 次全网络更新；仅按合成验证选择权重。",22)
coverage=json.loads((F/'physical_coverage_statistics.json').read_text())['validation'];text(d,44,581,f"始终猜画面中心：误差 {coverage['fixed_image_center_epe_px_network']:.2f} px，选对网格 {coverage['fixed_image_center_correct_grid_fraction']*100:.0f}%。",21)
d.rectangle((44,634,1236,688),fill=C['red']);text(d,62,648,'定位通过是初测前提，不能代替完整 6DoF 位姿精度。',24,'white');im.save(O/'合成定位_新留出集.png')
seq=cfg['real_sequence'];im,d=canvas('真实 E-POSE：定位与位姿分别报告',f'{seq} · 固定 200 个真实事件窗 · 合成训练，无真实微调')
xs=[44,316,533,750,992];ws=[260,205,205,230,244]
for x,w,h in zip(xs,ws,['模型 / 版本','中心误差 ↓','角点误差 ↓','ADD-S 均值 ↓','成功率 ↑']):d.rectangle((x,144,x+w,201),fill=C['pale']);text(d,x+12,161,h,23,C['blue'])
labels=[('eventgenesis_v1_control','EventGenesis','初次短训'),('eventgenesis','EventGenesis','扩充场景后'),('v2e_v1_control','V2E','初次短训'),('v2e','V2E','扩充场景后')]
labels=[item for item in labels if item[0] in results]
with (O/'EPOSE_真实初测_指标.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(['模型','版本','真实序列','窗口数','中心误差_px','角点误差_px','ADD-S_cm_有效PnP','ADD-S成功数_分母200','有效PnP数','checkpoint_sha256','ADD_cm_有效PnP','ADD成功率_全部窗口','平移误差_cm_有效PnP','旋转误差_deg_未处理对称性_有效PnP'])
 for j,(a,label,version) in enumerate(labels):
  r=results[a];y=218+j*86;text(d,56,y,label,25);text(d,56,y+33,version,18,C['muted']);text(d,328,y+8,number(r['centroid_epe_all_samples_px'])+' px',25);text(d,545,y+8,number(r['corner_epe_all_samples_px'])+' px',25);text(d,762,y+3,number(r['adds_mean_over_valid_pnp_m'],100)+' cm',25);text(d,762,y+38,f"有效 PnP {r['pnp_valid_count']}/200",18,C['muted']);text(d,1004,y+8,f"{r['adds_success_0_1_d_count']/2:.1f}%",27);d.line((44,y+74,1235,y+74),fill='#BED1DD');w.writerow([label,version,seq,200,r['centroid_epe_all_samples_px'],r['corner_epe_all_samples_px'],None if r['adds_mean_over_valid_pnp_m'] is None else r['adds_mean_over_valid_pnp_m']*100,r['adds_success_0_1_d_count'],r['pnp_valid_count'],r['checkpoint']['sha256'],None if r['add_mean_over_valid_pnp_m'] is None else r['add_mean_over_valid_pnp_m']*100,r['add_success_0_1_d_fraction_all_samples'],None if r['translation_mean_over_valid_pnp_m'] is None else r['translation_mean_over_valid_pnp_m']*100,r['rotation_raw_mean_over_valid_pnp_deg']])
if not S.get('paired_generator_comparison_available',True):text(d,56,476,'另一组未通过合成定位门槛；本表仅为单组有效初测。',24,C['red'])
text(d,44,582,'像素误差：346×260 域；ADD-S ≤ 2.374 cm 为成功，PnP 失败计为失败。',20);d.rectangle((44,636,1236,687),fill=C['red']);text(d,62,649,'严格位姿成功率仍偏低；本轮尚未证明生成器优势。',23,'white');im.save(O/'真实位姿初测_主指标.png')
rows={a:json.loads((F/f'real_eval/{a}_predictions.json').read_text()) for a in ['eventgenesis','v2e'] if a in results};edges=[(i,i^k) for i in range(8) for k in [1,2,4] if i<(i^k)];frames=O/'demo_frames';frames.mkdir(exist_ok=True)
def box(draw,points,color):
 p=np.asarray(points)[1:]*1.65
 for i,j in edges:
  if np.max(np.abs(p[[i,j]]))<1e7:draw.line(tuple(p[i])+tuple(p[j]),fill=color,width=3)
for i in range(200):
 n=next(iter(rows.values()))[i]['index'];raw=Image.open(F/f'real_eval/raw_undistorted/{n:04d}.png').convert('RGB');im,d=canvas('真实 E-POSE：合成训练后的位姿预测','绿色：真值    红色：EventGenesis    蓝色：V2E')
 for j,(a,label,color) in enumerate([('eventgenesis','EventGenesis',C['red']),('v2e','V2E',C['blue'])]):
  if a not in rows:
   x=40+j*630;tile=raw.resize((571,429));im.paste(tile,(x,174));text(d,x,127,label,28,color);text(d,x,615,'未通过合成定位门槛，未作真实评分',21)
   if n==153:tile.save(O/f'ppt_panel_{j}.png')
   continue
  row=rows[a][i];x=40+j*630;tile=raw.resize((571,429));td=ImageDraw.Draw(tile);box(td,row['gt_uv_ideal'],C['green'])
  if row['pnp_valid']:box(td,row['reprojected_estimated_uv'],color)
  if n==153:tile.save(O/f'ppt_panel_{j}.png')
  im.paste(tile,(x,174));text(d,x,127,label,28,color);s=f"角点误差 {row['corner_epe_px']:.1f} px";s+=f" · ADD-S {row['adds_m']*100:.1f} cm" if row['pnp_valid'] else ' · PnP 失败';text(d,x,615,s,21)
 text(d,40,667,f'{seq} · 固定窗口 {i+1:03d}/200（MAT {n:04d}）· 全部窗口依次展示',19,C['muted']);im.save(frames/f'{i:04d}.png')
 if n==153:im.save(O/'真实预测_固定第51窗.png')
subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%04d.png'),'-c:v','libx264','-threads','4','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'EPOSE_真实预测_配对短训.mp4')],check=True)
table='\n'.join(f"| {label} {version} | {number(results[a]['centroid_epe_all_samples_px'])} | {number(results[a]['corner_epe_all_samples_px'])} | {number(results[a]['adds_mean_over_valid_pnp_m'],100)} | {results[a]['adds_success_0_1_d_count']}/200 | {results[a]['pnp_valid_count']}/200 |" for a,label,version in labels)
syn='\n'.join(f"| {r['generator']} | {r['validation']['mean_centroid_epe_px']:.3f} | {r['validation']['mean_corner_epe_px']:.3f} | {r['validation']['correct_cell_count']}/{r['validation']['count']} | {r['selected_update']} |" for r in T['results'])
geometry_path=F/'synthetic_pose_geometry_final.json';geometry=json.loads(geometry_path.read_text()) if geometry_path.exists() else None
if geometry is not None:
 for a,r in geometry['results'].items():assert r['checkpoint_sha256']==S['weights_frozen']['checkpoints'][a]['sha256']
geometry_table='未取得最终所选权重的单独PnP诊断，不能据中心网格指标单独认证完整6DoF。' if geometry is None else '| 生成器 | 有效PnP | PnP后中心投影误差 px ↓ | ADD-S cm ↓ | ADD-S成功数 |\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"| {a} | {r['pnp_valid_count']}/{r['validation_count']} | {number(r['pnp_centroid_epe_native_px'])} | {number(r['adds_mean_valid_m'],100)} | {r['adds_success_count']}/{r['validation_count']} |" for a,r in geometry['results'].items())
gap_path=O/'posttest_domain_gap.json';gap=json.loads(gap_path.read_text()) if gap_path.exists() else None
gap_text='未进行额外域差异诊断。' if gap is None else f'''在固定权重和正式初测完成后，只读取现有逐窗预测及训练增强记录作诊断，没有新增优化、预处理修改或checkpoint选择。

- 原始物理bank中，70.5%的真实中心位置超出训练中心的轴对齐范围；计入实际9,600个增强训练样本后，仍有{gap['real_center_outside_actual_augmented_training_axis_aligned_range_fraction']*100:.1f}%的真实中心超出实际增强范围。后者才是包含增强的覆盖诊断，不能将70.5%当成增强后比例。
- {100*(1-gap['real_gt_all_corners_in_frame_fraction']):.1f}%的真实窗口，其GT包围框角点不完全位于画面内；合成生成与增强均要求完整可见。真实框宽/高中位数为画面的25.4%/50.4%，原始合成bank为19.3%/38.6%。
- 真实事件数中位数{gap['real_events_rectified_median']:.1f}，合成两组分别1743.5/520。事件总数更多不能直接等价于下游质量更好。

这些是测试后的相关性诊断，支持位置、尺度及画面截断覆盖不足的解释，但不能确认唯一根因。GT框仅用于统计，没有用于模型输入、裁剪或选解。该片段已登记为已见；后续改进不能继续把它当作未见最终测试。具体数据见posttest_domain_gap.json。'''
report=f'''# E-POSE 三小时扩充场景与服务器短训初测

## 真实初测结果

{seq}，固定 MAT 0003:3:0600，共 200 个原生约 10 ms 事件窗。全部训练与 checkpoint 选择发生在本次真实评分之前。旧 v1 权重也在同一片段评分，前后数值可在相同测试样本上比较；不能把改善唯一归因于某一项修改。

| 模型 | 中心误差 px ↓ | 角点误差 px ↓ | ADD-S cm ↓ | 成功数 ↑ | 有效 PnP |
|---|---:|---:|---:|---:|---:|
{table}

本轮EventGenesis的平均ADD-S较低，但V2E的角点误差和严格成功数更好。两组严格成功率分别仅0%和1%，尚未形成可靠的真实6DoF能力，也不能据此得出生成器整体优劣结论。与旧v1参照相比误差下降，但训练数据、训练适配及预算历史均有变化，不能将改善唯一归因于某一项修改。

实际获准真实评分的新模型：{', '.join(rows)}。若只有一组通过定位门槛，仅交付该组初测，不作生成器优劣对比；另一组的失败保留在合成验证表中。

像素误差在 346×260 理想去畸变像素域。ADD-S 均值只对有效 PnP 求均值，成功率分母始终为全部 200 窗，PnP 失败计为失败。阈值为 CAD 直径的 10%，即 2.3737 cm；原始旋转误差另存机器结果，因木块有对称性不作主要结论。

## 合成定位结果

| 生成器 | 中心误差 px ↓ | 角点误差 px ↓ | 正确网格 ↑ | 所选追加更新 |
|---|---:|---:|---:|---:|
{syn}

合成误差在 416×416 网络像素域。固定猜画面中心的误差为 {coverage['fixed_image_center_epe_px_network']:.3f} px，正确网格比例为 {coverage['fixed_image_center_correct_grid_fraction']*100:.1f}%；只用训练集平均中心作常量预测的误差为 {coverage['train_mean_center_prior_epe_px_network']:.3f} px。预先登记的本轮定位门槛为中心误差<25 px 且正确网格比例≥75%；角点误差和真实 6DoF 精度单独报告。这个新数据集的定位门槛与旧 8 窗开发集的完整关键点门槛不同，协议在新训练前已明确修订，未因真实分数而改变。不能据定位通过宣称完整姿态准确。

另以实际网络角点输入固定EPNP，对同一合成留出集做只读3D诊断；没有GT选点或GT解法选择，也没有据此更改checkpoint选择或门槛。下表中心投影误差在原生346×260像素域，不能直接与上一表416×416误差比较。ADD-S阈值与真实测试一致；这些仍是合成结果。

{geometry_table}

## 测试后只读域差异诊断

{gap_text}

## 排查与实质改变

最初两组在 16 个合成训练窗中都没有正确选中物体中心网格；强制 GT 网格仅用于诊断时，角点误差反而较小。标签、坐标解码、投影和原损失的数值检查通过，更换 BN 推理模式没有独自解决问题。原始数据仅 4 条 40 ms 训练轨迹、16 个训练窗，木块无纹理，相机静止，无法支撑可靠的泛化判断。旧开发留出集只有 2 条轨迹、8 窗，并且在排查中被反复使用；其较高分数不能当作最终独立验证。

本轮使用官方 YCB 木块纹理，随附 nontextured.ply 的 SHA256 与 E-POSE CAD 完全一致。物体固定、相机沿独立合成轨迹运动；姿态、方位、距离、高度、画面位置和速度均按预先固定的种子变化。生成 {ntrain} 条训练和 {nval} 条验证轨迹；每条 {bc['duration_ms']} ms，前 40 ms 仅作传感器运动预热，后续形成原生 10 ms 窗口，即 {nt}/{nv} 个训练/验证窗。窗口来自相同轨迹，存在相关性。共有物理 bank、标签和划分，两组没有按事件数或模型成绩单独筛选样本。

R8/R9 的扩充视角数据仍存在观察方式问题：相机始终朝向物体，所有 256 个合成窗口的物体中心位移中位数仅 0.0376 px/10ms。该量是输入诊断，不能单独证明全部事件无用；但它暴露了之前覆盖检查只看跨轨迹位置、漏看窗内运动的不足。R9 的固定预算训练保留，真实自动评分被主动关闭。R10 改为相机朝向固定、沿画面两个方向平移；物理生成前用共享投影规则检查完整可见性与中心位移>1px/10ms，再核对完整 bank 的事件图和逐窗计数。两组共用这一纠正，没有改变 simulator core、训练预算或定位门槛。R10 仍从共同 R6 边界开始，不使用 R9 的中途模型；本报告不把两批数据当作等规模控制消融。

几何生成预检曾拦住一个裁切视角，原因是 CAD 原点不等于几何中心。生成前已改为绕变换后的几何中心取景，并以物理投影作统一完整可见性检查；失败的部分 bank 和日志保留，未进入训练。最终逐轨迹网格轮廓投影最大误差 {B['max_geometry_error_px']:.4f} px。

## 训练协议和证据

- 实际设备：g1 服务器 CPU，两组并行、每组 8 线程；GPU 当时均有他人进程，没有占用或中止他人作业。使用完整原架构、float32 和 416×416 输入，不能说成缩小模型的本地 GPU 实验。
- 两组分别从 R6 的合成专用 checkpoint 开始。其共同计划历史包括初次 496 次全网络更新、累计 20,000 次输出层更新、400 次全网络原目标更新、共同中心分类适配与 200 次全网络更新；R7/R9 的额外续训没有作为本轮起点。它是多次诊断之后的短训适配结果，不能称为从随机初始化仅训练了本轮次数。初始 checkpoint 路径和哈希记录在 protocol.json。
- 本轮每组追加 {cfg['optimizer_updates_per_arm']:,} 次全网络 Adam 更新，batch4、初始学习率 1e-4、30 步预热与余弦衰减，weight decay 1e-5、梯度裁剪 100；相同采样序列和仿射增强。两组预算与选择规则相同。BN running buffers 只在合成训练时更新；验证与真实推理固定 model.eval。
- 训练目标为原关键点坐标 MSE 加 13×13 中心分类交叉熵；属于共同训练适配，未改变网络架构，不等于原 SingleShotPose 完整训练复现。原始 logits argmax 选网格，避免 sigmoid 浮点饱和并列。
- 在固定追加更新 0/100/…/{cfg['optimizer_updates_per_arm']} 上按相同合成定位规则选权重，之后冻结哈希。真实测试没有选 checkpoint、调参或真实微调。旧 v1 模型作为同片段参照，不是等总预算消融。
- EventGenesis 使用 frozen core e79e656ac00f7847b98db9e12bf43b093f3afff1，27 个源文件字节哈希再次核对。软件渲染采用私有 OSMesa/llvmpipe，浮点 FBO 的 2.5 亮度读回检查通过；仅使用进程级 framebuffer 兼容开关，未修改安装环境或 simulator core。软件渲染没有阴影和反射，两组共享相同物理渲染结果。
- 本轮物理渲染为 1,000 Hz。V2E 调用锁定官方 EventEmulator，先将同一物理端点映射的 8 位亮度作线性 DN 插值，再以 25 µs 更新内部传感器，在 CPU 上工作，不使用 SuperSlomo。预检已拒绝直接以 1 ms 更新传感器的方案，因为滤波更新系数约 0.95，步长偏粗；25 µs 检查通过。两组阈值/噪声标称设置一致。这是高帧率物理端点加线性插值的配对初测变体，不能混同于之前 100 Hz+SuperSlomo 的设置，也不用于跨设备生成效率比较。
- 真实输入仅为完整去畸变事件图三通道；不输入 GT crop/mask/box、GT 网格或 GT 解法选择。公开 CAD 和相机内参只用于固定 EPNP；GT 位姿只在预测完成后算指标。没有读取有几何错误的官方 keypointprojections 字段作为监督。
- 独立进程重新加载 checkpoint，使用独立 NumPy 网格解码复算合成结果；核对两组所有采样、增强、标签与时间端点，以及有限损失、梯度、Adam step 和恢复前向。真实评分还核对原始 HDF5 事件、MAT 时间因果性、固定样本索引和独立指标复算。

## 组会展示与结论边界

先展示合成场景和两组事件图，再展示独立合成验证表，最后放真实预测视频及真实指标表。视频依次包含全部 200 个预先固定的窗口，以30fps顺序播放，不声称与物理时间严格同步；海报固定为 MAT0153，未选取最好看的帧。绿色是真值，红/蓝为固定PnP得到的预测姿态投影；角点误差按PnP之前的原始网络2D预测统计。超出画面的预测仅在显示时被画布裁切，指标使用原始未裁切坐标。事件表示的精确公式为各通道 min(log1p(count)/log(9),1)，其中计数尺度参数为8，并非直接除以8。

这是一个物体、一个 seed、一个真实片段的初测。位置、角点和 ADD-S 需分别解释；成功率弱时应明确说还未形成可靠 6DoF 能力。不能据此宣布仿真器普遍优于 V2E。下一轮需要独立重复训练、更多对象和预留真实序列，并在更成熟的固定协议下比较生成器。

本轮之后 r_01_gl 和 r_05_gl 均登记为已见，后续不可利用它们的生成器排名调参并继续称作最终测试。新确认实验需保留其他序列。服务器产物：{manifest['server_run']}；大 checkpoint 与物理 bank 留在服务器，证据目录保留路径、哈希和原始逐窗结果。

参考：[E-POSE 论文](https://www.nature.com/articles/s41597-025-04536-5)、[官方 YCB 资产](https://ycb-benchmarks.s3.amazonaws.com/index.html)、[SingleShotPose](https://github.com/microsoft/singleshotpose)。
'''
(O/'EPOSE_服务器短训_核验报告.md').write_text(report,encoding='utf-8')
spoken=[]
for a,label in [('eventgenesis','EventGenesis'),('v2e','V2E')]:
 if a in rows:
  r=results[a];add_phrase='PnP未产生有效姿态，无法给出姿态均值' if r['adds_mean_over_valid_pnp_m'] is None else f"ADD-S 平均误差是 {number(r['adds_mean_over_valid_pnp_m'],100)} 厘米";spoken.append(f"{label} 的真实角点误差是 {number(r['corner_epe_all_samples_px'])} 像素，{add_phrase}，成功数为 {r['adds_success_0_1_d_count']}/200。")
zero_note='严格位姿成功率本次仍为零，不能把合成定位改善说成真实六自由度位姿已经解决。' if all(results[a]['adds_success_0_1_d_count']==0 for a in rows) else '严格成功率本轮仅为0%和1%，真实六自由度精度仍不足；合成定位改善没有直接等于跨域位姿成功。'
speech='''这一页展示的是物体位姿的真实初测。两组使用同一个木块模型、相同的物理轨迹和标签，训练输入只有合成事件，最后在真实E-POSE事件上测试。

此前取景方式让相机一直跟着物体，画面里的物体运动太弱。本轮改成相机朝向固定的平移，确认事件图中能清楚看到物体，再做两组等预算短训。两组都达到了事先登记的合成定位条件，之后才冻结权重、开启真实测试。

视频中绿色是真值，红色和蓝色分别是两组预测。全部200个固定窗口顺序播放，没有挑选最好的片段。

'''+ '\n'.join(spoken)+'\n\n'+zero_note+'''

这还是单物体、单片段、单种子的短训初测。我们目前展示实际预测和完整统计，下一步再扩充视角、对象与背景，在预留真实序列上确认结果。

如果问到算力：本次是在g1服务器CPU上训练完整网络；GPU当时有其他作业占用。两组使用相同预算，不能说成本地GPU训练，也不能说是从零只训练了本轮次数。
'''
(O/'组会讲稿_EPOSE更新.txt').write_text(speech,encoding='utf-8');print(json.dumps({'output':str(O),'verified_evidence_files':len(manifest['files']),'results':results},ensure_ascii=False,indent=2))
