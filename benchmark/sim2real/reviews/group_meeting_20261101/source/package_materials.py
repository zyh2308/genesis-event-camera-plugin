import csv,json,hashlib,subprocess,shutil,zipfile,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/Sim2Real_20261101';BUILD=Path(__file__).parent
REPO=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real')
if not REPO.exists():REPO=Path(r'\\wsl.localhost\Ubuntu-22.04\home\科研\Eventbased_WAM\repositories\gec-sim2real')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
d=json.loads((OUT/'results.json').read_text())
files=(BUILD/'tracked_files.txt').read_text(encoding='utf-8-sig').splitlines()
paths=[
 'benchmark/sim2real/scripts/train_e2vid_stage5.py',
 'benchmark/sim2real/scripts/build_e2vid_stage4_10ms_data.py',
 'benchmark/sim2real/scripts/infer_hqf_e2vid_candidate.py',
 'benchmark/sim2real/scripts/score_hqf14_reconstruction.py',
 'benchmark/sim2real/scripts/run_hqf14_reconstruction_eval.py',
 'benchmark/sim2real/scripts/evaluate_synthetic_reconstruction_fixture.py',
 'benchmark/sim2real/scripts/generate_stage8_replicad_dynamic_flow_pilot.py',
 'benchmark/sim2real/scripts/build_stage8_dynamic_flow_bank.py',
 'benchmark/sim2real/scripts/run_stage8_replicad_dynamic_gtfit.py',
 'benchmark/sim2real/scripts/stage5_flow_supervised_pilot.py',
 'benchmark/sim2real/compat/steflow_device_placement.py',
 'benchmark/sim2real/scripts/run_pose_stage2_pilot.py',
 'benchmark/sim2real/scripts/audit_epose_projection_domains.py',
 'benchmark/sim2real/protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json',
 'benchmark/sim2real/configs/pose_stage4_seed0_eval_stable.json',
]
inventory={'repository':'https://github.com/zyh2308/genesis-event-camera-plugin','branch':'exp/public-sim2real-v1','source_head':'e831751edcd65681f42dcc030b4b24e51ccec52c','frozen_simulator_core':'e79e656ac00f7847b98db9e12bf43b093f3afff1','tracked_file_count':len(files),'sim2real_tracked_file_count':sum(p.startswith('benchmark/sim2real/') for p in files),'source_files':[]}
for rel in paths:
 p=REPO/rel
 if p.exists():
  src=p.read_text(encoding='utf-8');inventory['source_files'].append({'path':rel,'bytes':p.stat().st_size,'lines':len(src.splitlines()),'sha256':sha(p)})
(OUT/'provenance/repository_source_inventory.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(BUILD/'tracked_files.txt',OUT/'provenance/repository_tracked_files.txt')
# Export reproducible scripts without build logs/private QA reports.
srcdir=OUT/'source';srcdir.mkdir(exist_ok=True)
for name in ('export_demos_remote.py','make_figures.py','build_deck.mjs','package_materials.py'):
 shutil.copyfile(BUILD/name,srcdir/name)
audit=f'''# 三线组会材料：来源与解释

核验日期：2026-11-01。完整仓库索引：{len(files):,} 个跟踪文件，其中 Sim-to-Real {inventory['sim2real_tracked_file_count']:,} 个。源分支 `exp/public-sim2real-v1`，HEAD `e831751edcd65681f42dcc030b4b24e51ccec52c`。

## 审阅范围

对完整仓库建立文件索引，并沿三条实验的关键链路核对场景/标签生成、事件表示、训练目标、模型推理、checkpoint绑定、评价和报告。重点源码的路径、行数与SHA256在 `provenance/repository_source_inventory.json`。这不等于对每个历史证据文件逐行重审；也不是重跑全部训练。本次没有修改 simulator core、训练协议或模型参数，没有重新选择checkpoint。

## 1. 重建：已产生真实测试结果

- 数据为 **HQF14**，不能写作 HS-ERGB；每组15,498帧。模型为E2VID+，分别用EventGenesis/V2E合成事件训练。
- 从逐序列CSV独立复算四项宏平均，与原结果一致。EG的LPIPS为0.570738，V2E为0.582220；SSIM为0.255283/0.250035；PSNR为11.634839/12.074158 dB；MSE为0.086147/0.080667。
- LPIPS：13/14序列EG更低；SSIM：9/14更高。PSNR和MSE：13/14序列V2E更好。没有多seed统计显著性结论。
- demo两个连续片段从每条序列25%位置开始，各至多8秒，选择规则没有使用分数。使用真实时间戳选择已有预测帧，采用固定中心裁剪。输出统一[0,1]显示，无逐图亮度拟合，无输出修复。**两组都有明显伪影与对比度不足，不应宣称重建质量已很好。**
- 全部14序列仍用于指标；demo片段不能替代全量指标。逐帧路径和哈希在export manifest。

## 2. 光流：最新的丰富场景诊断，尚非真实迁移结果

- 使用ReplicaCAD布局sc0训练、sc2验证，各只有一个30ms窗口，4个辐亮度端点；最终100次更新，GT监督诊断，seed0。不是原45epoch自监督实验，也不是MVSEC真实测试。两个布局属于同一公寓场景家族，不代表广泛跨场景泛化。
- 本次从两组最终checkpoint独立CPU推理，EPE与保存报告误差低于1e-5；完整数组保存在 `provenance/flow_predictions_and_gt.npz`。没有调参或挑最佳步数。
- 全有效区EPE：EG 2.254431，V2E 2.608889，零流3.497898；共同事件活跃区：EG 2.637690，V2E 2.599874，零流2.715453 px/30ms。
- 方向余弦约0.987/0.993，活跃区预测幅值约5.13/5.06，而GT平均约2.72。这更像是已学到方向、幅值仍偏大，不能归因某个生成器，也不能认为问题已完全解决。
- 有效区74,496像素；共同事件活跃并集7,871，占10.57%。**几乎全图GT都有运动；事件不活跃并不意味着物体或相机静止。**
- 原预览contact sheet的sc2 V2E事件数标题陈旧。本次根据原HDF5的微秒时间戳转秒并取(0,30ms]，核验为13,871；EventGenesis为31,019。demo据此重新生成。事件数本身不作为质量指标。
- 场景视频仅把真实保存的0/10/20/30ms四个端点放慢展示，没有伪造连续轨迹。GT、两组预测使用同一最大7.047 px/30ms的颜色亮度范围；误差图统一0–8 px、越亮误差越大。GT无效像素显示为暗灰。

## 3. 位姿：真实任务展示与合成结果分开

- E-POSE demo展示真实 `block/r_01_gl` 样本1–600的全画面事件及**真值**CAD框。没有可用的真实模型预测或ADD/ADD-S成绩，不能把绿色框叫预测。
- 使用匹配木块CAD、MAT `pose`、K和原始畸变参数，逐角点用cv2.projectPoints投影到原始畸变像素域；没有使用已发现投影有误的MAT `keypointprojections` 作为正确UV。真值仅用于展示。
- 另一个独立结果是通用0.16m立方体的合成pilot：4个验证窗口/2个rollout，普通eval、置信度最大位置选点，2D角点EPE为EG 31.583653、V2E 23.176161 px。**它不是匹配E-POSE木块的模型结果。**
- 没有将GT cell decode的privileged诊断当作模型成绩，也没有将PnP成功返回当作位姿准确率。

## 汇报中可以说什么

可以说：第一组真实重建迁移结果已经得到，感知指标有小幅优势但像素指标和画质仍需改进；丰富运动场景的光流诊断已超过零流并暴露幅值偏差；真实物体状态任务的样貌已能展示。

不能说：三条真实任务已经全部验证成功、EventGenesis全面优于V2E、光流已在MVSEC胜出、位姿已在E-POSE达到精度要求。现有证据不足以支持这些结论。

## 材料使用

- 先打开 `index.html`，可全屏播放三段MP4并查看图表；完全离线，无外部依赖。
- PPT为8页。数值表和三张统计图为原生可编辑对象，含内嵌Excel数据。图中数值为显示精度，完整精度保留在JSON/CSV。
- PPT第2/4/6页底部有视频文件链接。保留同目录结构；也可用浏览器或播放器单独播放。**MP4为独立文件，没有嵌入PPT。**
- `figure_*.svg` 的文字与图形可在支持SVG的工具中编辑；同时附PNG/PDF，便于论文或PPT排版。
- 素材与源数据溯源见 `provenance`；脚本见 `source`。未把大权重、完整数据集放入展示包。
'''
(OUT/'SOURCE_AUDIT.md').write_text(audit,encoding='utf-8')
talk='''# 组会讲稿（约5分钟，播放视频另计）

开场：我们想验证的是事件生成器本身的价值。相同物理轨迹、相同下游模型、相同合成训练预算，只替换事件生成方式，最终用公共真实事件数据检验。

重建：先播放真实HQF对照。左边是事件和参考图，右边两列是分别由两种合成数据训练出来的模型。当前两组都有明显伪影，不回避这一点。全14序列上，EventGenesis的LPIPS和SSIM略好；V2E的PSNR和MSE更好。这是第一组真正的迁移结果，但还不能讲成全面优势。

光流：先播放场景和预测对照。我们已把过于简单的场景换成有纹理、相机运动和独立物体运动的室内布局。新短训在留出布局上的全有效区误差低于零流，说明网络并非只能输出静止。不过活跃区位移幅值仍偏大，两个生成器谁更好也随评价区域变化。这个结果只说明诊断取得了运动信号，接下来必须用多方向、多布局数据和固定MVSEC协议检验真实泛化。

位姿：播放真实事件与绿色CAD框，明确告诉老师绿色是标注真值，用来说明我们希望恢复什么物体状态。已有的模型数字来自另一个通用立方体合成pilot，不能冒充真实E-POSE成绩。后续关键交付是匹配木块CAD的synthetic-only模型，在真实E-POSE输出可信的6D位姿及ADD/ADD-S。

收束：三项实验的样貌和评价方式已经具象化。现在最扎实的证据是真实重建配对结果；最需要集中的科学工作是把丰富场景中的运动信号转化为稳定真实迁移收益，补齐位姿的真实结果，再做独立seed和更多生成器比较。
'''
(OUT/'组会讲稿.md').write_text(talk,encoding='utf-8')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>EventGenesis · 三线实验</title><style>
:root{color-scheme:light;--ink:#14283e;--muted:#60758a;--blue:#2580cb;--line:#d9e3ec}*{box-sizing:border-box}body{margin:0;background:#f2f6fa;color:var(--ink);font-family:"Microsoft YaHei",system-ui,sans-serif}header{padding:48px max(24px,calc((100vw - 1200px)/2));background:#12243a;color:#fff}header p{max-width:1000px;color:#bfcddd;line-height:1.8}h1{font-size:36px;margin:0 0 18px}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:var(--blue)}header a{display:inline-block;color:#fff;text-decoration:none;border:1px solid #496279;border-radius:6px;padding:10px 16px}main{max-width:1240px;padding:26px 20px 60px;margin:auto}section{background:#fff;padding:28px;margin:0 0 28px;border:1px solid var(--line);border-radius:10px}h2{margin:0 0 10px;font-size:27px}.tag{display:inline-block;padding:5px 10px;background:#e9f3fc;color:#23649a;border-radius:4px;font-size:14px}.warm{background:#fff2de;color:#8c5717}.gray{background:#edf1f5;color:#5e6e7e}p{line-height:1.8}video{display:block;width:100%;background:#0d1727;border-radius:5px;margin-top:16px;max-height:74vh}.row{display:grid;grid-template-columns:1.3fr 1fr;gap:24px;align-items:start}img{width:100%;height:auto}.note{color:var(--muted);font-size:15px}table{width:100%;border-collapse:collapse;font-size:15px;margin:18px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid var(--line)}th{background:#edf3f8}.downloads{display:flex;gap:16px;flex-wrap:wrap}.claim{font-size:20px;font-weight:600}.warning{padding:14px 18px;background:#fff6e8;border-left:4px solid #e5962e}.download{margin-top:16px}@media(max-width:850px){.row{grid-template-columns:1fr}section{padding:18px}h1{font-size:29px}table{font-size:12px}td,th{padding:7px}}@media print{header,section{break-inside:avoid}video{display:none}}
</style><header><h1>EventGenesis · 从合成事件到真实任务</h1><p>相同物理轨迹、相同下游模型，只替换事件生成器。用 demo 展示实验样貌，用配对指标说明当前证据。2026.11.01 · seed 0 阶段结果</p><nav><a href="#reconstruction">L1 重建</a><a href="#flow">L2 光流</a><a href="#pose">L3 物体位姿</a><a href="EventGenesis_Sim2Real_组会汇报.pptx">下载可编辑 PPT</a><a href="result_tables.csv">结果 CSV</a></nav></header><main>
<section id="reconstruction"><span class="tag">已有真实测试</span><h2>L1 · 真实事件重建</h2><p>真实 HQF 事件 → E2VID+ → 灰度图像。两组权重均由合成数据训练和选定。</p><video controls preload="metadata" poster="reconstruction_boxes_poster.png"><source src="01_reconstruction_HQF.mp4" type="video/mp4"></video><p class="note">两个序列各8秒；同一时间戳、同一显示亮度、固定中心裁剪。没有修复输出或按得分筛选片段。</p><p class="claim">LPIPS、SSIM 略好；PSNR、MSE 较差。两组都有明显画面伪影。</p><table><thead><tr><th>生成器</th><th>LPIPS ↓</th><th>SSIM ↑</th><th>PSNR ↑</th><th>MSE ↓</th></tr></thead><tbody id="reconRows"></tbody></table><div class="row"><img src="figure_01_reconstruction_LPIPS.png" alt="HQF14逐序列LPIPS差异"><div><p>全14序列、每组15,498帧；等权序列平均。</p><p>13/14序列的LPIPS更低，9/14序列SSIM更高；13/14序列V2E的PSNR、MSE更好。</p><p class="note">没有多seed显著性结论；真实测试结果不能反过来用于选checkpoint或调参。当前数据集是HQF，HS-ERGB尚无正式结果。</p><a href="reconstruction_all14_sequences.csv">全部序列原始数值</a></div></div></section>
<section id="flow"><span class="tag warm">合成留出布局诊断</span><h2>L2 · 丰富室内场景中的事件光流</h2><p>共享Genesis轨迹 → 两种事件输入 → STE-FlowNet。图像、真值光流与两组模型预测对照。</p><video controls preload="metadata" poster="flow_scene_poster.png"><source src="02_flow_ReplicaCAD.mp4" type="video/mp4"></video><p class="note">仅4个真实保存的辐亮度端点，覆盖30ms，放慢显示，没有插帧或伪造连续场景视频。最终update100、合成GT监督诊断；1个训练窗口、1个留出布局窗口。</p><div class="row"><img src="figure_02_flow_EPE.png" alt="光流EPE与零流基线对比"><div><p class="claim">已出现运动响应；活跃区的位移幅值仍偏大。</p><p>全有效区EPE：2.254 / 2.609，零流3.498。共同事件活跃区：2.638 / 2.600，零流2.715 px/30ms。</p><p>事件活跃区占10.57%；GT幅值约2.72，预测约5.13 / 5.06。相机运动使几乎全图都有GT位移，事件不活跃不等于静止。</p><p class="warning">这是小规模合成诊断，尚无真实MVSEC成绩，不能据此宣布生成器排名。</p></div></div></section>
<section id="pose"><span class="tag gray">真实任务样例 + 独立合成 pilot</span><h2>L3 · 物体6DoF位姿</h2><p>真实E-POSE事件与匹配木块CAD的6D真值投影。绿色框表示我们希望模型估计的物体状态。</p><video controls preload="metadata" poster="pose_real_task_poster.png"><source src="03_pose_EPOSE_task.mp4" type="video/mp4"></video><p class="warning"><strong>绿色框为真值，不是模型预测。</strong> 尚未产生真实E-POSE的ADD / ADD-S、旋转／平移误差或成功率。</p><div class="row"><img src="figure_03_pose_pilot.png" alt="独立通用立方体合成pilot角点误差"><div><p class="claim">已有数字来自通用立方体合成pilot。</p><p>4个留出窗口／2个rollout，普通eval、置信度最大位置选点：EG角点EPE 31.58 px，V2E 23.18 px。</p><p>通用立方体不是E-POSE木块。这批pilot中V2E误差更低；PnP能返回解也不代表位姿准确。</p></div></div></section>
<section><h2>可带走的文件</h2><div class="downloads"><a href="EventGenesis_Sim2Real_组会汇报.pptx">8页可编辑PPT</a><a href="result_tables.csv">结果表CSV</a><a href="results.json">完整精度JSON</a><a href="SOURCE_AUDIT.md">来源与结论边界</a><a href="组会讲稿.md">5分钟讲稿</a></div><p class="note">统计图另附SVG、PNG、PDF；PPT图表包含内嵌Excel数据。PPT视频是同目录独立MP4，请保留文件结构。本页面不联网、不上传数据。</p></section></main><script>
const DATA=__DATA__;
document.getElementById('reconRows').innerHTML=['eventgenesis','v2e'].map(a=>{const r=DATA.reconstruction.metrics[a];return `<tr><td>${a==='eventgenesis'?'EventGenesis':'V2E'}</td><td>${r.lpips_alex.toFixed(4)}</td><td>${r.ssim.toFixed(4)}</td><td>${r.psnr_db.toFixed(2)}</td><td>${r.mse.toFixed(4)}</td></tr>`}).join('');
document.querySelectorAll('video').forEach(v=>v.addEventListener('play',()=>document.querySelectorAll('video').forEach(o=>{if(o!==v)o.pause()})));
</script></html>'''
(OUT/'index.html').write_text(html.replace('__DATA__',json.dumps(d,ensure_ascii=False)),encoding='utf-8')
(OUT/'README_使用说明.txt').write_text('优先双击 index.html，以浏览器离线播放三段demo和查看图表。\nPPT为8页，表格和统计图原生可编辑；第2/4/6页底部有独立MP4链接。请保留同目录文件。\n三类证据范围不同：重建是真实HQF结果；光流是小规模合成诊断；位姿视频是真值任务示例。\nSOURCE_AUDIT.md解释来源、显示规则和结论边界；组会讲稿.md可直接参考。\n',encoding='utf-8')
print(json.dumps({'tracked_files':len(files),'review_sources':len(inventory['source_files']),'output':str(OUT)},ensure_ascii=False))
