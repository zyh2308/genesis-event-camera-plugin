import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const WORK=path.resolve('../..');
const BUILD=path.join(WORK,'artifacts/meeting_build');
const OUT=path.join(WORK,'outputs/Sim2Real_20261101');
const SKILL='C:/Users/zhaoyuhan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const PY='C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const data=JSON.parse(await fs.readFile(path.join(OUT,'results.json'),'utf8'));
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={navy:'#12243A',text:'#20364D',muted:'#62778B',blue:'#2580CB',orange:'#E5962E',green:'#168361',line:'#D9E2EB',light:'#F1F5F9'};
const FONT='Microsoft YaHei';
function box(s,x,y,w,h,fill=C.light){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function text(s,str,x,y,w,h,size=24,color=C.text,bold=false){let q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=str;q.text.style={typeface:FONT,fontSize:size,color,bold};return q;}
function slide(title,sub,tag){let s=P.slides.add();s.background.fill='white';box(s,0,0,12,720,C.blue);text(s,title,48,25,1180,58,38,C.navy,true);text(s,sub,50,87,1160,35,20,C.muted);box(s,48,672,1184,1,C.line);const footer=text(s,tag,50,682,1090,25,16,C.muted); const movie=tag.match(/[0-9]{2}_[A-Za-z_]+[.]mp4/); if(movie)footer.text.get(movie[0]).link={uri:movie[0],isExternal:true};text(s,String(P.slides.items?.length??''),1180,682,50,25,16,C.muted);return s;}
async function img(s,name,x,y,w,h){const bytes=await fs.readFile(path.join(OUT,name));return s.images.add({blob:bytes,contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,w,h,widths){let t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++){for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.navy:(r%2?C.light:'white');cell.text.style={typeface:FONT,fontSize:21,color:r===0?'#FFFFFF':C.text,bold:r===0||c===0};}}return t;}
function chart(s,categories,series,pos,{horizontal=false,max=undefined,min=0,legend=true,format='0.00'}={}){series=series.map(q=>({...q,values:q.values.map(v=>Number(v.toFixed(6)))}));let c=s.charts.add('bar',{position:pos,categories,series,barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:70},hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:19}},xAxis:{tickLabelPosition:'low',textStyle:{fontSize:horizontal?15:18,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min,max,numberFormatCode:format,textStyle:{fontSize:18,fill:C.muted},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:!horizontal,position:'outEnd',numberFormatCode:format,textStyle:{fontSize:19,fill:C.text}},chartFill:'white',plotAreaFill:'white'});applyPresentationChartFont(c,{fontFamily:FONT});return c;}
let s=slide('从合成事件到真实任务：三条验证线','核心问题：只更换事件生成器，是否能让相同下游模型更好地泛化到真实事件？','EventGenesis Sim-to-Real | 2026.11.01 | 单 seed 阶段结果');
text(s,'共享 Genesis 场景与轨迹',58,152,1150,42,30,C.navy,true);
text(s,'EventGenesis / V2E  →  相同模型、合成数据训练  →  公共真实事件测试',58,207,1150,43,27,C.text);
const cards=[['L1  图像重建','真实 HQF：14 序列','已有配对真实测试结果','reconstruction_boxes_poster.png',C.blue],['L2  事件光流','丰富场景：留出布局诊断','已出现运动响应，待真实测试','flow_result_poster.png',C.orange],['L3  物体 6DoF 位姿','真实 E-POSE：任务样例','正式真实预测与指标尚未完成','pose_real_task_poster.png',C.green]];
for(let i=0;i<3;i++){let x=48+i*400;box(s,x,286,384,338,C.light);box(s,x,286,384,5,cards[i][4]);text(s,cards[i][0],x+18,310,348,40,29,C.navy,true);text(s,cards[i][1],x+18,359,348,32,22,C.text);await img(s,cards[i][3],x+12,400,360,155);text(s,cards[i][2],x+18,576,350,31,20,C.muted);}
s.speakerNotes.text='总目标：证明事件生成器的迁移价值。三线证据成熟度不同，不将合成诊断与真实测试混为一谈。正式对比目前仅 EventGenesis/V2E，其余生成器未完成。Frozen core e79e656ac00f7847b98db9e12bf43b093f3afff1。数据源详见 SOURCE_AUDIT.md / provenance/export_manifest.json。';
s=slide('重建 demo：真实事件输入，观察恢复出的图像','同一真实事件流；两组 E2VID+ 均仅由各自合成事件训练。参考图像只用于展示与评价。','真实测试 | HQF boxes / poster_pillar_1 | 播放 01_reconstruction_HQF.mp4');
await img(s,'reconstruction_boxes_poster.png',32,126,1216,506);
text(s,'画面保留原始输出：两组均有块状伪影与对比度不足。',50,636,1180,30,22,C.text,true);
s.speakerNotes.text='播放同目录 01_reconstruction_HQF.mp4，16秒，两个序列各取从25%时间位置开始的8秒，按时间重采样显示，不做图像插帧、不按分数筛片。所有图像以固定0–1显示，无逐帧亮度拟合。其余12序列仍纳入定量指标。GT不进入网络。原始推理路径：20261025-hqf14-e2vid-seed0-actual-r2/inference。';
s=slide('真实重建结果：感知距离略好，像素误差仍有差距','HQF 全 14 序列、每组 15,498 帧；表中为各序列均值的等权平均。','真实测试 | 固定合成验证选定 checkpoint | seed 0 | 非显著性结论');
const rm=data.reconstruction.metrics;
table(s,[['训练数据生成器','LPIPS ↓','SSIM ↑','PSNR ↑','MSE ↓'],['EventGenesis',rm.eventgenesis.lpips_alex.toFixed(4),rm.eventgenesis.ssim.toFixed(4),rm.eventgenesis.psnr_db.toFixed(2),rm.eventgenesis.mse.toFixed(4)],['V2E',rm.v2e.lpips_alex.toFixed(4),rm.v2e.ssim.toFixed(4),rm.v2e.psnr_db.toFixed(2),rm.v2e.mse.toFixed(4)]],48,146,1184,164,[360,206,206,206,206]);
chart(s,data.reconstruction.sequences,[{name:'EG − V2E',values:data.reconstruction.lpips_delta,fill:C.blue,points:data.reconstruction.lpips_delta.map((v,i)=>({idx:i,fill:v<0?C.blue:C.orange}))}],{left:45,top:344,width:780,height:290},{horizontal:true,min:-.05,max:.06,legend:false,format:'0.00'});
text(s,'LPIPS：EG − V2E（越低越好）',60,321,720,29,20,C.muted);
text(s,'13 / 14',884,368,330,71,52,C.blue,true);text(s,'序列的 LPIPS 更低',884,441,330,36,25,C.navy,true);text(s,'SSIM：9 / 14 序列更高\nPSNR、MSE：13 / 14 序列 V2E 更好',884,505,330,91,22,C.text);
s.speakerNotes.text='精确来源：paired_score.json（SHA256 4fb0679ddcde1f4af3b2cbccf1af99fc0e193c9c51e809b171947b2dd36a3692），本材料从已审计 hqf14_per_sequence.csv 独立重算宏平均。LPIPS为AlexNet版本。此处没有多seed误差条，没有以帧数充当独立重复数。HQF是真实最终测试结果；原计划HS-ERGB尚无结果，不可改名。禁止根据HQF结果选checkpoint或调参。';
s=slide('光流 demo：用丰富室内场景检查模型是否真的学会运动','共享物理轨迹、GT 与 30 ms 时间窗；两组模型仅输入各自事件。预测与真值使用同一颜色尺度。','合成诊断 | ReplicaCAD sc0 → sc2 | 播放 02_flow_ReplicaCAD.mp4');
await img(s,'flow_result_poster.png',32,129,1216,510);
text(s,'当前只是 1 个训练窗口 → 1 个留出布局窗口；不是 MVSEC 迁移结果。',48,638,1184,29,22,C.text,true);
s.speakerNotes.text='视频前半展示同一物理轨迹和两组原始事件，后半比较GT、预测及EPE。只有4个保存的辐亮度端点（0/10/20/30ms），以静帧放慢展示，非连续场景视频、无插帧。最终update100固定checkpoint，GT监督合成诊断。独立CPU前向重算误差并核对记录，模型事件输入无GT。原contact sheet的sc2 V2E标题事件数陈旧，已用原始HDF5的微秒时间转换和(0,30ms]窗重新核验为13871。';
s=slide('光流结果：不再只预测零，但位移幅值仍需改善','最终 update 100；每组相同初始化和预算。EPE 越低越好，单位 px / 30 ms。','合成留出布局诊断 | seed 0 | 没有真实 MVSEC 成绩');
chart(s,['全有效区域','共同事件活跃区域'],[{name:'EventGenesis',values:data.flow.eventgenesis,fill:C.blue},{name:'V2E',values:data.flow.v2e,fill:C.orange},{name:'零流基线',values:data.flow.zero,fill:'#8491A0'}],{left:40,top:152,width:790,height:420},{max:4.15});
text(s,'全有效区域',870,167,335,35,25,C.navy,true);text(s,'EG 相对零流误差降低 35.5%\nV2E 相对零流降低 25.4%',870,215,330,94,23,C.text);
text(s,'事件活跃区域',870,340,335,35,25,C.navy,true);text(s,'EG 2.638；V2E 2.600\nGT 平均幅值约 2.72\n预测幅值约 5.13 / 5.06',870,389,330,120,23,C.text);
box(s,48,589,1184,62,C.light);text(s,'活跃区域仅覆盖 10.57% 的有效像素；方向已较一致，幅值仍偏大，不能宣布生成器优胜。',64,606,1152,35,23,C.navy,true);
s.speakerNotes.text='训练布局sc0和验证布局sc2均仅一个30ms区间，同一公寓场景家族，因此只是小规模诊断。验证有效像素74496，共同事件活跃并集7871；近全画面GT有运动（相机运动），不能称非活跃区是静止区。EG方向余弦0.987、V2E0.993。预测活跃幅值5.134/5.064，GT活跃平均2.715。图中展示全有效区与两组输入共同的事件活跃并集，避免每组各自挑掩膜。源run：20261101-stage8-replicad-dynamic-v3-cpucompat。';
s=slide('机器人任务 demo：真实事件中的物体 6DoF 位姿','E-POSE 木块全画面事件，与匹配 CAD、6D 标注、K 和畸变参数投影出的真值包围框。','任务示例，非模型预测 | 播放 03_pose_EPOSE_task.mp4');
await img(s,'pose_real_task_poster.png',32,128,1216,512);
text(s,'绿色框是 Ground Truth；真实 E-POSE 的模型预测和 ADD / ADD-S 仍待完成。',48,639,1184,30,23,C.green,true);
s.speakerNotes.text='此演示只说明目标任务样貌，不代表估计器已达成真实6DoF。E-POSE block/r_01_gl样本1–600，连续三份10ms事件合为一个显示窗口，约6秒真实片段。用MAT pose、原始K和Brown-Conrady畸变逐点projectPoints投影CAD AABB；没有使用已知有问题的keypointprojections作为正确UV。CAD SHA256 8c653731426cf73b5b5d4ffbfa3bdd4ebb14cb82d413da82559bce1b5ee3ab37。真实GT只用于画面展示，不进入模型。';
s=slide('位姿结果：已有合成 pilot，真实迁移成绩尚未产生','合成通用立方体，4 个留出窗口／2 个 rollout；置信度最大位置选点，普通 eval()，seed 0。','合成 pilot | 通用立方体不是 E-POSE 木块 CAD | 禁止将两者混为同一实验');
chart(s,['EventGenesis','V2E'],[{name:'2D corner EPE',values:[data.pose.eventgenesis,data.pose.v2e],fill:C.blue,points:[{idx:1,fill:C.orange}]}],{left:45,top:161,width:650,height:380},{max:40,legend:false});
text(s,'2D 角点平均误差 / px ↓',60,137,650,30,23,C.muted);
table(s,[['正式真实 E-POSE 指标','状态'],['ADD / ADD-S','未测试'],['平移／旋转误差','未测试'],['成功率','未测试']],745,176,484,271,[280,204]);
text(s,'这批 pilot 中 V2E 的角点误差更低。\nPnP 能求解，并不代表位姿准确。',745,482,468,101,24,C.text,true);
box(s,48,600,1184,54,C.light);text(s,'下一份论文证据：匹配木块 CAD 的 synthetic-only 模型，在真实 E-POSE 上输出位姿与精度。',63,613,1150,34,22,C.navy,true);
s.speakerNotes.text='来源：20261101-stage8-pose-cpu-short-v2/training_output/result.json；31epoch、248updates每组。只展示正式confidence-argmax的角点误差31.583652973175/23.1761610507965；GT cell decode属于privileged诊断，不可当推理成绩。没有E-POSE训练或真实ADD/ADD-S。两个柱来自同一序列，色彩与标签需准确：本图以方法类别标注数值，不将PnP success作为准确率。';
s=slide('本轮可汇报的成果，以及距离论文主结论的差距','以真实输出和配对数值呈现：有进展，也清楚说明结论能走到哪里。','结论边界 | 当前尚不足以宣称 EventGenesis 在三类真实任务上全面优于 V2E');
table(s,[['任务','现在可展示的研究成果','下一份关键证据'],['重建','HQF14 真实配对结果；LPIPS、SSIM略好','提高可见图像质量；补独立seed'],['光流','丰富场景已有留出运动响应；超过零流','多布局、多方向训练 → 固定 MVSEC 测试'],['位姿','真实事件 + CAD/6D任务展示；合成pilot','匹配CAD训练 → E-POSE真实位姿评价']],48,158,1184,319,[170,520,494]);
text(s,'建议汇报主线',52,516,1170,40,28,C.navy,true);
text(s,'“我们已经拿到第一组真实迁移结果，并用更有运动信息的场景推进光流；\n接下来要把这种信号变成稳定、可重复、跨真实数据集的收益。”',52,568,1160,80,27,C.text);
s.speakerNotes.text='讲述顺序：先播放三段demo，再用相应图表讲能支持的结论。不讲优化器/显存/SSH等工程记录。不要把目标当结论，也不要隐瞒V2E占优的指标。所有源数据、显示规则和证据范围详见SOURCE_AUDIT.md和provenance。';

await fs.mkdir(path.join(BUILD,'slide_renders'),{recursive:true});
for(let i=0;i<P.slides.items.length;i++){
 const b=await P.export({slide:P.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,'slide_renders',`slide_${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await b.arrayBuffer()));
}
const candidate=path.join(BUILD,'candidate.pptx');await (await PresentationFile.exportPptx(P)).save(candidate);
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:path.join(OUT,'EventGenesis_Sim2Real_组会汇报.pptx'),pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','7','--require-native-table-slide','8'],requiredNativeTableOwnerSlides:[3,7,8],requiredNativeChartOwnerSlides:[3,5,7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log('Slides exported and finalized');
