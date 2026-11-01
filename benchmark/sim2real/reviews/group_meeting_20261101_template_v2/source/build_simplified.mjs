import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const WORK=path.resolve('../..'),BUILD=path.join(WORK,'artifacts/meeting_template_v2'),OUT=path.join(WORK,'outputs/Sim2Real_20261101_模板精简版');
const SKILL='C:/Users/zhaoyuhan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const ref=JSON.parse(await fs.readFile(path.join(BUILD,'template_reference.json'),'utf8'));
const data=JSON.parse(await fs.readFile(path.join(OUT,'results.json'),'utf8'));
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(BUILD,'template_base.pptx')));
const C={red:'#A00000',blue:'#3676A5',green:'#27805F',ink:'#263746',muted:'#617787',line:'#B7C9D5',pale:'#E7F1F6'};const FONT='微软雅黑';
function text(s,str,x,y,w,h,size=24,color=C.ink,bold=false){let q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=str;q.text.style={typeface:FONT,fontSize:size,color,bold,insets:0};return q;}
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
function header(i,title){const s=p.slides.items[i];for(const q of s.shapes.items){const f=q.frame;if(f.top<80&&f.left>80){q.text=title;q.text.style={typeface:'Calibri',fontSize:30.6667,color:C.red,bold:true};}else if(f.top<80&&f.left<80){q.text=String(i+1).padStart(2,'0');q.text.style={typeface:'Calibri',fontSize:18.6667,color:C.blue,bold:true};}else if(f.top>670){q.text=String(i+1);q.text.style={typeface:'Calibri',fontSize:10.6667,color:C.red,alignment:'right'};}}return s;}
function takeaway(s,str){rect(s,44,635,1192,36,C.red);text(s,str,64,643,1152,27,21,'#FFFFFF',true);}
function note(s,src){s.speakerNotes.text=src;}
async function img(s,name,x,y,w,h){s.images.add({blob:await fs.readFile(path.join(OUT,name)),contentType:'image/png',alt:name,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
function table(s,values,x,y,w,h,widths){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,values,columnWidths:widths});t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){let cell=t.getCell(r,c);cell.fill=r===0?C.pale:'white';cell.text.style={typeface:FONT,fontSize:24,color:r===0?C.blue:C.ink,bold:r===0||c===0};}return t;}
function chart(s,categories,series,position,{horizontal=false,min=0,max,fmt='0.00',legend=true}={}){const q=s.charts.add('bar',{position,categories,series:series.map(x=>({...x,values:x.values.map(v=>+v.toFixed(6))})),barOptions:{direction:horizontal?'bar':'column',grouping:'clustered',gapWidth:80},hasLegend:legend,legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:C.ink}},xAxis:{tickLabelPosition:'low',textStyle:{fontSize:horizontal?17:23,fill:C.muted},line:{fill:C.line,width:1},majorGridlines:null},yAxis:{min,max,numberFormatCode:fmt,textStyle:{fontSize:21,fill:C.muted},majorGridlines:{fill:C.line,width:1}},dataLabels:{showValue:!horizontal,position:'outEnd',numberFormatCode:fmt,textStyle:{fontSize:24,bold:true,fill:C.ink}},chartFill:'white',plotAreaFill:'white'});applyPresentationChartFont(q,{fontFamily:FONT});return q;}
function play(s,file){const a=text(s,'播放视频',1055,681,100,24,17,C.blue,true);a.text.get('播放视频').link={uri:file,isExternal:true};}
let s=header(0,'EventGenesis：公共真实数据验证');
text(s,'共享场景与轨迹  →  两种合成事件  →  相同模型  →  真实测试',80,156,1120,48,30,C.blue,true);
for(let i=0;i<3;i++){const x=62+i*405;const names=['图像重建','事件光流','物体位姿'];text(s,names[i],x,259,370,43,29,C.blue,true);await img(s,['reconstruction_boxes_panel2.png','flow_panel1.png','pose_panel1.png'][i],x,322,348,261);}
takeaway(s,'训练在仿真，测试在现实');
note(s,'共享Genesis轨迹/标签，合成数据训练与选checkpoint，真实事件最终测试。只更换事件生成器。当前重建为真实HQF14结果；光流为ReplicaCAD小规模合成诊断，MVSEC尚无成绩；位姿展示E-POSE真值任务示例。冻结core e79e656ac00f7847b98db9e12bf43b093f3afff1。原模板仅用于版式，不使用其旧实验内容。');
s=header(1,'图像重建：真实事件对照');
for(let i=0;i<4;i++){const x=48+i*300;text(s,['真实事件','参考图像','EventGenesis','V2E'][i],x,139,284,36,25,i===2?C.red:i===3?C.blue:C.ink,true);await img(s,`reconstruction_boxes_panel${i}.png`,x,184,282,212);await img(s,`reconstruction_crop_panel${i}.png`,x,435,282,170);}
text(s,'局部放大',48,405,250,25,19,C.muted);play(s,'01_reconstruction_HQF.mp4');takeaway(s,'HQF 真实输入；两组输出均有伪影');
note(s,'数据HQF，模型E2VID+，两组固定synthetic_best权重。视频16秒，boxes和poster_pillar_1各8秒，从序列25%处开始；相同时间戳、固定中心裁剪、统一0–1显示，未按指标筛片或调亮度。参考图仅展示和评分。不是HS-ERGB结果。来源和模型哈希见旧包SOURCE_AUDIT.md及provenance。');
s=header(2,'图像重建：HQF 测试结果');
const r=data.reconstruction.metrics;table(s,[['训练数据','LPIPS ↓','SSIM ↑','PSNR ↑','MSE ↓'],['EventGenesis',r.eventgenesis.lpips_alex.toFixed(4),r.eventgenesis.ssim.toFixed(4),r.eventgenesis.psnr_db.toFixed(2),r.eventgenesis.mse.toFixed(4)],['V2E',r.v2e.lpips_alex.toFixed(4),r.v2e.ssim.toFixed(4),r.v2e.psnr_db.toFixed(2),r.v2e.mse.toFixed(4)]],48,140,1184,168,[344,210,210,210,210]);
text(s,'优于 V2E 的序列数',52,330,760,30,21,C.muted);
chart(s,['LPIPS','SSIM','PSNR','MSE'],[{name:'序列数',values:[13,9,1,1],fill:C.red}],{left:45,top:375,width:795,height:240},{min:0,max:14,legend:false,fmt:'0'});
text(s,'13 / 14',906,377,300,65,50,C.red,true);text(s,'序列 LPIPS 更低',900,449,320,35,24,C.ink,true);text(s,'14 序列 · 单种子',902,526,320,30,22,C.muted);takeaway(s,'感知指标略好，像素误差仍偏高');
note(s,'HQF14，每组15498帧；各序列均值等权平均。EG LPIPS0.5707381395533369、SSIM0.2552825745644461、PSNR11.634839261823668、MSE0.08614728341404365；V2E对应0.5822195403288457、0.2500349422659423、12.074157567577435、0.08066656123013195。LPIPS13/14、SSIM9/14序列EG更好；PSNR/MSE13/14序列V2E更好。seed0，无显著性检验。中文序列名按原CSV顺序映射，原英文数据标识保留在CSV。');
s=header(3,'事件光流：场景与预测');
for(let i=0;i<3;i++){const x=44+405*i;text(s,['光流真值','EventGenesis','V2E'][i],x,137,385,35,26,i===1?C.red:i===2?C.blue:C.ink,true);await img(s,`flow_panel${i}.png`,x,184,382,287);text(s,['仿真场景','误差','误差'][i],x,488,380,31,21,C.muted);await img(s,`flow_small_panel${i}.png`,x,525,140,105);}
play(s,'02_flow_ReplicaCAD.mp4');takeaway(s,'合成诊断：单窗口训练 → 留出布局测试');
note(s,'ReplicaCAD sc0训练、sc2验证，各一个30ms区间，4个辐亮度端点。STE-FlowNet最终update100，合成GT监督诊断，seed0。不是MVSEC成绩，不是原45epoch自监督结果。GT/预测颜色范围统一0–7.047px/30ms；误差统一0–8px，越亮越大。两组模型输入仅事件。视频前半为4个端点慢放，非连续场景视频；后半为实际固定checkpoint预测。独立CPU复算EPE与报告差<1e-5。');
s=header(4,'事件光流：留出测试结果');
text(s,'EPE（像素 / 30 ms）↓',64,143,800,35,24,C.blue,true);
chart(s,['全有效区','事件活跃区'],[{name:'EventGenesis',values:data.flow.eventgenesis,fill:C.red},{name:'V2E',values:data.flow.v2e,fill:C.blue},{name:'零流',values:data.flow.zero,fill:'#9AA8B2'}],{left:54,top:200,width:855,height:405},{max:4.1,fmt:'0.000'});
text(s,'全有效区',972,227,250,35,25,C.blue,true);text(s,'误差降低',972,287,250,30,24,C.ink);text(s,'35.5%',960,336,280,70,50,C.red,true);text(s,'相对零流',974,414,250,30,22,C.muted);text(s,'单种子 · 合成诊断',960,530,270,30,21,C.muted);takeaway(s,'已出现运动响应，位移幅值仍偏大');
note(s,'全有效区EPE EG2.25443066/V2E2.60888874/零流3.49789798；共同事件活跃并集区EG2.63769029/V2E2.59987409/零流2.71545341。单区间不能支持广泛泛化或生成器排名。近全画面GT有运动，共同活跃并集7871/74496=10.57%，非活跃不等于静止。方向余弦0.987/0.993，预测活跃幅值5.13/5.06，GT约2.72。MVSEC未测试。');
s=header(5,'物体位姿：真实事件示例');
for(let i=0;i<2;i++){const x=62+i*615;text(s,['真实事件','位姿真值'][i],x,145,550,42,29,i===1?C.green:C.blue,true);await img(s,`pose_panel${i}.png`,x,204,540,405);}
play(s,'03_pose_EPOSE_task.mp4');takeaway(s,'E-POSE 真值示例，非模型预测');
note(s,'E-POSE block/r_01_gl样本1–600，连续三份约10ms事件积为一个显示窗口。绿色是GT框，不是预测。匹配木块CAD，用MAT pose、K、Brown-Conrady畸变逐点投影到原始畸变像素域；未使用有几何错误的keypointprojections。无真实模型推理或ADD/ADD-S结果。');
s=header(6,'物体位姿：合成预实验');
text(s,'角点误差（像素）↓',64,147,720,34,24,C.blue,true);
chart(s,['EventGenesis','V2E'],[{name:'角点误差',values:[data.pose.eventgenesis,data.pose.v2e],fill:C.red,points:[{idx:1,fill:C.blue}]}],{left:52,top:212,width:680,height:370},{max:40,legend:false});
table(s,[['真实 E-POSE','状态'],['ADD / ADD-S','未测试'],['平移 / 旋转误差','未测试']],800,228,428,227,[270,158]);text(s,'通用立方体 · 单种子',800,525,430,31,23,C.muted);takeaway(s,'V2E 角点误差更低；真实位姿结果待补');
note(s,'独立通用0.16m立方体pilot，非E-POSE匹配木块。31epoch、248updates每组，4个留出窗口/2rollouts，seed0，普通eval、训练集BN校准、置信度argmax选点。角点EPE31.583652973175/23.1761610507965。GT cell decode属于privileged诊断，未展示为推理结果；PnP成功求解不是准确率。');
s=header(7,'三项实验进展');
table(s,[['任务','现有证据','下一步'],['重建','HQF 真实对照','改善画质、多种子'],['光流','合成留出运动响应','多场景 → MVSEC'],['位姿','真实任务示例、合成预实验','匹配 CAD → E-POSE']],58,180,1160,359,[180,500,480]);takeaway(s,'目标：稳定、可重复的真实泛化收益');
note(s,'目前只有重建具备正式真实配对指标；光流仍为单窗口合成诊断；位姿真实框为真值展示。尚不能声称三类真实任务全面优于V2E。完整来源、统计范围和实现核验放在SOURCE_AUDIT.md，不挤占展示页。');
await fs.mkdir(path.join(BUILD,'renders'),{recursive:true});
for(let i=0;i<8;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(BUILD,'renders',`slide_${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
const candidate=path.join(BUILD,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:path.join(OUT,'EventGenesis_组会汇报_模板精简版.pptx'),pythonExecutable:'C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3','--require-native-table-slide','7','--require-native-table-slide','8'],requiredNativeTableOwnerSlides:[3,7,8],requiredNativeChartOwnerSlides:[3,5,7],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Calibri',FONT],referencePath:ref.path,referenceSha256:ref.sha256},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log('Completed template revision');
