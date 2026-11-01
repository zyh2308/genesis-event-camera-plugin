import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {PresentationFile,FileBlob} from '@oai/artifact-tool';
const WORK=path.resolve('../..'),BUILD=path.join(WORK,'artifacts/meeting_template_v3'),OLD=path.join(WORK,'outputs/Sim2Real_20261101_模板精简版'),OUT=path.join(WORK,'outputs/Sim2Real_20261101_标题预留版');
const SKILL='C:/Users/zhaoyuhan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(OLD,'EventGenesis_组会汇报_模板精简版.pptx')));
const C={red:'#A00000',blue:'#3676A5',ink:'#263746',muted:'#617787',pale:'#E7F1F6',green:'#27805F'},FONT='微软雅黑';
function text(s,str,x,y,w,h,size=24,color=C.ink,bold=false){let q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=str;q.text.style={typeface:FONT,fontSize:size,color,bold,insets:0};return q;}
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});}
const inspection=(await p.inspect({kind:'slide,shape,textbox,image,table,chart',maxChars:160000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
function clearBody(index){const ids=inspection.filter(q=>q.slideIndex===index&&q.position?.top>110&&q.position?.top<675&&/^(sh|im|tb|ch)\//.test(q.id)).map(q=>q.id);if(ids.length)p.delete(ids);}
function title(s,str){const q=s.shapes.items.find(q=>q.frame.top<80&&q.frame.left>80);q.text=str;q.text.style={typeface:'Calibri',fontSize:30.6667,color:C.red,bold:true};}
function takeaway(s,str){rect(s,44,635,1192,36,C.red);text(s,str,64,643,1152,27,21,'#FFFFFF',true);}
async function thumb(s,file,x,y,w,h,crop){s.images.add({blob:await fs.readFile(path.join(OLD,file)),contentType:'image/png',alt:file,fit:'contain',crop,position:{left:x,top:y,width:w,height:h}});}
const overview=p.slides.items[0],last=p.slides.items[7];clearBody(0);clearBody(7);
title(overview,'公共真实数据验证：三项任务');
text(overview,'共享物理轨迹与标签',62,142,1160,39,28,C.blue,true);
text(overview,'EventGenesis / V2E  →  相同模型、合成训练  →  真实测试',62,193,1160,42,27,C.ink);
const cards=[
 {title:'图像重建',scope:'HQF · 真实测试',image:'reconstruction_poster.png',status:'已完成两组真实对照',top:.105,bottom:.10},
 {title:'事件光流',scope:'ReplicaCAD · 合成留出诊断',image:'flow_result_poster.png',status:'已有运动响应，待 MVSEC 测试',top:.115,bottom:.08},
 {title:'物体位姿',scope:'E-POSE · 真实任务示例',image:'pose_poster.png',status:'真值示例；真实预测待补',top:.15,bottom:.17},
];
for(let i=0;i<3;i++){const a=cards[i],x=44+i*404;rect(overview,x,268,384,331,C.pale);rect(overview,x,268,384,4,[C.red,C.blue,C.green][i]);text(overview,a.title,x+18,289,348,42,29,C.blue,true);text(overview,a.scope,x+18,342,350,32,21,C.ink);await thumb(overview,a.image,x+14,391,356,157,{left:0,right:0,top:a.top,bottom:a.bottom});text(overview,a.status,x+18,558,352,29,20,C.muted);}
takeaway(overview,'训练在仿真，测试在现实');
overview.speakerNotes.text='三任务概览。EventGenesis/V2E共享Genesis物理轨迹和标签，固定下游模型与合成训练协议。当前重建已有HQF14真实配对结果；光流只完成ReplicaCAD小规模合成留出诊断，MVSEC未测试；位姿画面为E-POSE真值任务示例，非真实模型预测。概览沿用初版的三任务卡片结构，继续使用用户新模板母版、页眉和配色。';
title(last,'后续实验计划');
const plans=[
 ['重建','画质与结果稳定性','先在合成留出集检查伪影与亮度偏差，冻结训练方案；\n补充独立随机种子，检验真实重建收益是否稳定。'],
 ['光流','运动幅值与真实迁移','扩充纹理、场景布局和运动方向，重点校准位移幅值；\n通过合成留出验证后，执行固定 MVSEC 真实测试。'],
 ['位姿','匹配物体与真实位姿','使用匹配 CAD 和正确投影生成合成训练标签；\n完成训练后，在真实 E-POSE 上报告 ADD / ADD-S 及位姿误差。'],
];
for(let i=0;i<3;i++){const y=142+151*i;rect(last,44,y,1192,132,C.pale);text(last,plans[i][0],70,y+18,130,45,30,C.blue,true);text(last,plans[i][1],225,y+17,965,36,25,C.blue,true);text(last,plans[i][2],225,y+62,965,61,23,C.ink);}
takeaway(last,'统一训练预算与测试协议，补充多种子和更多生成器对照');
last.speakerNotes.text='后续计划不将现有小规模诊断当作最终结论。所有数据和训练改进只依据合成训练/留出验证；已看过的HQF不作未见验证或调参依据，新协议需独立记录。光流是近期重点：多布局、多方向、运动覆盖与幅值校准，再固定MVSEC真实测试。位姿需匹配CAD并使用逐点正确投影，禁止GT crop/mask/box或GT cell推理。最终统一预算、数据划分和评价协议，增加独立随机种子及其他事件生成器对比。';
// Add a truly blank first slide for the user's own title. Keep content numbering 01–08.
const cover=p.slides.add({background:{fill:'white'}});text(cover,'',90,230,1100,130,44,C.red,true);const order=(await p.inspect({kind:'slide',maxChars:20000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.kind==='slide');p.slides.reorder([order[8].id,...order.slice(0,8).map(x=>x.id)]);
await fs.mkdir(path.join(BUILD,'renders'),{recursive:true});
for(let i=0;i<9;i++){const b=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(path.join(BUILD,'renders',`slide_${i+1}.png`),new Uint8Array(await b.arrayBuffer()));}
const candidate=path.join(BUILD,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
execFileSync('C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',[path.join(BUILD,'preserve_chart_workbooks.py')]);
const ref=JSON.parse(await fs.readFile(path.join(WORK,'artifacts/meeting_template_v2/template_reference.json'),'utf8'));
await finalizePresentation({workspaceDir:WORK,candidatePath:candidate,finalPath:path.join(OUT,'EventGenesis_组会汇报_标题预留版.pptx'),pythonExecutable:'C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','4','--require-native-table-slide','8'],requiredNativeTableOwnerSlides:[4,8],requiredNativeChartOwnerSlides:[4,6,8],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'reference',families:['Calibri',FONT],referencePath:ref.path,referenceSha256:ref.sha256},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log('Revised deck: 9 slides');
