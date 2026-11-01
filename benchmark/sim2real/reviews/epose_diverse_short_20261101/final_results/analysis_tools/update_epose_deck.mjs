import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const workspace='C:/Users/zhaoyuhan/Documents/ChatGPT/New project';
process.env.RUNTIME_NODE_MODULES='C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const output=path.resolve(process.argv[2]);
const source='C:/Users/zhaoyuhan/Desktop/EventGenesis_组会汇报_标题预留版.pptx';
const build=workspace+'/artifacts/epose_short_server/ppt_build';
const skill='C:/Users/zhaoyuhan/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const summary=JSON.parse(await fs.readFile(output+'/evidence/final/real_eval/summary.json','utf8'));
if(!JSON.parse(await fs.readFile(output+'/evidence/final/real_eval/independent_verification.json','utf8')).passed)throw Error('unverified real result');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
if(p.slides.items.length!==9)throw Error('source slide count changed');
const records=(await p.inspect({kind:'textbox,image,chart,table',maxChars:200000})).ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x));
function record(slideIndex,old){const list=records.filter(x=>x.slideIndex===slideIndex&&x.text===old);if(list.length!==1)throw Error('missing/ambiguous text '+old);return list[0];}
function replace(slideIndex,old,value){const rec=record(slideIndex,old),shape=p.resolve(rec.id);shape.text.replace(old,value);return shape;}
function typed(shape,text,fontSize,color,bold=false){shape.text=text.split('\n').map(line=>[{run:line,style:{typeface:'微软雅黑',fontSize:`${fontSize}px`,color,bold}}]);}
// Explicit CJK face prevents missing fallback glyphs in imported title runs.
for(const rec of records.filter(x=>x.name==='TextBox 2'&&x.kind==='textbox'&&x.slideIndex>0))typed(p.resolve(rec.id),rec.text,30.67,'#A00000',true);
const R=summary.results,eligible=summary.eligible_new_arms||['eventgenesis','v2e'];
const paired=eligible.length===2;
replace(1,'E-POSE · 真实任务示例','E-POSE · 真实初测');
replace(1,'真值示例；真实预测待补',paired?'合成定位通过；真实精度仍不足':'通过定位模型的真实初测');
typed(replace(6,'物体位姿：真实事件示例','物体位姿：真实迁移初测'),'物体位姿：真实迁移初测',30.67,'#A00000',true);
typed(replace(6,'真实事件','EventGenesis'),eligible.includes('eventgenesis')?'EventGenesis':'EventGenesis（未评分）',29,'#A00000',true);
typed(replace(6,'位姿真值','V2E'),eligible.includes('v2e')?'V2E':'V2E（未评分）',29,'#3676A5',true);
replace(6,'E-POSE 真值示例，非模型预测',paired?'真实事件输入，绿色为真值，红色和蓝色为实际预测':'仅展示通过合成定位门槛模型的真实预测，不作生成器对比');
const imageRecords=records.filter(x=>x.kind==='image'&&x.slideIndex===6).sort((a,b)=>a.position.left-b.position.left);
const overviewImage=p.resolve(records.filter(x=>x.kind==='image'&&x.slideIndex===1).sort((a,b)=>a.position.left-b.position.left)[2].id);
const overviewFrame=overviewImage.frame,overviewCrop=overviewImage.crop,overviewFit=overviewImage.fit;
const poster=await fs.readFile(output+`/ppt_panel_${eligible.includes('eventgenesis')?0:1}.png`);overviewImage.replace({blob:poster.buffer.slice(poster.byteOffset,poster.byteOffset+poster.byteLength),contentType:'image/png',alt:'真实 E-POSE 模型预测缩略图',...(overviewFit?{fit:overviewFit}:{})});overviewImage.frame=overviewFrame;overviewImage.crop=overviewCrop;
for(let i=0;i<2;i++){
 const image=p.resolve(imageRecords[i].id);const frame=image.frame,crop=image.crop,fit=image.fit;const bytes=await fs.readFile(output+`/ppt_panel_${i}.png`);
 image.replace({blob:bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),contentType:'image/png',alt:i===0?'EventGenesis 实际预测':'V2E 实际预测',...(fit?{fit}: {})});image.frame=frame;image.crop=crop;
}
const play=p.resolve(record(6,'播放视频').id);
const videoDir='C:/Users/zhaoyuhan/Desktop/EventGenesis_组会素材_20261101';
const videoPath=videoDir+'/EPOSE_真实预测_配对短训.mp4';
play.text=[[{run:'播放视频',style:{typeface:'微软雅黑',fontSize:'17px',color:'#3676A5',bold:true,underline:'sng'},link:{uri:pathToFileURL(videoPath).href,isExternal:true}}]];
for(const [i,file] of [[2,'01_reconstruction_HQF.mp4'],[4,'02_flow_ReplicaCAD.mp4']]){
 const target=p.resolve(record(i,'播放视频').id);target.text=[[{run:'播放视频',style:{typeface:'微软雅黑',fontSize:'17px',color:'#3676A5',bold:true,underline:'sng'},link:{uri:pathToFileURL(videoDir+'/'+file).href,isExternal:true}}]];
}
typed(replace(7,'物体位姿：合成预实验','物体位姿：真实初测'),'物体位姿：真实初测',30.67,'#A00000',true);
replace(7,'角点误差（像素）↓','真实角点误差（像素）↓');
const chart=p.resolve(records.find(x=>x.kind==='chart'&&x.slideIndex===7).id),series=chart.series.getItemAt(0);
series.values=['eventgenesis','v2e'].map(a=>R[a]?.corner_epe_all_samples_px??null);series.categories=['EventGenesis','V2E'];series.name='真实角点误差';chart.displayBlanksAs='gap';
const max=Math.max(...eligible.map(a=>R[a].corner_epe_all_samples_px));const yMax=Math.max(10,Math.ceil(max*1.25/10)*10);chart.yAxis={min:0,max:yMax,majorUnit:yMax/4,numberFormatCode:yMax>10000?'0.0E+00':'0.0'};
const table=p.resolve(records.find(x=>x.kind==='table'&&x.slideIndex===7).id);table.insertColumns({index:2});table.position={left:750,top:228,width:478,height:227};table.columnTracks=[{mode:'fixed',value:140},{mode:'fixed',value:185},{mode:'fr',value:1}];
const fmt=v=>v===null?'—':v>=1e4?v.toExponential(1):v.toFixed(2);
const cells=[['指标','EventGenesis','V2E'],['ADD-S\n（cm）',...['eventgenesis','v2e'].map(a=>R[a]?fmt(R[a].adds_mean_over_valid_pnp_m===null?null:100*R[a].adds_mean_over_valid_pnp_m):'未评分')],['成功率',...['eventgenesis','v2e'].map(a=>R[a]?(100*R[a].adds_success_0_1_d_fraction_all_samples).toFixed(1)+'%':'未评分')]];
for(let y=0;y<3;y++)for(let x=0;x<3;x++){
 const cell=table.getCell(y,x);cell.value=cells[y][x].split('\n').map(line=>[{run:line,style:{typeface:'微软雅黑',fontSize:'21px',bold:y===0||x===0,color:y===0?'#3676A5':'#263746'}}]);cell.fill=y===0?'#E7F1F6':'#FFFFFF';cell.text.style={fontSize:21,bold:y===0||x===0,verticalAlignment:'middle',alignment:'left'};
}
table.borders.assign({style:'solid',fill:'#B7C9D5',width:0.7});
const note=replace(7,'通用立方体 · 单种子','ADD-S 对有效 PnP 求均值\n成功率以全部 200 窗为分母');note.position={left:750,top:500,width:478,height:76};typed(note,'ADD-S 对有效 PnP 求均值\n成功率以全部 200 窗为分母',21,'#617787');
replace(7,'V2E 角点误差更低；真实位姿结果待补',paired?'真实六自由度精度仍不足：单物体、单片段、单种子的短训初测':'单组有效真实初测，另一组未通过合成定位门槛');
const plan=records.find(x=>x.slideIndex===8&&x.text?.startsWith('使用匹配 CAD 和正确投影生成合成训练标签'));
if(plan)typed(p.resolve(plan.id),'补足位置、尺度与画面截断覆盖，扩大独立合成轨迹。\n在预留真实序列上做多种子确认，报告 ADD-S 与位姿误差。',23,'#263746');
const notes='本轮 E-POSE 初测：'+summary.protocol_frozen_before_training.real_sequence+'，固定 MAT0003:3:0600 共200窗，单对象、seed0，合成专用训练。仅按合成验证选checkpoint，真实事件全图输入，无GT裁剪或选点。CPU服务器配对短训属于共同中心分类训练适配，不是原版训练协议复现。ADD-S阈值0.1d，均值按有效PnP，失败计入成功率分母。若仅一组通过定位，不作生成器比较。原始记录见共享仓库 ep ose_diverse_short_20261101。E-POSE：https://www.nature.com/articles/s41597-025-04536-5；YCB：https://ycb-benchmarks.s3.amazonaws.com/index.html';
for(const i of [6,7])p.slides.items[i].speakerNotes.text=notes.replace('ep ose','epose');
p.slides.items[1].speakerNotes.text='三条验证线：重建已完成HQF的14个真实序列，结果有得有失；光流展示丰富合成场景与独立留出诊断，尚无MVSEC真实分数；位姿在合成专用短训后对固定E-POSE片段做真实初测。演示和结果均区分合成与真实，不据单seed宣布生成器普遍优势。\n'+notes.replace('ep ose','epose');
p.slides.items[8].speakerNotes.text='后续优先扩充共享物理场景，保证物体运动信息充足；三条线使用相同下游模型、标签、预算和选择规则比较生成器。位姿扩充对象、视角、背景并在预留真实序列做多种子确认。已看过的真实片段登记为已见，后续不反复依其排名调参并称为最终测试。';
const candidate=build+'/candidate_epose_updated.pptx';await (await PresentationFile.exportPptx(p)).save(candidate);
await fs.writeFile(build+'/updated_inspection.ndjson',(await p.inspect({kind:'slide,textbox,image,chart,table,notes',maxChars:200000})).ndjson);
for(let i=0;i<9;i++){const png=await p.export({slide:p.slides.items[i],format:'png',scale:1});await fs.writeFile(`${build}/after_slide${i+1}.png`,new Uint8Array(await png.arrayBuffer()));}
const sourceSha=crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex');
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const final=output+'/EventGenesis_组会汇报_EPOSE初测更新版.pptx';
// The importer drops embedded-workbook bindings. Restore the real supplied
// workbooks; the first two remain byte-identical, and the pose workbook gets
// only the authored literal cell updates. No invented provenance or formulas.
const restored=build+'/candidate_epose_workbooks_restored.pptx';
execFileSync('C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',[workspace+'/artifacts/epose_short_server/restore_imported_chart_workbooks.py',source,candidate,restored],{stdio:'ignore'});
const receipt=await finalizePresentation({workspaceDir:workspace,candidatePath:restored,finalPath:final,pythonExecutable:'C:/Users/zhaoyuhan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','8'],explicitTotalSlideCount:9,requiredNativeTableOwnerSlides:[8],requiredNativeChartOwnerSlides:[4,6,8],requiredEmbeddedWorkbookChartOwnerSlides:[4,6,8],materializeLiteralChartWorkbooks:false,fontPolicy:{basis:'reference',families:['微软雅黑','Calibri'],referencePath:source,referenceSha256:sourceSha},verifyArtifactToolImport:true,receiptPath:build+'/final_validation.json'});
console.log(JSON.stringify({final,receipt}));
