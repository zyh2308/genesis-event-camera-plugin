import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const root='C:/Users/zhaoyuhan/Documents/ChatGPT/New project';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/outputs/epose_diverse_short_20261101/EventGenesis_组会汇报_EPOSE初测更新版.pptx'));
const n=Number(process.argv[2]);const png=await p.export({slide:p.slides.items[n-1],format:'png',scale:1});await fs.writeFile(root+`/artifacts/epose_short_server/ppt_build/final_slide${n}.png`,new Uint8Array(await png.arrayBuffer()));
