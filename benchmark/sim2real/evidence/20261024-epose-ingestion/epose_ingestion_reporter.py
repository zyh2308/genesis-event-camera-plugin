"""One-shot workflow reporter; stops when this import completes or fails."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,time,hashlib
ROOT=Path('/mnt/e/EPOSE_Downloads/server_transfer')
OUT=Path('/home/科研/Eventbased_WAM/repositories/gec-sim2real/benchmark/sim2real/evidence/20261024-epose-ingestion')
OUT.mkdir(parents=True,exist_ok=True)
BASE='/home/zhaoyuhan/Eventbased_WAM/data/real/epose/figshare_28106717_v1'
SSH=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','Hostname=10.129.161.238','-o','HostKeyAlias=g1.pkuml.org','zhaoyuhan@g1.pkuml.org']
def atomic(p,d):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
source_manifest=Path(__file__).with_name('epose_original_download_manifest.json')
if source_manifest.exists():(OUT/'original_download_manifest.json').write_bytes(source_manifest.read_bytes())
for source in ['epose_ingest_server.py','epose_unrar_recovery_server.py','epose_install_unrar.py','epose_transfer_parts.py','epose_ingestion_reporter.py']:
 p=Path(__file__).with_name(source)
 if p.exists():(OUT/source).write_bytes(p.read_bytes())
start=time.monotonic()
while time.monotonic()-start<24*3600:
 try:
  local=json.loads((ROOT/'transfer_status.json').read_text())
  r=subprocess.run(SSH+['cat '+BASE+'/manifests/ingestion_status.json'],capture_output=True,text=True,timeout=25)
  remote=json.loads(r.stdout) if r.returncode==0 else {'phase':'connection_error','error':r.stderr[-500:]}
  result={'schema':'epose_import_combined_status_v1','observed_utc':datetime.now(timezone.utc).isoformat(),'local_transfer':local,'server_extraction':remote,'server_base':BASE}
  name=local.get('archive')
  if name:
   log=ROOT/(Path(name).stem+'.rsync.log')
   if log.exists():
    with log.open('rb') as f:f.seek(max(0,log.stat().st_size-1024));tail=f.read().decode(errors='replace')
    rows=[x.strip() for x in tail.replace('\r','\n').splitlines() if x.strip()]
    result['last_transfer_progress']=rows[-1] if rows else None
  atomic(ROOT/'combined_status.json',result);atomic(OUT/'ingestion_live_status.json',result)
  receipt=None
  if remote.get('phase')=='complete':
   r=subprocess.run(SSH+['cat '+BASE+'/manifests/ingestion_receipt.json'],capture_output=True,text=True,timeout=25);r.check_returncode();receipt=json.loads(r.stdout)
   atomic(OUT/'ingestion_receipt.json',receipt);atomic(ROOT/'ingestion_receipt.json',receipt)
  state='导入完成' if receipt else ('解压失败，见状态证据' if remote.get('phase')=='failed' else '正在导入（尚未全部完成）')
  body=f'''# E-POSE 服务器导入\n\n状态：{state}。此页由本次一次性导入流程自动刷新；不是训练或真实评测结果。\n\n- 本地原始ZIP：`E:\\EPOSE_Downloads\\28106717.zip`，29,759,156,265 bytes。\n- ZIP SHA256：`81fd5ef872ad5551bd170b0451eb3df3e4536a7634202b431d322e05d60e5884`。\n- g1根目录：`{BASE}`。\n- 原始压缩包：`archives/*.rar`；数据：`extracted/<子包名>/`；逐文件完整性清单与收据：`manifests/`。\n- 优先导入original_data.rar；其RAR头清单列306个HDF5、21,744,552,016解压字节。其余17包补齐逐物体和多物体标注/派生数据。\n- 原始外层ZIP保留在E盘；以其18个原始RAR成员上传，先验证ZIP成员CRC，再验证本地/服务器子包SHA256，原始HDF5记录每文件SHA256；其余包由官方UNRAR校验CRC且逐文件核对大小，清单明确标记校验类型。\n- 实时状态：`ingestion_live_status.json`；全量完成后生成`ingestion_receipt.json`。\n- 当前上传子包：`{local.get('archive')}`；已上传{len(local.get('completed_archives',[]))}/18；服务器解压阶段：`{remote.get('phase')}`，已完成{len(remote.get('completed_archives',[]))}/18。\n- 服务器操作日志：`{BASE}/logs/unrar_recovery.log`；本地日志：`E:\\EPOSE_Downloads\\server_transfer\\supervisor.log`。\n\n本次只传输、完整性校验、解压和生成清单，没有修改模拟器或训练配置，也没有运行真实数据评分。后续先审核HDF5/MAT、K/位姿/时间/CAD对应；数据导入完成不等于位姿模型已通过可用性验证。原始发布树保持原样，不使用GT mask/segmented events作为模型输入。\n'''
  if receipt:body+=f'\n已完成文件数：{receipt["total_files"]}，解压总字节：{receipt["total_uncompressed_bytes"]}。\n'
  (OUT/'README.md').write_text(body)
  if receipt or remote.get('phase')=='failed' or local.get('phase')=='failed':break
 except Exception as e:
  atomic(ROOT/'reporter_last_error.json',{'error':repr(e),'utc':datetime.now(timezone.utc).isoformat()})
 time.sleep(45)
