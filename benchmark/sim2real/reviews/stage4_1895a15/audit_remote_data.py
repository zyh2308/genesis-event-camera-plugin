from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_data_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=240)
(O/'data_audit_stdout.json').write_text(r.stdout);(O/'data_audit_stderr.txt').write_text(r.stderr)
if r.returncode:print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'tail':r.stdout[-1000:]}))
else:
    d=json.loads(r.stdout);print(json.dumps({'checked_at_JST':d['checked_at_JST'],'shared_bank':d['shared_bank'],'split_sha_match':d['split_sha_match'],'adapter_file_count':len(d['adapter_files']),'all_adapter_hashes_match':all(x['sha_match'] for x in d['adapter_files']),'paired_array_checks':len(d['paired_arrays']),'all_shared_arrays_equal':all(all(x['shared_array_equality'].values()) and x['physical_labels_match'] and x['endpoint_match'] for x in d['paired_arrays']),'progress':d['progress'],'paired_order':d['paired_sample_order_all_common_updates'],'RNG_api_check':d['nonCPU_RNG_state_rejected']},ensure_ascii=False))
