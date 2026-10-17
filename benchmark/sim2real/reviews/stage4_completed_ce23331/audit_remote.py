from pathlib import Path
import base64,json,shlex,subprocess
O=Path(__file__).resolve().parent
cmd=shlex.join(['env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1','/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python','-c','import base64,sys;exec(base64.b64decode(sys.stdin.buffer.read()))'])
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','ConnectionAttempts=1','-p','22','zhaoyuhan@10.129.163.68',cmd],input=base64.b64encode((O/'remote_program.py').read_bytes()).decode(),capture_output=True,text=True,timeout=240)
(O/'remote_verification.json').write_text(r.stdout);(O/'remote_stderr.txt').write_text(r.stderr)
if r.returncode:print(json.dumps({'exit':r.returncode,'stderr':r.stderr,'tail':r.stdout[-1000:]}))
else:
    d=json.loads(r.stdout);print(json.dumps({'checked_at_JST':d['checked_at_JST'],'head':d['head'],'status':d['status'],'campaign_exit':d['campaign_exit'],'training_processes':d['training_processes'],'paired_order_matches':d['paired_all_8910_sample_orders_match'],'arms':{k:{z:v[z] for z in ['line_count','update_count','validation_count','updates_exact_sequence','validation_epochs_exact_sequence','all_losses_gradients_finite','train_exit','log_sha256','best_validation','last_validation']} for k,v in d['arms'].items()},'checkpoints':d['checkpoints'],'zero_flow_EPE_equal_rollouts':d['zero_flow_EPE_equal_rollouts'],'zero_flow_rollouts':[{k:v for k,v in z.items() if k!='framewise_zero_flow_EPE'} for z in d['zero_flow_baseline']]},ensure_ascii=False))
