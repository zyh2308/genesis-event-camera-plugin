"""Build isolated sources only. No remote access, ready publication or training."""
import ast,base64,gzip,hashlib,json,shutil
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=Path(__file__).resolve().parent;old=p.parent/'reconstruction_repair_20261107'
    verified=json.loads((old/'formal_source_manifest_v2.json').read_text())
    for filename,oldname in [('train_e2vid_repaired.py','train_e2vid_repaired.py'),('training_recipe.json','repaired_training_recipe.json')]:
        shutil.copyfile(old/oldname,p/filename);assert sha(p/filename)==verified[filename]
    shutil.copyfile(old/'recon_short_diagnostic.py',p/'recon_short_diagnostic.py')
    shutil.copyfile(old/'recon_repair_gate.py',p/'inherited_gate.py')
    gate=(old/'recon_repair_gate.py').read_text().replace("METHODS=('eventgenesis','v2e')","METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')")
    gate=gate.replace('len(actual)!=6','len(actual)!=15').replace('Complete paired three-seed matrix required','Complete five-method three-seed matrix required')
    (p/'recon_selected_gate.py').write_text(gate)
    previous=json.loads((old/'final_server_snapshot.json').read_text())['paired_short']
    (p/'inherited_gate_results.json').write_text(json.dumps(previous['results'],indent=2)+'\n')
    budget=json.loads((p.parent/'recon_activation_budget_20261108/result_summary.json').read_text())
    hqf=json.loads((p.parent/'hqf_repair_20261108/hqf_result_summary.json').read_text())['protocol']
    resumed={}
    for method in ('eventgenesis','v2e'):
        refs=budget['workers'][method]['completion']['references']
        last=next(r for r in refs if r['name']=='sigmoid_stop');best=next(r for r in refs if r['name']=='sigmoid_4800')
        summary=next(r['safe_stop_summary'] for r in hqf['runs'] if r['method']==method)
        resumed[method+'/seed0']=dict(last={k:last[k] for k in ('checkpoint','checkpoint_sha256','updates')},
            best={k:best[k] for k in ('checkpoint','checkpoint_sha256','updates')},summary=summary)
    mapping=[(0,2,'GPU-b89ca413-eb9c-61a5-a89b-9c244d032fff'),(1,5,'GPU-dd93b58d-df3d-9d33-bffc-2203bb503e87'),(2,6,'GPU-48385047-4365-b814-dac9-7d0d2c63d9b0')]
    plan=dict(schema='recon_selected_full_plan_v1',root='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261108-recon-selected-full-v1',
        campaign='/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261103-stage9-direct-v1',methods=['eventgenesis','v2e','vid2e','film_esim','rife_esim'],seeds=[0,1,2],
        epochs=200,training_budget_updates=12000,gate_updates=300,gate_new_cases=9,gate_inherited_cases=6,
        final_activation='sigmoid',loss=dict(lpips_vgg=1.,temporal=2.,auxiliary_l1=0.),frozen_core='62da4f35ce3c5c96f04268bd1687a27657af9238',
        trainer_sha256=sha(p/'train_e2vid_repaired.py'),recipe_sha256=sha(p/'training_recipe.json'),resume_sources=resumed,
        inherited_gate=dict(root=previous['root'],source_manifest=previous['source_manifest'],completion_sha256=previous['completion_sha256']),
        lanes=[dict(seed=s,host='g3',gpu_index=i,gpu_uuid=u,gpu_model='NVIDIA GeForce RTX 4090') for s,i,u in mapping],
        real_sequences=['boxes','reflective_materials','still_life'],real_result_scope='posthoc_selected_hqf3_development',real_scoring_enabled=False,
        hqf=hqf,no_real_feedback_to_training_or_selection=True,original_hqf14_results_preserved=True,
        source_user_instruction='用户明确授权选定三条HQF优势轨迹作为展示开发子集，五方法三seed完成共同配方训练；优先光流位姿，真实评分另行隔离。')
    (p/'registered_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    names=['reserve_and_run_lane.py','selected_common.py','recon_repair_launch_contract.py','prepare_selected.py','run_selected_lane.py',
        'train_e2vid_repaired.py','training_recipe.json','recon_short_diagnostic.py','inherited_gate.py','recon_selected_gate.py',
        'inherited_gate_results.json','registered_plan.json','DEPLOYMENT_PLAN.md','build_bundle.py','test_selected_contracts.py']
    for n in names:
        if n.endswith('.py'):ast.parse((p/n).read_text())
    manifest={n:sha(p/n) for n in names};(p/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    files={n:base64.b64encode((p/n).read_bytes()).decode() for n in names+['source_manifest.json']}
    upload=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode();(p/'upload.b64').write_text(upload)
    decoded=json.loads(gzip.decompress(base64.b64decode(upload)))
    for n,h in manifest.items():assert hashlib.sha256(base64.b64decode(decoded[n])).hexdigest()==h
    print(json.dumps(dict(files=len(files),bundle_chars=len(upload),source_manifest_sha256=sha(p/'source_manifest.json'),
        trainer_byte_exact=True,recipe_byte_exact=True,ready_not_published=True),indent=2))

if __name__=='__main__':main()
