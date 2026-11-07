"""CPU-only integration check: exact repaired serializer -> existing inference loader."""
import argparse,ast,copy,hashlib,json,os,random,signal,sys,tempfile,types
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import numpy as np,torch
from pinned_checkpoint_to_model import checkpoint_to_model

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--formal-root',type=Path,required=True)
    parser.add_argument('--fixed-study',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    assert not torch.cuda.is_available(),'This test must not occupy any GPU'
    torch.set_num_threads(2);torch.set_num_interop_threads(1);signal.alarm(300)
    args.out.mkdir(parents=True,exist_ok=False)
    R=args.campaign;F=args.formal_root;P=R.parents[2];modelroot=P/'external/sources/downstream/event_cnn_minimal'
    sys.path.insert(0,str(modelroot))
    from model.model import E2VIDRecurrent
    source=F/'sources/train_e2vid_repaired.py';manifest=json.loads((F/'sources/source_manifest.json').read_text());assert sha(source)==manifest[source.name]
    tree=ast.parse(source.read_text());mainnode=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    save_node=next(n for n in ast.walk(mainnode) if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
    helpers=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['sha256','state_sha256','rng_state','optimizer_step_counts']]
    config=json.loads((F/'sources/training_recipe.json').read_text());cfg=config['reconstruction'];assert cfg['model_args']['final_activation']=='sigmoid'
    results=[]
    for method in ['eventgenesis','v2e']:
        checkpoint=args.fixed_study/'fixed'/f'{method}_seed2_sigmoid/last.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=E2VIDRecurrent(copy.deepcopy(cfg['model_args'])).eval();model.load_state_dict(payload['model'],strict=True)
        optimizer=torch.optim.Adam(model.parameters(),**{k:cfg['optimizer'][k] for k in ['lr','betas','weight_decay','amsgrad']});optimizer.load_state_dict(payload['optimizer'])
        with tempfile.TemporaryDirectory(prefix='serializer_procedure_',dir=args.out) as temporary:
            scope=dict(torch=torch,hashlib=hashlib,Path=Path,random=random,np=np,json=json,os=os,datetime=datetime,timezone=timezone,
                model=model,optimizer=optimizer,cfg=cfg,config=config,args=SimpleNamespace(generator=method,seed=2),
                initial_hash=payload['initial_state_sha256'],update=20,best_lpips=float('inf'),best_epoch=None,
                config_sha=sha(F/'sources/training_recipe.json'),data_manifest_sha=sha(R/'adapters'/method/'reconstruction/adapter_manifest.json'),
                trainer_script_sha=sha(source),train_ids=['procedure_only_not_a_training_run'],val_ids=[],run_root=Path(temporary))
            exec(compile(ast.Module(body=helpers+[save_node],type_ignores=[]),str(source),'exec'),scope)
            info=scope['save_checkpoint'](0,20,'serialization_procedure_only_not_formal_training')
            loaded,record,kwargs=checkpoint_to_model(Path(info['path']),torch.device('cpu'),E2VIDRecurrent,copy.deepcopy(cfg['model_args']))
            assert kwargs==cfg['model_args'] and kwargs['final_activation']=='sigmoid'
            assert scope['state_sha256'](loaded)==scope['state_sha256'](model)==record['model_sha256']
            counts=scope['optimizer_step_counts'](optimizer);assert set(counts)=={20}
            model.reset_states();loaded.reset_states();torch.manual_seed(8002);probes=[]
            with torch.inference_mode():
                for i in range(3):
                    events=torch.randn(1,5,64,64);expected=model(events)['image'];actual=loaded(events)['image']
                    error=float((expected-actual).abs().max());assert error<=1e-7
                    assert float(actual.min())>=0 and float(actual.max())<=1
                    probes.append(dict(step=i,max_abs_error=error,minimum=float(actual.min()),maximum=float(actual.max())))
            wrong=copy.deepcopy(cfg['model_args']);wrong['final_activation']=''
            try:checkpoint_to_model(Path(info['path']),torch.device('cpu'),E2VIDRecurrent,wrong);raise RuntimeError('Wrong activation accepted')
            except ValueError:pass
            results.append(dict(method=method,seed=2,passed=True,input_type='three procedural signed event tensors, not a physical/real accuracy measurement',
                diagnostic_checkpoint_path=str(checkpoint),diagnostic_checkpoint_sha256=sha(checkpoint),
                exact_formal_serializer_source_sha256=sha(source),serializer_checkpoint_was_temporary=True,
                model_args_preserved=True,model_state_hash_match=True,optimizer_counters=sorted(set(counts)),
                wrong_linear_config_rejected=True,forward_probes=probes,real_data_used=False,optimizer_updates_in_this_test=0))
    final=dict(status='complete',passed=all(r['passed'] for r in results),results=results,real_data_used=False,
        GPU_used=False,formal_training_completed=False,inference_loader_provenance=json.loads((Path(__file__).parent/'inference_loader_provenance.json').read_text()))
    (args.out/'completion.json').write_text(json.dumps(final,indent=2)+'\n');signal.alarm(0);print('CPU_SERIALIZATION_PARITY_PASS',flush=True)
if __name__=='__main__':main()
