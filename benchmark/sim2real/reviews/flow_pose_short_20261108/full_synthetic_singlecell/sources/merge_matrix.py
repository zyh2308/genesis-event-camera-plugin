"""Merge original14+conditional1 fixed cells, CPU only; never overwrite."""
import argparse,json
from pathlib import Path
import singlecell_contract as C
def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();c=C.read(a.contract)
    root=Path(c['root']);old=Path(c['origin_root']);C.need(a.out.resolve()==root/'merged_matrix.json' and not a.out.exists(),'Independent new merge output only')
    C.need(C.sha(old/'sources/source_manifest.json')==c['origin_source_manifest_sha256'],'Original sources changed')
    cells=[]
    for seed in (0,1,2):
        for method in c['methods']:
            b=c['checkpoint_binding'][f'{method}/seed{seed}'];shard='seed01' if seed<2 else 'seed2';path=old/shard/f'{method}_seed{seed}.json'
            try:value=C.read(path);valid=C.valid_cell(value,b)
            except (OSError,ValueError,KeyError,TypeError):valid=False
            if not valid:
                C.need(method=='rife_esim' and seed==1,'Only the registered missingRIFE seed1 can be supplemented')
                path=root/'rife_seed1/rife_esim_seed1.json';value=C.read(path);C.need(C.valid_cell(value,b),'Supplement is not a completed exact528 fixed cell')
                completion=C.read(root/'rife_seed1/completion.json');C.need(completion['status']=='complete' and completion['cells']==1,'Supplement worker incomplete')
                admission=C.read(root/'rife_seed1/admission.json');C.need(admission['source_manifest_sha256']==C.sha(root/'sources/source_manifest.json'),'Supplement source differs')
            else:
                admission=C.read(old/shard/'admission.json');C.need(admission['source_manifest_sha256']==c['origin_source_manifest_sha256'],'Original cellsource differs')
            cells.append(dict(method=method,seed=seed,path=str(path),sha256=C.sha(path),checkpoint_sha256=b['selected_checkpoint_sha256'],regions=value['regions'],strata=value['strata'],windows=528))
    result=dict(schema='flow_full_synthetic_readonly_merged15_v1',status='complete',cell_count=15,cells=cells,checkpoint_selection_changed=False,optimizer_updates=0,real_data_used=False,real_scores_used=False)
    with a.out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
if __name__=='__main__':main()
