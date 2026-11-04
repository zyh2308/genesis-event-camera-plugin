import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('campaign',type=Path)
p.add_argument('task',choices=['reconstruction','flow','pose'])
a=p.parse_args()
root=a.campaign.resolve()
recovery=root/'recovery_20261103'
contract=json.loads((recovery/'real_evaluation_contract.json').read_text())
budget={'reconstruction':12000,'flow':8910,'pose':30000}[a.task]
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
recipe_hash=sha(root/'sources/training_recipe.json')
rows=[]
missing=[]
for method in contract['methods']:
    for seed in contract['seeds']:
        run=root/'training'/a.task/f'seed{seed}'/method
        receipt=run/'completion.json'
        if not receipt.is_file():
            missing.append(f'{method}/seed{seed}')
            continue
        row=json.loads(receipt.read_text())
        assert (row['task'],row['method'],row['seed'])==(a.task,method,seed)
        assert row['status']=='complete' and row['completed_updates']==budget
        assert row['recipe_sha256']==recipe_hash and row['real_data_used_in_training'] is False
        selected=Path(row['selected_checkpoint']).resolve()
        assert selected.parent==run and selected.name in ('synthetic_best.pt','best.pt')
        assert sha(selected)==row['selected_checkpoint_sha256']
        rows.append(row)
if missing:
    raise SystemExit('Missing formal training receipts: '+', '.join(missing))
for seed in contract['seeds']:
    assert len({row['initial_state_sha256'] for row in rows if row['seed']==seed})==1
target=recovery/f'{a.task}_frozen_checkpoints.json'
with target.open('x') as f:
    json.dump({'task':a.task,'recipe_sha256':recipe_hash,'runs':rows,'scoring_authorized':False},f,indent=2)
