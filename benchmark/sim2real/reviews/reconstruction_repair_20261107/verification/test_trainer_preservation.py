"""Check that the repair preserves the dataset, optimizer, sampler, RNG and resume machinery."""
import ast,copy,difflib,json
from pathlib import Path

p=Path(__file__).parent
parent=Path('/home/科研/Eventbased_WAM/repositories/gec-mainline-resume-20261105/benchmark/sim2real/campaigns/newcore_20261103/train_e2vid_stage9.py')
if (p/'parent_trainer.py').exists():parent=p/'parent_trainer.py'
left=parent.read_text();right=(p/'train_e2vid_repaired.py').read_text();a=ast.parse(left);b=ast.parse(right)
def find(tree,name):return next(x for x in ast.walk(tree) if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)
def dump(node):return ast.dump(node,include_attributes=False)
checks=[]
for name in ['state_sha256','rng_state','restore_rng','sample_digest','scalar_losses','optimizer_step_counts',
             'Stage4SequenceDataset','Stage4E2VIDDataset','get_batch','recurrent_state_fixture']:
    assert dump(find(a,name))==dump(find(b,name)),name;checks.append(name+'_AST_unchanged')
oldsave=find(a,'save_checkpoint');newsave=copy.deepcopy(find(b,'save_checkpoint'))
for node in ast.walk(newsave):
    if isinstance(node,ast.Dict):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value in ['model_args','reconstruction_repair','real_data_used'])]
        node.keys=[x[0] for x in pairs];node.values=[x[1] for x in pairs]
assert dump(oldsave)==dump(newsave);checks.append('checkpoint_and_cursor_unchanged_except_declared_metadata')
def optimizer_calls(tree):
    found=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='optimizer':
            found.append(dump(node))
    return found
assert optimizer_calls(a)==optimizer_calls(b);checks.append('all_optimizer_calls_AST_unchanged')
oldcfg=json.loads((p/'upstream/training_recipe.json').read_text());newcfg=json.loads((p/'repaired_training_recipe.json').read_text())
metadata=newcfg.pop('reconstruction_repair');newcfg['reconstruction']['model_args']['final_activation']=''
assert oldcfg==newcfg;checks.append('recipe_only_changes_output_activation_and_repair_metadata')
(p/'trainer_preservation_preflight.json').write_text(json.dumps(dict(passed=True,checks=checks,real_data_used=False),indent=2)+'\n')
(p/'trainer_delta.patch').write_text(''.join(difflib.unified_diff(left.splitlines(True),right.splitlines(True),fromfile='parent/train_e2vid_stage9.py',tofile='repair/train_e2vid_repaired.py')))
print('preservation',len(checks),'passed')
