"""Re-run the 35 procedural controls from the committed review without a GPU."""
import hashlib,json,shutil,subprocess,sys,tempfile
from pathlib import Path

p=Path(__file__).resolve().parent;review=p.parent
parent=p.parents[2]/'campaigns/newcore_20261103/train_e2vid_stage9.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='81b00422eddc59527c903a6c82f1deb40a840289d2dd4d32eb266eee9708cb11'
with tempfile.TemporaryDirectory(prefix='recon_repair_procedural_') as temporary:
    work=Path(temporary)
    for source in (review/'formal_sources').glob('*.py'):shutil.copyfile(source,work/source.name)
    for source in p.glob('test_*.py'):shutil.copyfile(source,work/source.name)
    shutil.copyfile(parent,work/'parent_trainer.py');(work/'upstream').mkdir()
    shutil.copyfile(p/'parent_training_recipe.json',work/'upstream/training_recipe.json')
    shutil.copyfile(review/'formal_sources/training_recipe.json',work/'repaired_training_recipe.json')
    results={}
    for script,receipt in [('test_repair_gate.py','repair_gate_preflight.json'),
                           ('test_repair_launch_contract.py','repair_launch_contract_preflight.json'),
                           ('test_trainer_preservation.py','trainer_preservation_preflight.json')]:
        subprocess.run([sys.executable,str(work/script)],cwd=work,check=True)
        result=json.loads((work/receipt).read_text());assert result['passed'];results[script]=len(result['checks'])
    assert sum(results.values())==35
print(json.dumps(dict(passed=True,checks=results,total=35,GPU_used=False,real_data_used=False)))
