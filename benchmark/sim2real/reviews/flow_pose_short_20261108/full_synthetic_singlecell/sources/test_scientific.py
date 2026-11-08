import ast,copy,hashlib,json,unittest
from pathlib import Path
import evaluate_single_cell as E
class Scientific(unittest.TestCase):
    def test_explicit_seed1_method_and_device(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());uuid=c['shards']['rife_seed1']['gpu_uuid']
        self.assertEqual(E.validate(c,[1],['rife_esim'],E.ROOT/'rife_seed1',uuid),'rife_seed1')
        for seed,methods in [([2],['rife_esim']),([1],['v2e']),([0,1],['rife_esim'])]:
            with self.assertRaises(ValueError):E.validate(c,seed,methods,E.ROOT/'rife_seed1',uuid)
    def test_scientific_functions_unchanged(self):
        p=Path(__file__).parent;text=(p/'evaluate_single_cell.py').read_text();tree=ast.parse(text);pins=json.loads((p/'origin_function_pins.json').read_text())
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in pins:self.assertEqual(hashlib.sha256(ast.get_source_segment(text,n).encode()).hexdigest(),pins[n.name])
    def test_original_full15_bindings_retained(self):
        c=json.loads((Path(__file__).parent/'contract.json').read_text());self.assertEqual(len(c['checkpoint_binding']),15)
        actual=hashlib.sha256(json.dumps(c['checkpoint_binding'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.assertEqual(actual,c['origin_checkpoint_binding_sha256']);self.assertEqual(c['optimizer_updates'],0)
if __name__=='__main__':unittest.main()
