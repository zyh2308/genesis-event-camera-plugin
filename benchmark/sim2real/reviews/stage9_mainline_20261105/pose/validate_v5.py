"""Check post-return manifest hashes; show the same oracle rejects preservedv4."""
import ast,importlib.util
from pathlib import Path
d=Path(__file__).resolve().parent;p=d/'stage9_epose_v5.py';ast.parse(p.read_text())
spec=importlib.util.spec_from_file_location('v5',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.fixture(d/'metric_fixture_v5_local.json')
f=d/'interface_fixture.py'
extra="""
    protocol=json.loads((args.out/'protocol.json').read_text())
    manifest_checks=[]
    for seq in protocol['sequences']:
        path=Path(seq['sample_manifest'])
        actual=m.sha(path)
        count=len(path.read_text().splitlines())
        assert actual==seq['sample_manifest_sha256'],'POST_RETURN_MANIFEST_HASH_MISMATCH'
        assert count==seq['sample_count'],'POST_RETURN_MANIFEST_LINE_COUNT_MISMATCH'
        manifest_checks.append(dict(sequence=seq['sequence'],actual_sha256=actual,recorded_sha256=seq['sample_manifest_sha256'],actual_lines=count,recorded_lines=seq['sample_count']))
    result['manifest_checks_after_prepare_return']=manifest_checks
"""
def code_for(version,outfile):
    code=f.read_text().replace("HERE/'stage9_epose.py'",f"HERE/'stage9_epose_{version}.py'")
    code=code.replace("HERE/'interface_fixture_local.json'",f"HERE/'{outfile}'")
    marker="    result=json.loads((args.out/'preparation_receipt.json').read_text())"
    assert marker in code
    return code.replace(marker,marker+extra)
v4_rejected=False
try:exec(compile(code_for('v4','v4_hash_must_reject.json'),str(f),'exec'),{'__file__':str(f)})
except AssertionError as exc:
    assert str(exc)=='POST_RETURN_MANIFEST_HASH_MISMATCH';v4_rejected=True
assert v4_rejected
exec(compile(code_for('v5','interface_fixture_v5_local.json'),str(f),'exec'),{'__file__':str(f)})
m.dump(d/'v5_validation_receipt.json',dict(passed=True,source_sha256=m.sha(p),
    preserved_v4_sha256=m.sha(d/'stage9_epose_v4.py'),AST_parsed=True,
    metric_fixture_sha256=m.sha(d/'metric_fixture_v5_local.json'),
    fabricated_interface_fixture_sha256=m.sha(d/'interface_fixture_v5_local.json'),
    preserved_v4_rejected_by_post_return_hash_oracle=v4_rejected,
    v5_all_manifest_hashes_and_line_counts_checked_after_prepare_return=True,
    scope='Close JSONL outputs before registering hashes; model,input,metrics,geometry unchanged',
    other_output_hash_audit='dump() closes before hashes; protocol/preflight/metric receipt references are after dump returns; CAD/physical inputs are read-only and closed before hashing',
    model_loaded=False,real_scores_computed=False))
print('V5_SHA256',m.sha(p))
