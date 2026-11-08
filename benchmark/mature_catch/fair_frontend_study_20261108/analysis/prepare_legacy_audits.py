"""Copy established auditors; never change their raw event or window contracts."""
from pathlib import Path
import ast,hashlib,json

out=Path(__file__).resolve().parent
fair=out.parent
source=fair.parent/'advantage_diagnosis_20261108/audit_contact_observation_windows.py'
dest=out/source.name
if dest.exists():assert dest.read_bytes()==source.read_bytes()
else:dest.write_bytes(source.read_bytes())
legacy=fair/'previous_tools/audit_cadence_final.py'
text=legacy.read_text(encoding='utf-8')
replacements={
 "assert len(records)==spec['expected_episodes']":"assert len(records)==sum(len(c['method_order']) for c in spec['cases'])",
 "for shard in [0,1]:":"for shard in sorted({int(c['shard']) for c in spec['cases']}):",
 "and not p.name.endswith('.final.log')":"and not p.name.endswith('.final.log') and not p.name.endswith('.supervisor.log')",
}
for old,new in replacements.items():assert old in text;text=text.replace(old,new)
ast.parse(text)
with (out/'audit_fair_raw.py').open('w',encoding='utf-8',newline='\n') as f:f.write(text)
receipt=dict(legacy_raw_auditor_sha256=hashlib.sha256(legacy.read_bytes()).hexdigest(),
 adapted_raw_auditor_sha256=hashlib.sha256((out/'audit_fair_raw.py').read_bytes()).hexdigest(),
 original_window_helper_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),
 adaptations=list(replacements.items()),raw_event_checks_changed=False,window_checks_changed=False)
with (out/'LEGACY_AUDIT_REUSE.json').open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
