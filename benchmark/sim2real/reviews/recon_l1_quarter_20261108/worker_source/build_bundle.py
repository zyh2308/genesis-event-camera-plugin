"""Build only: same quarter workers, one bounded GPU2 paired manager."""
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    here = Path(__file__).resolve().parent
    previous = here.parent/'recon_l1_quarter_20261108'
    assert here.name == 'recon_l1_quarter_gpu2_20261108'
    parent_manifest_path = previous/'source_manifest.json'
    parent_manifest = json.loads(parent_manifest_path.read_text())
    for name, digest in parent_manifest.items():
        assert sha(previous/name) == digest, ('Parent source changed', name)
    reused = ['recon_short_diagnostic.py', 'activation_budget_diagnostic.py', 'metric_source_binding.json',
              'auxiliary_losses.py', 'test_auxiliary_losses.py', 'l1_quarter_diagnostic.py']
    for name in reused:
        shutil.copyfile(previous/name, here/name)
        assert sha(here/name) == parent_manifest[name]
    registration = json.loads((previous/'registered_protocol.json').read_text())
    registration.pop('output_root_names')
    registration.update(output_parent_root_name='20261108-recon-l1-quarter-paired-gpu2-v1',
                        worker_output_subdirectories=['eventgenesis', 'v2e'],
                        gpu_index=2, gpu_uuid='GPU-a02affe0-7598-579e-ca39-f551f35ad6e9', gpu_model='V100S 32GB',
                        execution='serial_on_one_device',
                        worker_sequence=['smoke_eventgenesis', 'smoke_v2e', 'study_eventgenesis', 'study_v2e'],
                        overall_timeout_seconds=7200, smoke_optimizer_updates=4,
                        protocol_sha256=sha(here/'PROTOCOL.md'),
                        resource_change_only=True, previous_resource_failure_preserved=True)
    assert registration['arms'] == ['control', 'photometric'] and registration['arm_count'] == 2
    assert registration['auxiliary_l1_weight'] == .25 and registration['unchanged_primary_gate_count'] == 9
    assert registration['total_main_optimizer_updates'] == 1200
    (here/'registered_protocol.json').write_text(json.dumps(registration, indent=2)+'\n')
    provenance = dict(verified_parent_directory=previous.name,
                      verified_parent_source_manifest_sha256=sha(parent_manifest_path),
                      worker_semantics_changed=False,
                      reused_file_sha256={name: sha(here/name) for name in reused},
                      changes=['one parent atomic UUID lease on GPU2', 'four phases serial on same V100S',
                               'both smokes and cross-method labels required before studies', 'two-hour overall bound',
                               'independent parent and child output directories'],
                      guards_reused_from='verified run_photometry wrapper: UUID/global+device flock/apps/registry/source hashes/own process group/finally release',
                      analysis_collector_included=False, no_training_started=True)
    (here/'bundle_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    names = reused+['run_paired_gpu2.py', 'PROTOCOL.md', 'registered_protocol.json', 'bundle_provenance.json', 'build_bundle.py']
    for name in names:
        if name.endswith('.py'):
            ast.parse((here/name).read_text())
    runner_tree = ast.parse((here/'run_paired_gpu2.py').read_text())
    phases = next(n.value for n in runner_tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PHASES' for t in n.targets))
    assert [row[0] for row in ast.literal_eval(phases)] == registration['worker_sequence']
    manifest = {name: sha(here/name) for name in names}
    (here/'source_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    files = {name: base64.b64encode((here/name).read_bytes()).decode() for name in names+['source_manifest.json']}
    payload = base64.b64encode(gzip.compress(json.dumps(files).encode())).decode()
    (here/'upload.b64').write_text(payload)
    decoded = json.loads(gzip.decompress(base64.b64decode(payload)))
    assert set(decoded) == set(files)
    for name, digest in manifest.items():
        assert hashlib.sha256(base64.b64decode(decoded[name])).hexdigest() == digest
    print(json.dumps(dict(files=len(files), bundle_chars=len(payload), source_manifest_sha256=sha(here/'source_manifest.json'),
                         all_worker_bytes_unchanged=True, static_validation_passed=True, no_training_started=True), indent=2))


if __name__ == '__main__':
    main()
