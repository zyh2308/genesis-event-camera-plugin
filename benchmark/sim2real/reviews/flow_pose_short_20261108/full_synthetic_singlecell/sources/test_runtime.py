"""CPU-only safety fixtures; never invokes nvidia-smi or CUDA."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import lease_single_cell as r


def record(p):
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=r.sha(p))


def fixture(root):
    sources=root/'sources';sources.mkdir()
    runtime=root/'runtime';runtime.mkdir()
    parent=runtime/'lease_single_cell.py';parent.write_text('fixture parent bytes\n')
    entry=sources/'evaluate_single_cell.py';entry.write_text('fixture worker bytes\n')
    scientific=sources/'contract.json';scientific.write_text(json.dumps({'schema':'flow_full_synthetic_readonly_v1'}))
    manifest=sources/'source_manifest.json';manifest.write_text(json.dumps({p.name:r.sha(p) for p in (entry,scientific)}))
    ready=root/'DEPLOYMENT_READY.json';ready.write_text(json.dumps(dict(schema='flow_full_synthetic_cell_ready_v1',
        passed=True,task_root=str(root),source_manifest_sha256=r.sha(manifest),worker_contract_sha256=r.sha(scientific),
        read_only_model=True,optimizer_updates=0,real_scores_computed=False)))
    out=root/'rife_seed1'
    c=dict(schema='flow_full_synthetic_cell_command_v1',task='flow_full_synthetic_readonly_singlecell',task_root=str(root),
        gpu_index=1,gpu_uuid=r.GPUS[1],optimizer_updates=0,real_data_used=False,real_scores_computed=False,
        model_weights_read_only=True,timeout_seconds=1200,min_free_mib=22000,
        parent_source=record(parent),source_manifest=record(manifest),worker_contract=record(scientific),
        entrypoint=record(entry),ready=record(ready),cwd=str(sources),out=str(out),
        argv=[str(Path(sys.executable).resolve()),'-B','-u',str(entry),'--contract',str(scientific),
              '--seeds','1','--methods','rife_esim','--out',str(out)])
    return c,parent


class RuntimeTests(unittest.TestCase):
    def test_stdin_sealed_bytes_size_and_duplicate_key(self):
        raw=b'{"schema":"fixture"}\n';h=hashlib.sha256(raw).hexdigest()
        self.assertEqual(r.sealed_input(io.BytesIO(raw),h)['schema'],'fixture')
        with self.assertRaisesRegex(RuntimeError,'digest'):r.sealed_input(io.BytesIO(raw+b' '),h)
        raw=b'{"x":1,"x":2}'
        with self.assertRaisesRegex(RuntimeError,'Duplicate'):r.sealed_input(io.BytesIO(raw),hashlib.sha256(raw).hexdigest())
        raw=b'x'*(1024**2+1)
        with self.assertRaisesRegex(RuntimeError,'1MiB'):r.sealed_input(io.BytesIO(raw),hashlib.sha256(raw).hexdigest())

    def test_valid_exact_worker_and_safety_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,parent=fixture(root)
            self.assertEqual(r.verify_contract(c,root,1,parent)['verified_source_files'],2)
            mutations=[('optimizer',lambda d:d.__setitem__('optimizer_updates',1)),
                       ('real',lambda d:d.__setitem__('real_data_used',True)),
                       ('GPU',lambda d:d.__setitem__('gpu_uuid',r.GPUS[0])),
                       ('long timeout',lambda d:d.__setitem__('timeout_seconds',7200)),
                       ('shell',lambda d:d['argv'].__setitem__(0,'/bin/bash')),
                       ('inline python',lambda d:d['argv'].__setitem__(1,'-c')),
                       ('injected flag',lambda d:d['argv'].append('--enable-real-scoring')),
                       ('wrong seed list',lambda d:d['argv'].__setitem__(7,'2')),
                       ('cwd escape',lambda d:d.__setitem__('cwd',str(root.parent))),
                       ('out escape',lambda d:d.__setitem__('out',str(root.parent/'seed01')))]
            for label,change in mutations:
                with self.subTest(label=label):
                    d=copy.deepcopy(c);change(d)
                    with self.assertRaises((RuntimeError,FileNotFoundError)):r.verify_contract(d,root,1,parent)
            (root/'rife_seed1').mkdir()
            with self.assertRaisesRegex(RuntimeError,'existing worker output'):r.verify_contract(c,root,1,parent)

    def test_changed_source_or_ready_never_launchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,parent=fixture(root)
            Path(c['entrypoint']['path']).write_text('changed bytes')
            with self.assertRaisesRegex(RuntimeError,'Source bytes changed'):r.verify_contract(c,root,1,parent)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);c,parent=fixture(root);ready=Path(c['ready']['path'])
            v=r.read(ready);v['passed']=False;ready.write_text(json.dumps(v));c['ready']=record(ready)
            with self.assertRaisesRegex(RuntimeError,'CPU-ready'):r.verify_contract(c,root,1,parent)

    def test_symlink_and_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'inside').mkdir();(root/'alias').symlink_to(root/'inside',target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError,'symlink'):r.within(root/'alias',root,exists=True)
            with self.assertRaisesRegex(RuntimeError,'escapes'):r.within(root/'../outside',root)

    def test_all_matching_registry_records_must_be_released(self):
        uuid=r.GPUS[0]
        r.registry_admissible([('old',dict(gpu_uuid=uuid,status='released_complete'))],uuid)
        for status in ('released_after_catchit_probe','released_combined_finished',
                       'released_render_complete','released_incomplete_sampling_gap'):
            r.registry_admissible([('legacy',dict(gpu_uuid=uuid,status=status,current_job=None))],uuid)
        r.registry_admissible([('other',dict(gpu_uuid=r.GPUS[1],status='running'))],uuid)
        for status in ('running','complete','preparing','cleanup_failed','unknown',''):
            with self.subTest(status=status):
                with self.assertRaises(RuntimeError):r.registry_admissible([('old',dict(gpu_uuid=uuid,status=status))],uuid)
        with self.assertRaisesRegex(RuntimeError,'active job'):
            r.registry_admissible([('old',dict(gpu_uuid=uuid,status='released',current_job={'pid':123}))],uuid)
        with self.assertRaisesRegex(RuntimeError,'active job'):
            r.registry_admissible([('old',dict(gpu_uuid=uuid,status='released_after_catchit_probe',current_job={}))],uuid)

    def test_actual_compute_app_uuid_model_and_memory(self):
        uuid=r.GPUS[0];row=uuid+', NVIDIA GeForce RTX 3090, 24000'
        r.hardware_admissible(row,'',uuid)
        r.hardware_admissible(row,r.GPUS[1]+', 123\n',uuid)
        for gpu,apps in ((row,uuid+', 321\n'),(row.replace('24000','10000'),''),
                         (row.replace('RTX 3090','V100S'),''),(row.replace(uuid,r.GPUS[1]),'')):
            with self.assertRaises(RuntimeError):r.hardware_admissible(gpu,apps,uuid)

    def test_nonblocking_flock_is_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lock'
            with path.open('a+') as first,path.open('a+') as second:
                r.fcntl.flock(first,r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):r.fcntl.flock(second,r.fcntl.LOCK_EX|r.fcntl.LOCK_NB)

    def test_cleanup_only_own_new_group(self):
        unrelated=subprocess.Popen([sys.executable,'-B','-c','import time;time.sleep(30)'],start_new_session=True)
        worker=subprocess.Popen([sys.executable,'-B','-c',
            'import subprocess,sys,time;subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(30)"]);time.sleep(30)'],
            start_new_session=True)
        boundary=r.child_boundary(worker)
        try:
            time.sleep(.2)
            self.assertGreaterEqual(len(r.checked_group(boundary)),1)
            forged=dict(boundary,owner_pid=os.getpid()+1000000)
            with self.assertRaisesRegex(RuntimeError,'not created'):r.checked_group(forged)
            r.stop_own(worker,boundary)
            self.assertEqual(r.checked_group(boundary),[])
            self.assertIsNone(unrelated.poll())
        finally:
            r.stop_own(worker,boundary)
            unrelated.terminate();unrelated.wait(timeout=5)


if __name__=='__main__':unittest.main(verbosity=2)
