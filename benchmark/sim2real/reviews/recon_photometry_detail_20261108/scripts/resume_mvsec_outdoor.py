"""Resume the existing immutable MVSEC outdoor projection; default is offline.

Run in the existing Ubuntu WSL environment. This does not download, project,
train, score, delete, overwrite logs, or write a final verified_input_receipt.
Execution requires BOTH --execute and --writer-check-confirmed. The parent must
first check remote owner and active writers through its established connection.
SSH failures exit immediately with preserved receipts; no retry is implemented.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import shutil
import sys
import types
import zipfile
from pathlib import Path


BASE = Path('/home/科研/Eventbased_WAM/data/sim2real/stage5/real')
PROJECTION = BASE / 'MVSEC-EINexus-left-transfer-20261107-v1'
SOURCE = PROJECTION / 'outdoor_day'
REGISTRATION = PROJECTION / 'input_registration.json'
REG_SHA = '56297b67edde57f6d8780550919bd190ac319c1a03e6d0db16e5ae92d61e9614'
TRANSFER_BASE = (BASE / 'MVSEC-EINexus-left-transfer-sources-20261107-v1'
                 / 'pinned_transfer_base.py')
BASE_SHA = 'b3aec475efbaa2c6a50905e792af842f8141ccef9f3e22eb273152fb7e722692'
RECEIPT_ROOT = BASE / 'MVSEC-EINexus-left-resume-20261108-v1'
TARGET = ('/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/'
          'MVSEC-EINexus-left-20261107-v1/outdoor_day')
RUN_ID = 'MVSEC-EINexus-left-20261107-v1'
HOST = 'zhaoyuhan@g1.pkuml.org'
SSH = '/mnt/c/Windows/System32/OpenSSH/ssh.exe'
BW_LIMIT_KIB = 16384
NAMES = ('outdoor_day1_data.hdf5', 'outdoor_day1_gt.hdf5',
         'outdoor_day_calib.zip')
EXPECTED = {
    NAMES[0]: (4258503229,
              '1e4bbf6ea579448e5129911f09e4adcc794d3dd24e24c263967e32610b8395cd'),
    NAMES[1]: (6267452840,
              '3b0b2ce2b082b6d44c373bb4eaf3acfc7d79a73fba13b44ad344625d610ac608'),
    NAMES[2]: (1290038,
              'b5a5c370623beb01608e5b6c789a1a2d3065efcc36bb9b6c59584f14308a16c3'),
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def regular_canonical(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError('Expected fixed canonical regular file: ' + str(path))


def load_locked_transfer():
    """Load source bytes only after a full SHA check, without writing pycache."""
    regular_canonical(TRANSFER_BASE)
    code = TRANSFER_BASE.read_bytes()
    if hashlib.sha256(code).hexdigest() != BASE_SHA:
        raise ValueError('Pinned transfer source SHA mismatch; preserve and stop')
    module = types.ModuleType('mvsec_outdoor_pinned_transfer')
    module.__file__ = str(TRANSFER_BASE)
    exec(compile(code, str(TRANSFER_BASE), 'exec'), module.__dict__)
    module.HOST = HOST
    module.TARGET = TARGET
    module.SOURCE = str(SOURCE)
    module.REG_SHA = REG_SHA
    module.NAMES = NAMES
    module.SSH = SSH
    module.SSH_OPTIONS = [
        '-p', '22', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=360',
        '-o', 'StrictHostKeyChecking=yes', '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3', '-o', 'ConnectionAttempts=1',
    ]

    def specs(registration, selected):
        registration = Path(registration)
        regular_canonical(registration)
        if registration != REGISTRATION or digest(registration) != REG_SHA:
            raise ValueError('Fixed outdoor registration SHA/path mismatch')
        document = json.loads(registration.read_text())
        if document['schema'] != 'mvsec_lossless_left_projection_inputs_v1':
            raise ValueError('Unexpected projection registration schema')
        result = []
        for name in selected:
            if name not in NAMES:
                raise ValueError('Unregistered outdoor filename')
            if name.endswith('.hdf5'):
                kind = name[:-5].rsplit('_', 1)[1]
                record = document['files']['outdoor_day1']
                size, sha = record[kind + '_size_bytes'], record[kind + '_sha256']
                remote_path = record[kind + '_path_on_g1']
            else:
                record = document['calibration_files']['outdoor_day']
                size, sha = record['size_bytes'], record['sha256']
                remote_path = record['path_on_g1']
            if (size, sha) != EXPECTED[name] or remote_path != TARGET + '/' + name:
                raise ValueError('Registered size/SHA/target differs from fixed input')
            result.append(dict(name=name, source=str(SOURCE / name),
                               target=remote_path, expected_bytes=size,
                               expected_sha256=sha))
        return result

    module.specs = specs
    original_command = module.rsync_command

    def compressed_command(item, bwlimit):
        if bwlimit != BW_LIMIT_KIB:
            raise ValueError('Expected fixed shared-bandwidth cap')
        command = original_command(item, bwlimit)
        return [command[0], '--compress', '--compress-level=3', *command[1:]]

    module.rsync_command = compressed_command
    original_audit = module.remote_audit_program

    def owner_checked_audit(name):
        guard = (
            'import json\nfrom pathlib import Path\n'
            'owner_root=Path(%r)\n'
            'if not owner_root.is_dir() or owner_root.resolve()!=owner_root:\n'
            ' raise RuntimeError("Expected canonical fixed outdoor input root")\n'
            'owner_file=owner_root.parent/"owner.json"\n'
            'if owner_file.is_symlink():\n'
            ' raise RuntimeError("Refuse symlink owner receipt")\n'
            'if json.loads(owner_file.read_text())["run_id"]!=%r:\n'
            ' raise RuntimeError("Outdoor input owner run_id mismatch")\n'
        ) % (TARGET, RUN_ID)
        return guard + original_audit(name)

    module.remote_audit_program = owner_checked_audit
    return module


def preflight(module):
    """Complete local checks, including all registered source file SHA values."""
    if RECEIPT_ROOT.exists() or RECEIPT_ROOT.is_symlink():
        raise FileExistsError('New receipt root already exists; preserve: '
                              + str(RECEIPT_ROOT))
    if not Path(SSH).is_file() or not shutil.which('rsync'):
        raise RuntimeError('Existing Windows OpenSSH and WSL rsync are required')
    regular_canonical(PROJECTION / 'projection_manifest.json')
    manifest = json.loads((PROJECTION / 'projection_manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Original lossless projection has not completed')
    items = module.specs(REGISTRATION, NAMES)
    for item in items:
        path = Path(item['source'])
        regular_canonical(path)
        before = path.stat()
        if before.st_size != item['expected_bytes']:
            raise ValueError('Local immutable input size mismatch: ' + item['name'])
        if digest(path) != item['expected_sha256']:
            raise ValueError('Local immutable input SHA mismatch: ' + item['name'])
        after = path.stat()
        if any(getattr(before, k) != getattr(after, k)
               for k in ('st_size', 'st_mtime_ns', 'st_ino', 'st_dev')):
            raise ValueError('Local immutable input changed during SHA audit')
    with zipfile.ZipFile(SOURCE / NAMES[2]) as archive:
        if archive.testzip() is not None:
            raise ValueError('Local calibration ZIP is corrupt')

    # Reuse the original safety classifier tests (no SSH), including correct
    # full input skip, exact prefix resume, mismatch/oversize/symlink refusal.
    # Its command self-test uses a trial bandwidth cap; test the original
    # command there, then check our fixed compressed command separately below.
    compressed = module.rsync_command
    module.rsync_command = lambda item, bwlimit: compressed(item, BW_LIMIT_KIB)
    try:
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            module.selftest()
        original_checks = json.loads(captured.getvalue())
    finally:
        module.rsync_command = compressed
    if not original_checks['passed'] or original_checks['SSH_used']:
        raise ValueError('Original offline classifier self-test failed')

    command = module.rsync_command(items[1], BW_LIMIT_KIB)
    mandatory = ('--partial', '--append-verify', '--protect-args', '--compress',
                 '--compress-level=3', '--bwlimit=16384')
    if any(flag not in command for flag in mandatory):
        raise ValueError('Required transfer safety/compression flags are missing')
    if any(flag in command for flag in ('--delete', '--inplace')):
        raise ValueError('Unexpected destructive rsync flag')
    if command[-1] != HOST + ':' + TARGET + '/' + NAMES[1]:
        raise ValueError('Transfer command target mismatch')
    if '-o ConnectionAttempts=1' not in command[command.index('-e') + 1]:
        raise ValueError('Expected single SSH connection attempt')
    audit_program = module.remote_audit_program(NAMES[1])
    compile(audit_program, '<fixed_outdoor_audit>', 'exec')
    evidence = dict(
        status='passed_offline_preflight', SSH_used=False,
        source_base_sha256=BASE_SHA, registration_sha256=REG_SHA,
        existing_projection=str(PROJECTION), new_receipt_root=str(RECEIPT_ROOT),
        host=HOST, target=TARGET, local_full_sha256_verified=True,
        original_safety_checks=original_checks['checks'], files=items,
        real_scores_computed=False, GPU_used=False,
        downloads_or_reprojection=False, automatic_retries=0,
        note='GT historical logical progress is not a remote verified prefix; '
             'execution independently audits the current remote prefix SHA',
    )
    return items, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true',
                        help='Explicit offline mode (also the default)')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--writer-check-confirmed', action='store_true',
                        help='Parent has checked ownership/active writers')
    args = parser.parse_args()
    if args.preflight and args.execute:
        parser.error('--preflight and --execute are mutually exclusive')
    if args.execute and not args.writer_check_confirmed:
        parser.error('--execute requires parent --writer-check-confirmed')
    if args.writer_check_confirmed and not args.execute:
        parser.error('--writer-check-confirmed is meaningful only with --execute')
    transfer = load_locked_transfer()
    items, evidence = preflight(transfer)
    print(json.dumps(evidence, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    transfer_args = argparse.Namespace(
        writer_check_confirmed=True, out=RECEIPT_ROOT,
        bwlimit_kib=BW_LIMIT_KIB,
    )
    # Pinned execute creates the new receipt root exclusively and preserves a
    # failure.json plus exit.json on any SSH/rsync/audit failure. No retry loop.
    transfer.execute(transfer_args, items)
    transfer.save(RECEIPT_ROOT / 'offline_preflight.json', evidence)
    transfer.save(RECEIPT_ROOT / 'prepared.json', dict(
        status='transferred_full_hash_verified_pending_parent_array_validation',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        target=TARGET, registration_sha256=REG_SHA,
        projection_manifest=str(PROJECTION / 'projection_manifest.json'),
        real_scores_computed=False, final_verified_input_receipt_written=False,
        next_action='Parent verifies required HDF5 arrays and calibration through '
                    'its existing authenticated RPC before registering inputs',
    ))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(type(exc).__name__ + ': ' + str(exc), file=sys.stderr, flush=True)
        raise SystemExit(1)
