"""Build a fresh isolated Stage9 dense-render worker; do not run it here."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

PINS = {
    'rich_physical_worker.py': '0eafc57043eb2a4b4e963fbb6ccf4e58d4286df12d3f1d20856637f839097d17',
    'stage9_replicad_scene_loader.py': '8aa173ff3bc0d91c7e432fb5edbd1414338f805f5d2f71865c6b94068adef5dd',
    'rich_bank_protocol.json': 'e3fd835346e422deedeeb77d2d110fc27bf0d6e568708ec7c690f7473bbb13ff',
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-dir', type=Path, required=True)
    p.add_argument('--target-root', type=Path, required=True)
    p.add_argument('--baseline-bank-file', type=Path, required=True)
    p.add_argument('--slot', type=int, default=0)
    p.add_argument('--stratum', type=int, choices=range(4), default=3)
    p.add_argument('--start-s', type=float, default=1.)
    p.add_argument('--duration-s', type=float, default=.1)
    a = p.parse_args()
    source_dir, target = a.source_dir.resolve(strict=True), a.target_root.resolve()
    assert not target.exists(), 'Use a fresh, isolated target root'
    assert 0 <= a.slot < 8
    start, duration = round(a.start_s*1000), round(a.duration_s*1000)
    assert abs(start/1000-a.start_s) < 1e-12 and start%10 == 0
    assert abs(duration/1000-a.duration_s) < 1e-12 and duration%10 == 0 and 10 <= duration <= 200
    assert 0 <= start and start+duration <= 2000
    baseline = a.baseline_bank_file.resolve(strict=True)
    assert baseline.name == f's{a.stratum}_{a.slot:02d}.h5'
    hashes = {name: hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in PINS}
    assert hashes == PINS, 'Stage9 source bytes differ from the reviewed WSL source pins'
    source = (source_dir/'rich_physical_worker.py').read_text()
    node, = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'capture']
    implementation = (Path(__file__).parent/'dense_reference_capture.py').read_text()
    capture_node, = [n for n in ast.parse(implementation).body if isinstance(n, ast.FunctionDef) and n.name == 'capture']
    capture = ''.join(implementation.splitlines(keepends=True)[capture_node.lineno-1:capture_node.end_lineno])
    spec = dict(schema='sampling_quality_stage9_dense_render_spec_v1', slot=a.slot, stratum=a.stratum,
                start_tick_1ms=start, duration_ticks_1ms=duration, baseline_bank_file=str(baseline),
                source_dir=str(source_dir), source_sha256=hashes,
                source_capture_replaced_only_in_private_worker=True,
                unchanged_trajectory_equations=True, genesis_physics_dt_s=.001,
                actual_subframe_render_dt_s=.001, noise_or_event_generator_called=False)
    lines = source.splitlines(keepends=True)
    candidate = ''.join(lines[:node.lineno-1])+'GEOMETRY_QUALITY_SPEC = '+repr(spec)+'\n\n'+capture+'\n'+''.join(lines[node.end_lineno:])
    before = "sys.path[:0]=[str(P/'runtime/sim2real/genesis_hdr_v2'),str(CORE),str(R/'sources'),str(R/'sources/benchmark/sim2real/scripts')]"
    after = ('sys.dont_write_bytecode=True\n'
             'SOURCE=Path('+repr(str(source_dir))+')\n'
             "sys.path[:0]=[str(P/'runtime/sim2real/genesis_hdr_v2'),str(CORE),str(SOURCE),str(SOURCE/'benchmark/sim2real/scripts')]")
    assert candidate.count(before) == 1
    candidate = candidate.replace(before, after)
    old = "cfg=json.loads((R/'sources/rich_bank_protocol.json').read_text())"
    assert candidate.count(old) == 1
    candidate = candidate.replace(old, "cfg=json.loads((SOURCE/'rich_bank_protocol.json').read_text())")
    compile(candidate, '<isolated Stage9 actual1ms-render worker>', 'exec')
    target.mkdir(parents=True)
    worker = target/'dense_physical_worker.py'
    worker.write_text(candidate)
    spec.update(worker=str(worker), worker_sha256=hashlib.sha256(worker.read_bytes()).hexdigest(),
                capture_helper_sha256=hashlib.sha256(implementation.encode()).hexdigest(),
                launch_argv=['python','-u',str(worker),'--campaign',str(target),'--slot',str(a.slot)],
                required_private_cpu_environment={'CUDA_VISIBLE_DEVICES':'','PYOPENGL_PLATFORM':'osmesa',
                    'GALLIUM_DRIVER':'llvmpipe','LIBGL_ALWAYS_SOFTWARE':'1','MESA_GL_VERSION_OVERRIDE':'4.5',
                    'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','LP_NUM_THREADS':'1',
                    'QD_NUM_THREADS':'2','QD_OFFLINE_CACHE':'1','PYTHONDONTWRITEBYTECODE':'1'},
                renderer_osmesa_library='Reuse the existing private Stage9 OSMesa runtime/library; do not install packages',
                built_not_executed=True)
    (target/'dense_reference_build.json').write_text(json.dumps(spec, indent=2)+'\n')
    assert {name:hashlib.sha256((source_dir/name).read_bytes()).hexdigest() for name in PINS} == PINS
    print(json.dumps(spec, indent=2))

if __name__ == '__main__':
    main()
