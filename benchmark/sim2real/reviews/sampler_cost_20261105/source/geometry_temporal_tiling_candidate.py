"""Offline static builder: tile only coverage splat, never projection/sampling.

Supply a frozen or histfree/deadmix/prune candidate GPU-geometry module. A fresh
module is written; source contents and running modules are never changed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


def build(source: str, chunk_size: int) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'depth_aware_bidirectional_warp_projected_batch_torch']
    assert len(functions) == 1
    node = functions[0]
    assert node.args.args[-1].arg == 'surface_depth_rtol'
    assert not any(arg.arg == 'coverage_chunk_size' for arg in node.args.args)
    lines = source.splitlines(keepends=True)
    line_index = node.args.args[-1].end_lineno
    assert lines[line_index-1].rstrip().endswith(',')
    lines.insert(line_index, f'    coverage_chunk_size: Optional[int] = {chunk_size},\n')
    result = ''.join(lines)
    old = '''            result = _splat_projected_coverage_batch_torch(
                frame_m, depth_m, uv_m, z_m, valid_m, surface_m,
                min_coverage_fraction, depth_tie_abs_m, depth_tie_rtol,
            )'''
    assert result.count(old) == 1, 'Coverage-only dispatch differs from reviewed source'
    new = '''            # Only the splat is tiled. Full-alpha projection and subsequent
            # composition/endpoint handling retain their original shapes/math.
            _tile = coverage_chunk_size
            if _tile is not None and (type(_tile) is not int or _tile <= 0):
                raise ValueError("coverage_chunk_size must be None or a positive int")
            _can_tile = _tile is not None and M > _tile
            if _can_tile:
                # Conservative numerical eligibility; unusual inputs select the
                # unmodified full-stack dispatch, preserving IEEE corner cases.
                _can_tile = (
                    frame.dtype == torch.float32 and frame.numel() > 0
                    and math.isfinite(float(depth_tie_abs_m))
                    and math.isfinite(float(depth_tie_rtol))
                )
                if _can_tile:
                    _lo, _hi = torch.stack((frame.amin(), frame.amax())).cpu().tolist()
                    _bound = float(torch.finfo(frame.dtype).max) / (8.0 * max(1, 4*M*H*W))
                    _can_tile = (
                        math.isfinite(_lo) and math.isfinite(_hi)
                        and _lo >= 0.0 and _hi <= _bound
                    )
            if not _can_tile:
                result = _splat_projected_coverage_batch_torch(
                    frame_m, depth_m, uv_m, z_m, valid_m, surface_m,
                    min_coverage_fraction, depth_tie_abs_m, depth_tie_rtol,
                )
            else:
                _pieces = []
                for _left in range(0, M, _tile):
                    _right = min(M, _left + _tile)
                    _pieces.append(_splat_projected_coverage_batch_torch(
                        frame_m[_left:_right], depth_m[_left:_right],
                        uv_m[_left:_right], z_m[_left:_right],
                        valid_m[_left:_right],
                        None if surface_m is None else surface_m[_left:_right],
                        min_coverage_fraction, depth_tie_abs_m, depth_tie_rtol,
                    ))
                _keys = tuple(_pieces[0][2])
                if not all(tuple(_part[2]) == _keys for _part in _pieces):
                    raise RuntimeError("Coverage diagnostic keys differ between tiles")
                result = (
                    torch.cat([_part[0] for _part in _pieces], dim=0),
                    torch.cat([_part[1] for _part in _pieces], dim=0),
                    {key: torch.cat([_part[2][key] for _part in _pieces], dim=0)
                     for key in _keys},
                )'''
    result = result.replace(old, new)
    parsed = ast.parse(result)
    has_math = any(isinstance(item, ast.Import)
                   and any(alias.name == 'math' for alias in item.names)
                   for item in parsed.body)
    if not has_math:
        assert result.count('\nimport torch\n') == 1
        result = result.replace('\nimport torch\n', '\nimport math\nimport torch\n')
    # Whole projection/flow functions must remain structurally identical.
    expected = {item.name: ast.dump(item, include_attributes=False)
                for item in tree.body if isinstance(item, ast.FunctionDef)
                and item.name not in {'depth_aware_bidirectional_warp_projected_batch_torch'}}
    actual = {item.name: ast.dump(item, include_attributes=False)
              for item in ast.parse(result).body if isinstance(item, ast.FunctionDef)
              and item.name not in {'depth_aware_bidirectional_warp_projected_batch_torch'}}
    assert actual == expected, 'Tiling must not modify projection or splat formulas'
    compile(result, '<coverage-splat-only temporal tiling>', 'exec')
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--chunk-size', type=int, choices=[16,32], required=True)
    args = p.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if source == target or target.exists():
        raise ValueError('Tiling candidate requires a fresh path distinct from source')
    before = source.read_bytes()
    result = build(source.read_text(encoding='utf-8'), args.chunk_size)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result, encoding='utf-8')
    assert source.read_bytes() == before
    receipt = {
        'mode': 'coverage-splat-only-temporal-tiling',
        'chunk_size_default': args.chunk_size,
        'baseline_override': 'coverage_chunk_size=None',
        'source': str(source), 'candidate': str(target),
        'source_sha256': hashlib.sha256(before).hexdigest(),
        'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'projection_and_other_splat_AST_unchanged': True,
        'full_alpha_stack_preserved': True,
        'all_diagnostics_concatenated': True,
        'source_unchanged': True,
        'GPU_or_quality_tests_run': False,
    }
    target.with_suffix(target.suffix+'.build.json').write_text(
        json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
