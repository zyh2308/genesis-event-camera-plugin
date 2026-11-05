"""Offline builder for isolated exact-geometry optimization candidates.

This file never imports/patches a running simulator. It reads a frozen source
and writes a different new source file. The source input remains untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build(source: str, mode: str) -> str:
    modes = {'histfree', 'prune-full', 'deadmix', 'histfree+deadmix', 'prune-full+deadmix'}
    if mode not in modes:
        raise ValueError('Unknown static candidate mode')
    result = source
    histfree = mode != 'deadmix'
    if histfree:
        old = '''        pixel_ids = torch.cat([item[0] for item in candidates])
        max_layers = int(torch.bincount(pixel_ids, minlength=total).max().item())
        unresolved = [torch.ones_like(item[0], dtype=torch.bool) for item in candidates]
        for _ in range(max_layers):'''
        assert result.count(old) == 1, 'Frozen coverage loop signature differs'
        new = '''        # Finite nonnegative tie tolerances guarantee progress. Unsupported
        # nonfinite values have already selected the unchanged bounded path.
        unresolved = [torch.ones_like(item[0], dtype=torch.bool) for item in candidates]
        while True:'''
        result = result.replace(old, new)
    if mode.startswith('prune-full'):
        needle = '    if candidates:\n        # Finite nonnegative'
        assert result.count(needle) == 1
        guard = '''    if candidates:
        # Optional exact zero-contribution pruning. A fully covered pixel has
        # accepted==0 in every later layer. Retain the original calculation
        # for unusual signed/NaN/HDR-overflow inputs, where inf*0 or signed
        # zero could otherwise make skipping the update observably different.
        _prune_completed = False
        if dtype == torch.float32:
            _probe = src[0] if src.stride(0) == 0 else src
            _lo, _hi = torch.stack((_probe.amin(), _probe.amax())).cpu().tolist()
            _bound = float(torch.finfo(dtype).max) / (8.0 * max(1, 4 * total))
            _prune_completed = (
                math.isfinite(_lo) and math.isfinite(_hi)
                and _lo >= 0.0 and _hi <= _bound
            )
        # Finite nonnegative'''
        result = result.replace(needle, guard)
        needle = '''            unresolved = next_unresolved
            if not bool(torch.stack([item.any() for item in unresolved]).any()):'''
        assert result.count(needle) == 1
        change = '''            unresolved = next_unresolved
            if _prune_completed:
                unresolved = [
                    active & (coverage[item[0]] < 1.0)
                    for item, active in zip(candidates, unresolved)
                ]
            if not bool(torch.stack([item.any() for item in unresolved]).any()):'''
        result = result.replace(needle, change)
    if 'deadmix' in mode:
        for expression in [
            '(1.0 - alpha_view) * torch.log(safe0) + alpha_view * torch.log(safe1)',
            '(1.0 - alpha_view[..., None]) * torch.log(safe0) + alpha_view[..., None] * torch.log(safe1)',
        ]:
            old = f'''            mixed = torch.exp({expression})
            if composite == "coverage_time_log_linear":
                mixed = conditional_mixed'''
            new = f'''            if composite == "coverage_time_log_linear":
                # Skip only an expression whose result was overwritten.
                mixed = conditional_mixed
            else:
                mixed = torch.exp({expression})'''
            assert result.count(old) == 1, 'Frozen overwritten mix signature differs'
            result = result.replace(old, new)
    if histfree:
        start = source.index('def _splat_projected_coverage_batch_torch(')
        end = source.index('\ndef depth_aware_bidirectional_warp_projected_batch_torch(', start)
        legacy = source[start:end].replace(
            'def _splat_projected_coverage_batch_torch(',
            'def _splat_projected_coverage_bounded_reference(', 1)
        marker = '    """Tensor-native front-to-back bilinear-footprint candidate splat."""'
        assert result.count(marker) == 1
        guarded = marker + '''
    # Preserve the original finite loop bound for NaN/Inf tolerances; NaN
    # comparisons need not consume any candidate, so an unbounded loop would
    # otherwise be a behavior change. No numerical tolerance is widened.
    if not (math.isfinite(float(depth_tie_abs_m)) and math.isfinite(float(depth_tie_rtol))):
        return _splat_projected_coverage_bounded_reference(
            image, source_depth, target_uv, target_z, source_valid,
            source_surface_id, min_coverage_fraction, depth_tie_abs_m,
            depth_tie_rtol,
        )'''
        result = result.replace(marker, guarded)
        start_new = result.index('def _splat_projected_coverage_batch_torch(')
        result = result[:start_new] + legacy + '\n' + result[start_new:]
        assert result.count('\nimport torch\n') == 1
        result = result.replace('\nimport torch\n', '\nimport math\nimport torch\n')
    compile(result, '<isolated geometry candidate>', 'exec')
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--mode', choices=['histfree', 'prune-full', 'deadmix', 'histfree+deadmix', 'prune-full+deadmix'], default='histfree')
    args = p.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if source == target:
        raise ValueError('Source and candidate must be different paths')
    if target.exists():
        raise FileExistsError('Candidate must be a fresh, isolated output')
    before = source.read_bytes()
    result = build(source.read_text(encoding='utf-8'), args.mode)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result, encoding='utf-8')
    assert source.read_bytes() == before
    report = {
        'mode': args.mode,
        'source': str(source),
        'candidate': str(target),
        'source_sha256': hashlib.sha256(before).hexdigest(),
        'candidate_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'source_unchanged': True,
        'tests_run': False,
        'noise_sampler_occlusion_thresholds_changed': False,
    }
    target.with_suffix(target.suffix + '.build.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
