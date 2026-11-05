"""Offline builder: reuse repeated Boolean-mask indexing in coverage splats.

The input is an already-built histfree+deadmix geometry module. The unchanged
bounded-reference fallback, all FP32 arithmetic, candidate ordering, and FP32
scatter operations retain their source text/order. Only integral row selection
and first-layer integer/Boolean assignments are replaced.

Usage::

    python geometry_index_reuse_candidate.py \
        --source /isolated/histfree_deadmix.py --target /isolated/index_reuse.py

This only builds an untrusted optimization candidate. A real CUDA byte gate
against the frozen reference, including stress_geometry.run_stress, must pass
before it is considered for deployment. This builder never imports a simulator.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


_FUNCTION = "_splat_projected_coverage_batch_torch"


def _replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"Expected one frozen source signature, observed {count}: {old[:100]!r}")
    return source.replace(old, new, 1)


def _scatter_calls(source: str) -> list[str]:
    """Static guard: scatter receivers/methods/arguments and order are fixed.

    Selection expressions inside these statements intentionally change to named
    index_select results. Verify their receiver, method, count and text order;
    the explicit replacements below preserve the actual selected data/order.
    """
    tree = ast.parse(source)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _FUNCTION:
            calls = [child for child in ast.walk(node)
                     if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                     and child.func.attr in {"scatter_", "scatter_add_", "scatter_reduce_"}]
            calls.sort(key=lambda child: (child.lineno, child.col_offset))
            result = [ast.unparse(call.func) for call in calls]
    return result


def build(source: str) -> str:
    source = source.replace("\r\n", "\n")
    start = source.index(f"def {_FUNCTION}(")
    end = source.index("\ndef depth_aware_bidirectional_warp_projected_batch_torch(", start)
    body = source[start:end]
    if "while True:" not in body or "_splat_projected_coverage_bounded_reference(" not in body:
        raise ValueError("Expected isolated histfree+deadmix source with bounded fallback")
    if "# Skip only an expression whose result was overwritten." not in source:
        raise ValueError("Expected deadmix base; build histfree+deadmix first")

    body = _replace_once(body,
        '''    for ox, oy, weight in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),''',
        '''    # Expanding one endpoint over M may make src non-contiguous.
    # Materialize its flattened rows at most once, rather than once per tap.
    source_rows = src.reshape(total, src.shape[-1])
    target_depth_rows = ztarget.reshape(-1)
    surface_rows = surface.reshape(-1) if surface is not None else None
    for ox, oy, weight in (
        (0, 0, (1.0 - wx) * (1.0 - wy)),''')

    body = _replace_once(body,
        '''        if bool(keep.any()):
            idx = (batch_offset + y * W + x)[keep]
            w = weight[keep]
            z = ztarget[keep]
            values = src[keep]
            labels = surface[keep] if surface is not None else None''',
        '''        # Boolean indexing would independently recompute keep.nonzero()
        # for each selected field. Reuse the identical row-major source rows.
        selected = keep.reshape(-1).nonzero(as_tuple=False).reshape(-1)
        if selected.numel():
            idx = (batch_offset + y * W + x).reshape(-1).index_select(0, selected)
            w = weight.reshape(-1).index_select(0, selected)
            z = target_depth_rows.index_select(0, selected)
            values = source_rows.index_select(0, selected)
            labels = surface_rows.index_select(0, selected) if surface is not None else None''')

    body = _replace_once(body,
        '''                if bool(active.any()):
                    nearest.scatter_reduce_(0, idx[active], z[active], reduce="amin", include_self=True)''',
        '''                selected_active = active.nonzero(as_tuple=False).reshape(-1)
                if selected_active.numel():
                    nearest.scatter_reduce_(0, idx.index_select(0, selected_active), z.index_select(0, selected_active), reduce="amin", include_self=True)''')

    body = _replace_once(body,
        '''                if bool(take.any()):
                    idx_t, w_t, value_t = idx[take], w[take], value[take]
                    layer_weight.scatter_add_(0, idx_t, w_t)
                    layer_value.scatter_add_(0, idx_t[:, None].expand(-1, value.shape[-1]), value_t * w_t[:, None])
                    if label is not None:
                        label_min.scatter_reduce_(0, idx_t, label[take], reduce="amin", include_self=True)
                        label_max.scatter_reduce_(0, idx_t, label[take], reduce="amax", include_self=True)''',
        '''                selected_take = take.nonzero(as_tuple=False).reshape(-1)
                if selected_take.numel():
                    idx_t = idx.index_select(0, selected_take)
                    w_t = w.index_select(0, selected_take)
                    value_t = value.index_select(0, selected_take)
                    layer_weight.scatter_add_(0, idx_t, w_t)
                    layer_value.scatter_add_(0, idx_t[:, None].expand(-1, value.shape[-1]), value_t * w_t[:, None])
                    if label is not None:
                        label_t = label.index_select(0, selected_take)
                        label_min.scatter_reduce_(0, idx_t, label_t, reduce="amin", include_self=True)
                        label_max.scatter_reduce_(0, idx_t, label_t, reduce="amax", include_self=True)''')

    body = _replace_once(body,
        '''            if surface is not None and bool(first_pixels.any()):
                surface_id[first_pixels] = label_min[first_pixels]
                surface_consistent[first_pixels] = label_min[first_pixels] == label_max[first_pixels]''',
        '''            if surface is not None:
                selected_first = first_pixels.nonzero(as_tuple=False).reshape(-1)
                if selected_first.numel():
                    first_label_min = label_min.index_select(0, selected_first)
                    first_label_max = label_max.index_select(0, selected_first)
                    surface_id.index_copy_(0, selected_first, first_label_min)
                    surface_consistent.index_copy_(0, selected_first, first_label_min == first_label_max)''')

    result = source[:start] + body + source[end:]
    compile(result, "<isolated geometry index reuse candidate>", "exec")
    ast.parse(result)
    if _scatter_calls(source) != _scatter_calls(result):
        raise AssertionError("Scatter receivers/methods/order changed")
    # The strict reference fallback is copied from the frozen source and must
    # remain exactly identical, including its NaN/Inf tolerance behavior.
    legacy_start = source.index("def _splat_projected_coverage_bounded_reference(")
    legacy_end = source.index(f"def {_FUNCTION}(", legacy_start)
    if source[legacy_start:legacy_end] != result[legacy_start:legacy_end]:
        raise AssertionError("Bounded reference fallback was modified")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    arguments = parser.parse_args()
    source, target = arguments.source.resolve(), arguments.target.resolve()
    if source == target:
        raise ValueError("Source and output must differ")
    if target.exists():
        raise FileExistsError("Output must be a new isolated candidate")
    before = source.read_bytes()
    result = build(source.read_text(encoding="utf-8"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result, encoding="utf-8")
    if source.read_bytes() != before:
        raise AssertionError("Source changed during isolated build")
    report = {
        "mode": "histfree+deadmix+index_reuse", "source": str(source), "candidate": str(target),
        "source_sha256": hashlib.sha256(before).hexdigest(),
        "candidate_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_unchanged": True, "compile_ast_passed": True,
        "scatter_receiver_method_order_unchanged": True,
        "bounded_reference_fallback_unchanged": True,
        "sampler_noise_occlusion_rules_changed": False,
        "actual_cuda_byte_gate_run": False,
    }
    target.with_suffix(target.suffix + ".build.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
