#!/usr/bin/env python3
"""Import the registered ReplicaCAD apt_1 layout into Genesis and render one view."""
from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path

import genesis as gs
import numpy as np
from PIL import Image


COORD_H_TO_G = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)
Q_BASIS_WXYZ = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def habitat_xyzw_to_genesis_wxyz(value: list[float]) -> list[float]:
    if len(value) != 4 or not np.isfinite(value).all():
        raise ValueError(f"invalid Habitat quaternion: {value}")
    x, y, z, w = map(float, value)
    q_h = np.array([w, x, y, z], dtype=np.float64)
    q_h /= np.linalg.norm(q_h)
    q_basis_inv = Q_BASIS_WXYZ * np.array([1.0, -1.0, -1.0, -1.0])
    q_g = qmul(qmul(Q_BASIS_WXYZ, q_h), q_basis_inv)
    q_g /= np.linalg.norm(q_g)
    return q_g.tolist()


def habitat_xyz_to_genesis_xyz(value: list[float] | tuple[float, ...]) -> list[float]:
    p = np.asarray(value, dtype=np.float64)
    if p.shape != (3,) or not np.isfinite(p).all():
        raise ValueError(f"invalid Habitat vector: {value}")
    return (COORD_H_TO_G @ p).tolist()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_coordinate_contract() -> None:
    if not np.allclose(COORD_H_TO_G @ COORD_H_TO_G.T, np.eye(3), atol=1e-12):
        raise RuntimeError("coordinate basis is not orthonormal")
    if not np.isclose(np.linalg.det(COORD_H_TO_G), 1.0, atol=1e-12):
        raise RuntimeError("coordinate basis changes handedness")
    if not np.allclose(habitat_xyz_to_genesis_xyz([0, 1, 0]), [0, 0, 1]):
        raise RuntimeError("Habitat +Y up does not map to Genesis +Z up")
    if not np.allclose(habitat_xyz_to_genesis_xyz([0, 0, -1]), [0, 1, 0]):
        raise RuntimeError("Habitat -Z forward basis does not map as declared")
    half = math.sqrt(0.5)
    # Habitat +Y 90 degree yaw, stored [x,y,z,w], becomes Genesis +Z yaw.
    q_h_yaw = [0.0, half, 0.0, half]
    expected = [half, 0.0, 0.0, half]
    if not np.allclose(habitat_xyzw_to_genesis_wxyz(q_h_yaw), expected, atol=1e-12):
        raise RuntimeError("quaternion basis conversion failed the independent yaw case")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    asset_root = args.asset_root.resolve()
    scene_path = asset_root / "configs/scenes/apt_1.scene_instance.json"
    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    verify_coordinate_contract()
    scene_data = json.loads(scene_path.read_text())
    stage_name = scene_data["stage_instance"]["template_name"]
    stage_config_path = asset_root / "configs/stages" / f"{stage_name}.stage_config.json"
    stage_config = json.loads(stage_config_path.read_text())
    if stage_config.get("up") != [0, 1, 0]:
        raise RuntimeError(f"unexpected stage up vector: {stage_config.get('up')}")
    stage_asset = (stage_config_path.parent / stage_config["render_asset"]).resolve()
    if not stage_asset.is_file():
        raise FileNotFoundError(stage_asset)

    rigid_specs: list[dict] = []
    for row in scene_data["object_instances"]:
        template = row["template_name"].rsplit("/", 1)[-1]
        config_path = asset_root / "configs/objects" / f"{template}.object_config.json"
        config = json.loads(config_path.read_text())
        asset_path = (config_path.parent / config["render_asset"]).resolve()
        if not asset_path.is_file():
            raise FileNotFoundError(asset_path)
        scale = row.get("uniform_scale", 1.0)
        rigid_specs.append(
            {
                "template_name": row["template_name"],
                "asset_path": asset_path,
                "position_g": habitat_xyz_to_genesis_xyz(row["translation"]),
                "quaternion_g_wxyz": habitat_xyzw_to_genesis_wxyz(row["rotation"]),
                "scale": scale,
            }
        )

    art_urdf_rel = {
        "fridge": "urdf/fridge/fridge.urdf",
        "kitchen_counter": "urdf/kitchen_counter/kitchen_counter.urdf",
        "kitchenCupboard_01": "urdf/kitchen_cupboards/kitchenCupboard_01.urdf",
        "chestOfDrawers_01": "urdf/chest_of_drawers/chestOfDrawers_01.urdf",
        "cabinet": "urdf/cabinet/cabinet.urdf",
        "door2": "urdf/doors/door2.urdf",
    }
    art_specs = []
    for row in scene_data.get("articulated_object_instances", []):
        template = row["template_name"]
        rel = art_urdf_rel.get(template)
        if rel is None:
            raise RuntimeError(f"no checked URDF mapping for articulated asset {template}")
        path = asset_root / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        art_specs.append(
            {
                "template_name": template,
                "asset_path": path,
                "position_g": habitat_xyz_to_genesis_xyz(row["translation"]),
                "quaternion_g_wxyz": habitat_xyzw_to_genesis_wxyz(row["rotation"]),
                "scale": row.get("uniform_scale", 1.0),
                "fixed_base": row.get("fixed_base", False),
                "translation_origin": row.get("translation_origin"),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    # ReplicaCAD v1.5 uses comma-separated URDF mesh scale vectors, which the
    # bundled Genesis urdfpy parser rejects, and the kitchen-counter URDF has
    # one additional comma in a box size vector. Normalize only these delimiters
    # in run-local copies; preserve every numeric value and the shared source.
    compat_root = args.output_dir / "compat_assets"
    compat_patches: list[dict] = []
    copied_dirs: dict[Path, Path] = {}
    mesh_scale_pattern = re.compile(r'(<mesh\b[^>]*\bscale=")([^"]+)(")')
    bad_counter_size = '<box size="0.617 0.145, 0.894"/>'
    fixed_counter_size = '<box size="0.617 0.145 0.894"/>'
    for spec in art_specs:
        source_urdf = spec["asset_path"]
        source_dir = source_urdf.parent
        overlay_dir = copied_dirs.get(source_dir)
        if overlay_dir is None:
            overlay_dir = compat_root / source_dir.relative_to(asset_root)
            overlay_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_dir, overlay_dir)
            copied_dirs[source_dir] = overlay_dir
        overlay_urdf = overlay_dir / source_urdf.name
        source_text = source_urdf.read_text()
        scale_replacements = 0

        def normalize_scale(match: re.Match[str]) -> str:
            nonlocal scale_replacements
            raw_value = match.group(2)
            parts = [part for part in re.split(r"[\s,]+", raw_value.strip()) if part]
            if len(parts) != 3:
                raise RuntimeError(f"unexpected mesh scale in {source_urdf}: {raw_value!r}")
            values = [float(part) for part in parts]
            if not np.isfinite(values).all():
                raise RuntimeError(f"nonfinite mesh scale in {source_urdf}: {raw_value!r}")
            if "," in raw_value:
                scale_replacements += 1
            return match.group(1) + " ".join(parts) + match.group(3)

        patched_text = mesh_scale_pattern.sub(normalize_scale, source_text)
        size_replacements = 0
        if spec["template_name"] == "kitchen_counter":
            size_replacements = patched_text.count(bad_counter_size)
            if size_replacements != 1:
                raise RuntimeError(
                    "expected exactly one known malformed kitchen-counter box vector; "
                    f"found {size_replacements}"
                )
            patched_text = patched_text.replace(bad_counter_size, fixed_counter_size)
        overlay_urdf.write_text(patched_text)
        spec["asset_path"] = overlay_urdf
        compat_patches.append(
            {
                "template_name": spec["template_name"],
                "source_urdf": str(source_urdf),
                "source_urdf_sha256": sha256_file(source_urdf),
                "patched_urdf": str(overlay_urdf),
                "patched_urdf_sha256": sha256_file(overlay_urdf),
                "mesh_scale_delimiter_replacements": scale_replacements,
                "malformed_counter_size_replacements": size_replacements,
                "numeric_values_changed": False,
            }
        )

    # Validate all patched files with the exact bundled parser before asking
    # Genesis to build the complete layout, so any remaining contract issue is
    # attributed to a specific asset rather than a late scene failure.
    from genesis.ext.urdfpy import URDF

    for spec in art_specs:
        parsed = URDF.load(str(spec["asset_path"]))
        if not parsed.links:
            raise RuntimeError(f"patched URDF contains no links: {spec['asset_path']}")

    gs.init(backend=gs.cpu, precision="32", logging_level="warning")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, -9.81)),
            vis_options=gs.options.VisOptions(ambient_light=(0.55, 0.55, 0.55)),
            renderer=gs.renderers.Rasterizer(),
            show_viewer=False,
        )
        scene.add_entity(
            gs.morphs.Mesh(
                file=str(stage_asset),
                pos=(0.0, 0.0, 0.0),
                quat=(1.0, 0.0, 0.0, 0.0),
                scale=1.0,
                visualization=True,
                collision=False,
                fixed=True,
                file_meshes_are_zup=False,
            )
        )
        for spec in rigid_specs:
            scene.add_entity(
                gs.morphs.Mesh(
                    file=str(spec["asset_path"]),
                    pos=tuple(spec["position_g"]),
                    quat=tuple(spec["quaternion_g_wxyz"]),
                    scale=spec["scale"],
                    visualization=True,
                    collision=False,
                    fixed=True,
                    file_meshes_are_zup=False,
                )
            )
        for spec in art_specs:
            scene.add_entity(
                gs.morphs.URDF(
                    file=str(spec["asset_path"]),
                    pos=tuple(spec["position_g"]),
                    quat=tuple(spec["quaternion_g_wxyz"]),
                    scale=spec["scale"],
                    fixed=True,
                    visualization=True,
                    collision=False,
                    file_meshes_are_zup=False,
                )
            )

        # Habitat frame: camera at (0.5, 1.55, 7.1), looking toward (0.5, 1.1, 2.0).
        camera = scene.add_camera(
            res=(320, 240),
            pos=tuple(habitat_xyz_to_genesis_xyz([0.5, 1.55, 7.1])),
            lookat=tuple(habitat_xyz_to_genesis_xyz([0.5, 1.1, 2.0])),
            fov=72,
            GUI=False,
            near=0.03,
            far=30.0,
        )
        scene.build()
        # Match the Stage-4 HDR overlay's separate render passes: the scene-linear
        # radiance request and the geometry depth/segmentation pass must not be
        # conflated with the ordinary 8-bit display RGB output.
        radiance, _, _, _ = camera.render(
            rgb=True, depth=False, segmentation=False, radiance=True, force_render=True
        )
        rgb, _, _, _ = camera.render(
            rgb=True, depth=False, segmentation=False, force_render=True
        )
        _, depth, segmentation, _ = camera.render(
            rgb=False, depth=True, segmentation=True, force_render=True
        )
        radiance = np.asarray(radiance)
        rgb = np.asarray(rgb)
        depth = np.asarray(depth)
        segmentation = np.asarray(segmentation)
        if radiance.shape != (240, 320, 3) or not np.isfinite(radiance).all():
            raise RuntimeError(f"invalid scene-linear radiance output shape/values: {radiance.shape}")
        if rgb.shape != (240, 320, 3) or not np.isfinite(rgb).all():
            raise RuntimeError(f"invalid RGB output shape/values: {rgb.shape}")
        if depth.shape != (240, 320) or not np.isfinite(depth).all():
            raise RuntimeError(f"invalid depth output shape/values: {depth.shape}")
        if segmentation.shape != (240, 320):
            raise RuntimeError(f"invalid segmentation shape: {segmentation.shape}")
        if rgb.dtype == np.uint8 or float(rgb.max()) > 1.5:
            rgb_u8 = np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)
        else:
            rgb_u8 = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb_u8, mode="RGB").save(args.output_dir / "apt_1_rgb.png")
        np.save(args.output_dir / "apt_1_radiance.npy", radiance.astype(np.float32, copy=False))
        np.save(args.output_dir / "apt_1_depth.npy", depth)
        np.save(args.output_dir / "apt_1_segmentation.npy", segmentation)
        result = {
            "schema": "genesis_replicad_apt1_cpu_import_preview_v1",
            "status": "passed_one_static_layout_camera_render",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "genesis_version": gs.__version__,
            "backend": str(gs.backend),
            "device": str(gs.device),
            "cuda_visible_devices": "empty",
            "offline_cache_redirected_to_run_local_tmp": True,
            "renderer_overlay": "/home/zhaoyuhan/Eventbased_WAM/runtime/sim2real/genesis_hdr_v2",
            "radiance_signal": {
                "request": "camera.render(rgb=True, radiance=True, force_render=True)",
                "payload": "scene-linear float32 from the isolated Genesis HDR rasterizer overlay",
                "physical_radiometric_calibration": "unverified",
                "shape": list(radiance.shape),
                "finite_fraction": float(np.isfinite(radiance).mean()),
                "range": [float(radiance.min()), float(radiance.max())],
            },
            "scene_id": "apt_1",
            "scene_config": str(scene_path),
            "scene_config_sha256": sha256_file(scene_path),
            "stage_asset": str(stage_asset),
            "stage_asset_sha256": sha256_file(stage_asset),
            "source_archive_sha256": "d571c2c1315601c7faed35452e54a15c56646393a4c330235f20fb6fc8d8edfd",
            "coordinate_contract": {
                "habitat_world": "right-handed Y-up; rotation coefficients XYZW",
                "genesis_world": "right-handed Z-up; quaternion WXYZ",
                "basis_matrix": COORD_H_TO_G.tolist(),
                "position_mapping": "(x, y, z)_habitat -> (x, -z, y)_genesis",
                "rotation_mapping": "q_g = q_basis * q_h * inverse(q_basis); q_basis is +90deg about X",
                "glb_meshes": "file_meshes_are_zup=False; Genesis applies the documented Y-up to Z-up mesh conversion",
            },
            "counts": {
                "stage_meshes": 1,
                "rigid_scene_instances": len(rigid_specs),
                "unique_rigid_templates": len({s["template_name"] for s in rigid_specs}),
                "articulated_urdf_instances": len(art_specs),
                "entities_added": 1 + len(rigid_specs) + len(art_specs),
            },
            "camera": {
                "resolution_hw": [240, 320],
                "position_habitat": [0.5, 1.55, 7.1],
                "lookat_habitat": [0.5, 1.1, 2.0],
                "fov_degrees": 72,
                "intrinsics": np.asarray(camera.intrinsics).tolist(),
            },
            "outputs": {
                "rgb_shape": list(rgb.shape),
                "rgb_range": [float(rgb.min()), float(rgb.max())],
                "radiance_shape": list(radiance.shape),
                "radiance_dtype": str(radiance.dtype),
                "radiance_range": [float(radiance.min()), float(radiance.max())],
                "depth_shape": list(depth.shape),
                "depth_finite_fraction": float(np.isfinite(depth).mean()),
                "depth_positive_fraction": float((depth > 0).mean()),
                "segmentation_shape": list(segmentation.shape),
                "segmentation_ids": np.unique(segmentation).tolist(),
            },
            "articulated_instances": [
                {**spec, "asset_path": str(spec["asset_path"])} for spec in art_specs
            ],
            "run_local_compatibility_patches": compat_patches,
            "source_assets_modified": False,
            "rigid_instance_assets": [
                {k: v for k, v in s.items() if k != "asset_path"} | {"asset_path": str(s["asset_path"])}
                for s in rigid_specs
            ],
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "single static render only; no physical trajectory or ground-truth flow",
                "no event generation, training, or real data",
                "articulated URDF joint state is its importer default and was not animated",
                "this is an import/appearance pilot, not a validated scene conversion or coverage pass",
            ],
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        print(json.dumps({"status": result["status"], "counts": result["counts"],
                          "rgb_range": result["outputs"]["rgb_range"],
                          "depth_positive_fraction": result["outputs"]["depth_positive_fraction"],
                          "segmentation_ids": result["outputs"]["segmentation_ids"],
                          "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
