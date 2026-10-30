#!/usr/bin/env python3
"""Audit ReplicaCAD archive references and embedded GLB PBR payloads.

This is a static archive/configuration audit only. It does not import a scene
into Genesis or verify rendering, camera geometry, or motion labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import posixpath
import struct
import zipfile
from pathlib import Path


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_glb_json(payload: bytes, path: str) -> dict:
    if len(payload) < 20:
        raise ValueError(f"GLB is too short: {path}")
    magic, version, declared_length = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(payload):
        raise ValueError(f"Invalid GLB header/length: {path}")
    offset = 12
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk_end = offset + chunk_length
        if chunk_end > len(payload):
            raise ValueError(f"Truncated GLB chunk: {path}")
        chunk = payload[offset:chunk_end]
        offset = chunk_end
        if chunk_type == 0x4E4F534A:  # JSON
            return json.loads(chunk.decode("utf-8").rstrip("\x00 "))
    raise ValueError(f"GLB has no JSON chunk: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--scene", default="configs/scenes/apt_1.scene_instance.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive = args.archive.resolve()
    with zipfile.ZipFile(archive) as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failure: {bad_member}")
        names = set(zf.namelist())
        scene_bytes = zf.read(args.scene)
        scene = json.loads(scene_bytes)
        refs: list[dict] = []
        config_paths: list[str] = []
        missing_configs: list[str] = []

        stage_name = scene["stage_instance"]["template_name"]
        stage_config = f"configs/stages/{stage_name}.stage_config.json"
        config_paths.append(stage_config)
        if stage_config not in names:
            missing_configs.append(stage_config)
        else:
            stage = json.loads(zf.read(stage_config))
            stage_asset = posixpath.normpath(
                posixpath.join(posixpath.dirname(stage_config), stage["render_asset"])
            )
            refs.append({"role": "stage_render", "path": stage_asset, "exists": stage_asset in names})

        templates = sorted({row["template_name"] for row in scene["object_instances"]})
        for template in templates:
            object_name = template.rsplit("/", 1)[-1]
            config_path = f"configs/objects/{object_name}.object_config.json"
            config_paths.append(config_path)
            if config_path not in names:
                missing_configs.append(config_path)
                continue
            config = json.loads(zf.read(config_path))
            for role in ("render_asset", "collision_asset"):
                if not config.get(role):
                    continue
                asset_path = posixpath.normpath(
                    posixpath.join(posixpath.dirname(config_path), config[role])
                )
                refs.append({"role": role, "path": asset_path, "exists": asset_path in names})

        render_refs = [r for r in refs if r["role"] in ("stage_render", "render_asset")]
        render_paths = sorted({r["path"] for r in render_refs if r["exists"]})
        glb_stats = []
        for path in render_paths:
            doc = parse_glb_json(zf.read(path), path)
            materials = doc.get("materials", [])
            glb_stats.append({
                "path": path,
                "bytes": zf.getinfo(path).file_size,
                "materials": len(materials),
                "pbr_materials": sum("pbrMetallicRoughness" in m for m in materials),
                "images": len(doc.get("images", [])),
                "textures": len(doc.get("textures", [])),
            })

        missing_render = [r["path"] for r in render_refs if not r["exists"]]
        missing_collision = [r["path"] for r in refs if r["role"] == "collision_asset" and not r["exists"]]
        scene_bytes_hash = sha256_bytes(scene_bytes)
        stats_hash = sha256_bytes(json.dumps(glb_stats, sort_keys=True).encode())
        license_text = zf.read("LICENSE.txt").decode("utf-8", errors="replace")
        result = {
            "schema": "replicad_scene_archive_contract_audit_v1",
            "archive_path": str(archive),
            "archive_size_bytes": archive.stat().st_size,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "zip_crc": "pass",
            "scene_config": args.scene,
            "scene_config_sha256": scene_bytes_hash,
            "stage_template": stage_name,
            "object_instance_count": len(scene["object_instances"]),
            "unique_object_templates": len(templates),
            "config_count": len(config_paths),
            "missing_configs": missing_configs,
            "render_asset_count": len(render_paths),
            "missing_render_assets": missing_render,
            "missing_collision_assets": missing_collision,
            "render_assets_with_pbr": sum(s["pbr_materials"] > 0 for s in glb_stats),
            "render_assets_with_embedded_images": sum(s["images"] > 0 for s in glb_stats),
            "total_materials": sum(s["materials"] for s in glb_stats),
            "glb_stats_sha256": stats_hash,
            "archive_license_first_line": next(
                (line.strip() for line in license_text.splitlines() if line.strip()), ""
            ),
            "static_audit_only": True,
            "genesis_import_performed": False,
            "render_performed": False,
        }

    if missing_configs or missing_render or missing_collision:
        result["status"] = "fail"
    else:
        result["status"] = "pass_static_contract"
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        out = args.output.resolve()
        if out.exists():
            raise FileExistsError(f"refusing to overwrite audit output: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text, end="")
    if result["status"] != "pass_static_contract":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
