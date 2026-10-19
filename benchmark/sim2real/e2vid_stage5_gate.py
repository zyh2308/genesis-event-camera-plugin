"""Fingerprint and fail-closed helpers for the corrected E2VID Stage-5 run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


FROZEN_CORE = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
EXPECTED_E2VID_COMMIT = "78aa471a66593f55e349303563997297fbc816e8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_model_bundle(model_root: Path, bundle_path: Path) -> dict:
    bundle = json.loads(bundle_path.read_text())
    matches = [row for row in bundle["repositories"] if row["repo"] == "event_cnn_minimal"]
    if len(matches) != 1:
        raise RuntimeError("downstream source bundle must contain exactly one event_cnn_minimal pin")
    source = matches[0]
    if source["commit"] != EXPECTED_E2VID_COMMIT:
        raise RuntimeError(f"unexpected pinned E2VID commit: {source['commit']}")
    failures = []
    checked = []
    for item in source["files"]:
        path = model_root / item["path"]
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        actual_size = path.stat().st_size
        actual_sha = sha256_file(path)
        if actual_size != item["size_bytes"] or actual_sha != item["sha256"]:
            failures.append({"path": str(path), "expected_size": item["size_bytes"],
                             "actual_size": actual_size, "expected_sha256": item["sha256"],
                             "actual_sha256": actual_sha})
        checked.append(item["path"])
    if failures:
        raise RuntimeError(f"pinned E2VID source bundle mismatch: {failures[:5]}")
    return {
        "bundle_path": str(bundle_path.resolve()),
        "bundle_sha256": sha256_file(bundle_path),
        "repository": source["repo"], "commit": source["commit"],
        "included_tree_sha256": source["included_tree_sha256"],
        "checked_file_count": len(checked),
    }


def collect_data_files(data_root: Path, data_manifest: dict, verify: bool = True) -> dict[str, list[dict]]:
    rows = {"eventgenesis": [], "v2e": []}
    for item in data_manifest.get("rollout_rows", []):
        generator = item.get("generator")
        if generator not in rows or "path" not in item:
            continue
        path = Path(item["path"]).resolve()
        try:
            relative = path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"data manifest points outside registered data root: {path}") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        digest = sha256_file(path) if verify else None
        if verify and (size != int(item["size_bytes"]) or digest != item["sha256"]):
            raise RuntimeError(f"data artifact fingerprint mismatch: {path}")
        rows[generator].append({
            "relative_path": relative.as_posix(), "path": str(path), "size_bytes": size,
            "sha256": digest or item["sha256"], "rollout_id": item["rollout_id"],
            "split": item["split"], "flow_label_sha256": item["flow_label_sha256"],
        })
    for generator in rows:
        rows[generator].sort(key=lambda row: (row["split"], row["rollout_id"]))
        if len(rows[generator]) != 32:
            raise RuntimeError(f"expected 32 manifest-bound HDF5 files for {generator}, got {len(rows[generator])}")
        for split in ("train", "validation"):
            count = sum(row["split"] == split for row in rows[generator])
            expected = 24 if split == "train" else 8
            if count != expected:
                raise RuntimeError(f"expected {expected} {split} rollouts for {generator}, got {count}")
    paired = {}
    for generator in rows:
        for row in rows[generator]:
            paired.setdefault(row["rollout_id"], {})[generator] = row["flow_label_sha256"]
    if len(paired) != 32 or any(set(pair) != set(rows) or len(set(pair.values())) != 1 for pair in paired.values()):
        raise RuntimeError("EventGenesis/V2E files do not share all physical label hashes")
    return rows


def verify_data_rows(data_root: Path, expected_rows: list[dict]) -> None:
    failures = []
    for item in expected_rows:
        path = Path(item["path"]).resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f"registered training artifact outside data root: {path}") from exc
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
            continue
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != int(item["size_bytes"]) or digest != item["sha256"]:
            failures.append({"path": str(path), "size_bytes": size, "sha256": digest})
    if failures:
        raise RuntimeError(f"registered data fingerprint mismatch: {failures[:5]}")

