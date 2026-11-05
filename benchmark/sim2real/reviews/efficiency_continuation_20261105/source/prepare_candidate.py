"""Materialize an isolated full candidate from the verified archived v1 package.

This performs file hashing/copying only, never imports the package or builds CUDA.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checked_relative(value):
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"Unsafe manifest path: {value!r}")
    return relative


def file_row(root, relative):
    path = root.joinpath(*checked_relative(relative).parts)
    data = path.read_bytes()
    return {"path": relative, "size_bytes": len(data), "sha256": sha256(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    review = here.parent.parent
    parser.add_argument("--base-release", required=True, type=Path,
                        help="Existing complete archived index-reuse v1 package")
    parser.add_argument("--out", required=True, type=Path,
                        help="New or empty independent output directory")
    parser.add_argument("--manifest", type=Path,
                        default=review.parent / "sampler_cost_20261105/release/manifest.json")
    parser.add_argument("--delta", type=Path,
                        default=review / "release/genesis_event_plugin/gpu_geometry.py")
    args = parser.parse_args()
    base = args.base_release.resolve(strict=True)
    destination = args.out.resolve()
    manifest_path = args.manifest.resolve(strict=True)
    delta_path = args.delta.resolve(strict=True)
    if destination == base or base in destination.parents or destination in base.parents:
        raise ValueError("Output and source package paths must be disjoint")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Output must be absent or empty; existing files will not be overwritten")
    manifest_data = manifest_path.read_bytes()
    manifest = json.loads(manifest_data)
    rows = manifest["candidate_files"]
    paths = [row["path"] for row in rows]
    changed = "genesis_event_plugin/gpu_geometry.py"
    if len(set(paths)) != len(paths) or changed not in paths:
        raise ValueError("Invalid v1 manifest paths")
    # Validate every byte before creating/copying any output.
    checked = []
    for row in rows:
        actual = file_row(base, row["path"])
        if actual != row:
            raise ValueError(f"v1 source mismatch: {actual['path']}: {actual}")
        checked.append(actual)
    delta_data = delta_path.read_bytes()
    delta_sha = sha256(delta_data)
    if delta_sha == next(row["sha256"] for row in checked if row["path"] == changed):
        raise ValueError("Delta is identical to v1; this is not the new candidate")
    destination.mkdir(parents=True, exist_ok=True)
    for row in checked:
        relative = checked_relative(row["path"])
        source = base.joinpath(*relative.parts)
        target = destination.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    destination.joinpath(*checked_relative(changed).parts).write_bytes(delta_data)
    candidate_rows = [file_row(destination, path) for path in paths]
    actual_changes = [path for old, new, path in zip(checked, candidate_rows, paths) if old != new]
    if actual_changes != [changed]:
        raise RuntimeError(f"Unexpected changed files: {actual_changes}")
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(here.parent), "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    receipt = {
        "schema": "isolated_projection_storage_candidate_v1",
        "implementation_commit": commit,
        "base_release": str(base),
        "base_manifest": str(manifest_path),
        "base_manifest_sha256": sha256(manifest_data),
        "base_candidate_files": checked,
        "candidate_release": str(destination),
        "delta_source": str(delta_path),
        "delta_sha256": delta_sha,
        "candidate_files": candidate_rows,
        "actual_changed_files": actual_changes,
        "all_other_release_files_byte_identical": True,
        "automatic_production_promotion_performed": False,
        "default_cuda_strict_qualification": "failed_as_recorded_in_handoff",
        "change": "Isolated geometry delta at delta_source; qualification is recorded separately",
    }
    receipt_path = destination / "SOURCE_MANIFEST.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"release": str(destination), "manifest": str(receipt_path),
                      "delta_sha256": delta_sha, "changed_files": actual_changes}, indent=2))


if __name__ == "__main__":
    main()
