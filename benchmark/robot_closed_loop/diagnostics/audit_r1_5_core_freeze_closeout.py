#!/usr/bin/env python3
"""Audit the already-run R1.5 closeout evidence without rerunning Genesis.

This script verifies result manifests, acceptance gates, and that the four
candidate core modules are byte-identical across the CUDA, renderer, causal,
CPU, and current source commits. It does not produce new simulation evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


CORE_FILES = (
    "genesis_event_plugin/physics_interpolator.py",
    "genesis_event_plugin/physics_plugin.py",
    "genesis_event_plugin/physics_warp.py",
    "genesis_event_plugin/torch_physics.py",
)
RUNS = {
    "cuda_F": {
        "run_id": "r1_5_cuda_support_forensics_20261031f",
        "source_commit": "9e6b5d7e2689d541087a56ff1fbf0fc904571b69",
        "summary": "summary.json",
    },
    "hdr_I": {
        "run_id": "r1_5_strict_direct_rerender_20261031i",
        "source_commit": "26de5e410ecc550a9dc7f61b25aa2926d6399e0a",
        "summary": "output/summary.json",
    },
    "causal_K": {
        "run_id": "r1_5_live_causal_hdr_20261031k",
        "source_commit": "7559893ea4168b979679ee32eac03806050a9b03",
        "summary": "output/summary.json",
    },
    "cpu_B": {
        "run_id": "r1_5_core_candidate_final_20261102b",
        "source_commit": "e71f2b7b7369994d714f540ff498f3902b90081c",
        "summary": "output/summary.json",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_manifests(run_root: Path) -> int:
    manifests = sorted(run_root.rglob("SHA256SUMS"))
    require(bool(manifests), f"no SHA256SUMS below {run_root}")
    checked = 0
    for manifest in manifests:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            require(len(line) > 66 and line[64:66] == "  ", f"invalid checksum row: {manifest}: {line}")
            expected, relative = line[:64], line[66:].lstrip("* ")
            target = (manifest.parent / relative).resolve()
            require(target.is_relative_to(manifest.parent.resolve()), f"checksum path escapes run: {target}")
            require(target.is_file(), f"missing checksum target: {target}")
            actual = sha256(target)
            require(actual == expected, f"checksum mismatch: {target}: {actual} != {expected}")
            checked += 1
    return checked


def core_hashes(root: Path, revision: str) -> dict[str, str]:
    result = {}
    for relative in CORE_FILES:
        payload = subprocess.check_output(["git", "show", f"{revision}:{relative}"], cwd=root)
        result[Path(relative).name] = hashlib.sha256(payload).hexdigest()
    return result


def audit(root: Path) -> dict:
    branch = git(root, "branch", "--show-current")
    require(branch == "exp/r1_5-core-candidate-20261030", f"unexpected branch: {branch}")
    head = git(root, "rev-parse", "HEAD")
    results = root / "benchmark/robot_closed_loop/results"

    run_summaries = {}
    checked_files = {}
    revisions = {"current_HEAD": head}
    for label, spec in RUNS.items():
        run_root = results / spec["run_id"]
        require(run_root.is_dir(), f"missing run directory: {run_root}")
        require(git(root, "cat-file", "-e", f"{spec['source_commit']}^{{commit}}") == "", "")
        require(
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", spec["source_commit"], head], cwd=root
            ).returncode == 0,
            f"{label} source commit is not an ancestor of HEAD",
        )
        require((run_root / "README.md").is_file(), f"missing run report for {label}")
        checked_files[label] = verify_manifests(run_root)
        summary_path = run_root / spec["summary"]
        require(summary_path.is_file(), f"missing summary: {summary_path}")
        run_summaries[label] = json.loads(summary_path.read_text(encoding="utf-8"))
        revisions[label] = spec["source_commit"]

    module_hashes = {name: core_hashes(root, rev) for name, rev in revisions.items()}
    reference_hashes = module_hashes["current_HEAD"]
    require(
        all(hashes == reference_hashes for hashes in module_hashes.values()),
        "candidate core module hashes differ across audited runs",
    )

    f = run_summaries["cuda_F"]
    require(f["acceptance_result"]["all_gates_pass"], "CUDA F acceptance failed")
    require(f["acceptance_result"]["threshold_dvs_cpu_cuda_exact"], "CUDA threshold probe mismatch")
    for relative in ("genesis_event_plugin/physics_warp.py", "genesis_event_plugin/torch_physics.py"):
        require(
            f["source_sha256"][relative] == reference_hashes[Path(relative).name],
            f"CUDA F source hash does not match current module: {relative}",
        )

    i = run_summaries["hdr_I"]
    i_episodes = i["episodes"]
    render_event_counts = {}
    for episode in ("4", "5"):
        data = i_episodes[episode]
        comparison = data["event_comparison"]
        require(comparison["pixel_polarity_l1"] == 0, f"HDR I episode {episode}: event count mismatch")
        require(
            comparison["exact_time_pixel_polarity_matches_1ns"] == comparison["reference_count"],
            f"HDR I episode {episode}: event match failure",
        )
        require(len(data["frame_records"]) == 9, f"HDR I episode {episode}: expected nine interior frames")
        require(
            all(frame["radiance_luma_max_abs"] == 0 and frame["segmentation_mismatch_pixels"] == 0
                for frame in data["frame_records"]),
            f"HDR I episode {episode}: direct HDR frame mismatch",
        )
        render_event_counts[episode] = comparison["reference_count"]
    runtime = i["execution"]["runtime"]
    require(runtime["camera_radiance_parameter"], "HDR I runtime did not expose radiance")
    require(i["execution"]["renderer_backend"] == "rasterizer", "HDR I was not rendered by rasterizer")

    k = run_summaries["causal_K"]
    require(k["status"] == "PASS" and k["causal_order_gate"], "causal K order gate failed")
    require(not k["physics_state_overwrite_during_episode"], "causal K overwrote physics state")
    require(k["aggregate"]["all_event_times_no_later_than_right_endpoint"], "event timestamp exceeded packet")
    require(k["aggregate"]["all_provider_samples_previously_observed"], "causal K used an unobserved sample")
    require(len(k["episodes"]) == 2, "causal K did not contain two trajectories")
    require(
        all(ep["event_only_action_applied_next_step"] for ep in k["episodes"].values()),
        "causal K did not apply event-driven action on next step",
    )
    require(
        k["aggregate"]["renderer_and_module_hashes"]["camera"] == runtime["renderer_module_sha256"]["camera"],
        "HDR renderer camera module differs between I and K",
    )

    b = run_summaries["cpu_B"]
    require(b["source_module_sha256"] == reference_hashes, "CPU B source module hashes differ from current core")
    controls = b["unit_controls"]
    require(controls["exact_endpoint_identity"]["max_abs_radiance_delta"] == 0, "endpoint control failed")
    require(controls["exact_endpoint_identity"]["torch_max_abs_radiance_delta"] == 0, "Torch endpoint failed")
    require(controls["numpy_torch_cpu_parity"]["support_mismatch_pixels"] == 0, "CPU support parity failed")
    require(controls["numpy_torch_cpu_parity"]["max_abs_luma_delta"] == 0, "CPU radiance parity failed")
    ramp = b["known_log_linear_ramp"]
    require(ramp["static_scene_event_count"] == 0, "static control emitted events")
    require(ramp["candidate_event_count"] == 48, "known ramp event count changed")
    require(ramp["max_abs_timing_error_s"] <= 1e-8, "known ramp event timing failed")
    replays = b["strict_cached_direct_hdr_replays"]
    require(len(replays) == 2, "CPU B cached replay is incomplete")
    require(
        all(item["exact_image_sequence_match"] and item["raw_events_exact_match"] for item in replays),
        "CPU B direct-HDR replay mismatch",
    )

    return {
        "audit_status": "PASS",
        "branch": branch,
        "current_source_commit": head,
        "candidate_core_module_sha256": reference_hashes,
        "module_hashes_identical_across": list(revisions),
        "runs": {
            "cuda_F": {"source_commit": RUNS["cuda_F"]["source_commit"], "all_gates_pass": True,
                       "checksummed_files": checked_files["cuda_F"]},
            "hdr_I": {"source_commit": RUNS["hdr_I"]["source_commit"], "episodes": render_event_counts,
                      "strict_extra_render_count": i["aggregate"]["strict_extra_sample_count"],
                      "render_plus_readback_wall_s": i["aggregate"]["sum_camera_render_call_wall_s_including_gl_readback"],
                      "checksummed_files": checked_files["hdr_I"]},
            "causal_K": {"source_commit": RUNS["causal_K"]["source_commit"], "physics_steps": k["aggregate"]["physics_steps"],
                         "episodes": len(k["episodes"]), "causal_order_gate": True,
                         "checksummed_files": checked_files["causal_K"]},
            "cpu_B": {"source_commit": RUNS["cpu_B"]["source_commit"], "ramp_events": ramp["candidate_event_count"],
                      "cached_replays": len(replays), "checksummed_files": checked_files["cpu_B"]},
        },
        "strict_profile_sha256": sha256(root / "benchmark/robot_closed_loop/configs/r1_5_core_strict_sampled_20261102.json"),
        "claim_limits": [
            "strict sampled linear-HDR behavior only; changes faster than observed anchors remain unsupported",
            "offline render cost exceeds the 1 ms simulated sample interval; no wall-clock real-time claim",
            "localized floor appearance in causal K is not uniquely attributed to shadow without a matched live shadow-off run",
            "no EVK4 absolute radiometric calibration, old-checkpoint recovery, or grasp-success claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(output)
    summary = audit(root)
    output.mkdir(parents=True)
    summary["audit_script_sha256"] = sha256(Path(__file__).resolve())
    summary["audited_at_utc"] = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# R1.5 core closeout evidence audit\n\n"
        "This is a manifest-level audit of the existing CUDA, HDR rerender, causal, and CPU validation runs. "
        "It does not rerun Genesis, DVS, physics, or training. The audit confirms that the candidate core "
        "modules are byte-identical across all four run source commits and the current source commit, and "
        "that their preserved result manifests and stated acceptance gates verify. See `summary.json`.\n",
        encoding="utf-8",
    )
    rows = []
    for path in sorted(output.iterdir()):
        if path.is_file():
            rows.append(f"{sha256(path)}  {path.name}\n")
    (output / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")
    print(json.dumps({"audit_status": summary["audit_status"], "output": str(output),
                      "current_source_commit": summary["current_source_commit"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
