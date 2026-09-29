#!/usr/bin/env python3
"""Fail-closed Phase 4.7 readiness gate.

``--preflight`` runs only loader and implementation smokes.  It never invokes
the formal metric code, never searches a time offset, and never writes a formal
result.  A failed live smoke is a STOP condition, not a partial benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


EXPECTED_CORE = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
EXPECTED_PROTOCOL = "v3-frozen-preformal-readiness"
PRIMARY_METRICS = {
    "event_rate_relative_error",
    "polarity_ratio_error",
    "active_pixel_ratio_error",
    "spatial_distribution_l1_8x8",
}


def load_yaml(path: Path) -> dict:
    import yaml
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"{path}: YAML root must be a mapping")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def check_protocol_history(repo: Path, protocol_path: Path) -> dict:
    v1 = repo / "benchmark/protocol_v1.yaml"
    v2 = repo / "benchmark/protocol_v2.yaml"
    v1_expected = "b1b4519446bbc821220b3502a0eb3811bf8c0baf8bd457eca022a0d082ebdd97"
    v2_sidecar = repo / "benchmark/protocol_v2.sha256"
    require(sha256(v1) == v1_expected, "protocol_v1 bytes changed")
    require(v2_sidecar.read_text(encoding="utf-8").split()[0] == sha256(v2), "protocol_v2 bytes/hash changed")
    current_hash = sha256(protocol_path)
    sidecar = repo / "benchmark/protocol_v3.sha256"
    require(sidecar.is_file(), "protocol_v3.sha256 is missing")
    require(sidecar.read_text(encoding="utf-8").split()[0] == current_hash, "protocol_v3 sidecar hash mismatch")
    return {
        "protocol_v1_sha256": sha256(v1),
        "protocol_v2_sha256": sha256(v2),
        "protocol_v3_sha256": current_hash,
    }


def check_git_and_core(repo: Path) -> dict:
    status = git(repo, "status", "--porcelain")
    require(not status, "worktree is not clean; commit the readiness files before preflight")
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", EXPECTED_CORE, "--", "genesis_event_plugin/"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    require(diff.returncode == 0, "genesis_event_plugin/ differs from frozen core e79e656")
    return {"head": git(repo, "rev-parse", "HEAD"), "frozen_core": EXPECTED_CORE, "core_diff": "empty"}


def check_configs(repo: Path) -> dict:
    ledger = load_json(repo / "benchmark/configs/CONFIG_HASHES.json")
    require(ledger.get("algorithm") == "sha256", "config hash ledger algorithm mismatch")
    result = {}
    for name, recorded in ledger.get("files", {}).items():
        actual = sha256(repo / "benchmark/configs" / name)
        require(actual == str(recorded).removeprefix("sha256:"), f"config hash mismatch: {name}")
        result[name] = actual
    require(set(result) == {"ours.yaml", "v2e.yaml", "esim.yaml"}, "unexpected config set")
    return result


def check_pair_and_replay(repo: Path, pair_path: Path) -> dict:
    pair = load_yaml(pair_path)
    replay_path = repo / "benchmark/replays/checkerboard_translation_v3.yaml"
    replay = load_yaml(replay_path)
    require(pair.get("status") == "preformal_readiness_only", "pair manifest is not v3 preformal")
    require(pair.get("protocol") == "benchmark/protocol_v3.yaml", "pair manifest protocol mismatch")
    require(pair.get("shared_replay") == "benchmark/replays/checkerboard_translation_v3.yaml", "pair replay mismatch")
    require(set(pair.get("primary_metrics", [])) == PRIMARY_METRICS, "primary metric set changed")
    require(pair["qualitative_windows"]["aggregate_metric_eligible"] is False, "5 ms windows entered aggregate metrics")
    pairs = pair.get("pairs", [])
    require([p.get("direction") for p in pairs] == ["LR", "RL"], "v3 must contain exactly LR and RL pairs")
    for item in pairs:
        require(item["genesis_replay"]["definition"] == "benchmark/replays/checkerboard_translation_v3.yaml", "wrong replay definition")
        require(float(item["real_reference"]["duration_s"]) == float(item["genesis_replay"]["duration_s"]), f"full duration mismatch: {item['pair_id']}")
    text = replay_path.read_text(encoding="utf-8").lower()
    for forbidden in ("placeholder", "definition_only_pending", "tbd"):
        require(forbidden not in text, f"replay still contains forbidden placeholder marker: {forbidden}")
    require(replay.get("status") == "executable_scene_definition", "replay is not executable")
    for key in ("board", "camera", "board_pose", "motion", "render", "outputs"):
        require(key in replay, f"replay missing executable field: {key}")
    require(replay["render"]["native_radiance_required"] is True, "replay does not require native radiance")
    require(replay["render"]["display_rgb_fallback"] == "forbidden", "display RGB fallback is not closed")
    sidecar = repo / "benchmark/evaluation_pairs_v3.sha256"
    require(sidecar.read_text(encoding="utf-8").split()[0] == sha256(pair_path), "pair v3 sidecar hash mismatch")
    return {"sha256": sha256(pair_path), "pair_ids": [p["pair_id"] for p in pairs], "replay_sha256": sha256(replay_path)}


def check_metric_and_time_contract(protocol: dict) -> dict:
    require(protocol.get("protocol_version") == EXPECTED_PROTOCOL, "protocol is not v3 readiness protocol")
    require(protocol.get("formal_run_allowed") is False, "formal gate must remain disabled")
    metrics = set(protocol["primary_evaluation"]["metrics"])
    require(metrics == PRIMARY_METRICS, "primary metric definitions changed")
    disabled = protocol["primary_evaluation"]["disabled"]
    require(all(key in disabled for key in ("CD", "GD", "temporal_rate_error_per_5ms_bin")), "CD/GD/temporal metric is enabled")
    qualitative = protocol["primary_evaluation"]["qualitative_only"]
    require(qualitative["allowed_in_formal_aggregate"] is False, "5 ms windows are not qualitative-only")
    require(protocol["time_origin"]["method_alignment"]["per_method_offset_search"] is False, "temporal offset search enabled")
    require(protocol["time_origin"]["method_alignment"]["first_event_alignment"] is False, "first-event alignment enabled")
    return {"primary_metrics": sorted(metrics), "CD": "disabled", "GD": "disabled", "temporal_offset_search": "disabled"}


def run_smoke(repo: Path, args: list[str], log_path: Path) -> dict:
    completed = subprocess.run([sys.executable, *args], cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    require(completed.returncode == 0, f"smoke failed: {' '.join(args)}; see {log_path}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"smoke did not emit JSON: {log_path}") from exc


def preflight(repo: Path, protocol_path: Path, pair_path: Path, smoke_root: Path) -> dict:
    protocol = load_yaml(protocol_path)
    history = check_protocol_history(repo, protocol_path)
    core = check_git_and_core(repo)
    contracts = check_metric_and_time_contract(protocol)
    configs = check_configs(repo)
    pairs = check_pair_and_replay(repo, pair_path)
    smoke_root.mkdir(parents=True, exist_ok=True)

    # Discover the installed official ECF filter before the smoke imports h5py.
    # This keeps the gate portable across Metavision package layouts without
    # changing the declared source or activating a decoded fallback.
    from benchmark.io.real_stream import discover_ecf_plugin
    ecf_plugin_path = discover_ecf_plugin()
    real = run_smoke(repo, ["benchmark/scripts/real_timestamp_origin_smoke.py"], smoke_root / "real_timestamp_origin.log")
    genesis = run_smoke(repo, ["benchmark/scripts/genesis_replay_smoke.py", "--direction", "LR", "--duration", "0.30", "--resolution", "64x48", "--output-dir", str(smoke_root / "genesis_LR")], smoke_root / "genesis_hdr_ours.log")
    v2e = run_smoke(repo, ["benchmark/scripts/v2e_official_pipeline_smoke.py", "--frames", str(smoke_root / "genesis_LR/frames"), "--output-dir", str(smoke_root / "v2e_LR")], smoke_root / "v2e_official_pipeline.log")
    require(real.get("status") == "PASS", "real LR/RL loader smoke did not pass")
    require(genesis.get("status") == "PASS" and genesis.get("native_radiance") is True, "Genesis HDR/Ours smoke did not pass")
    require(genesis.get("duration_s") == 0.30, "active readiness smoke duration is not the frozen 0.30 s")
    require(genesis.get("segmentation_expected_entity_resolution_coverage") == 1.0, "not all checkerboard entities resolved by segmentation_idx_dict")
    require(not genesis.get("segmentation_motion_map_background_overlap"), "background entered the motion map")
    require(v2e.get("status") == "PASS" and v2e.get("official_pipeline") is True, "official V2E pipeline smoke did not pass")
    return {
        "status": "preflight_pass",
        "formal_benchmark_executed": False,
        "formal_metrics_executed": False,
        "history": history,
        "core": core,
        "contracts": contracts,
        "configs": configs,
        "pairs": pairs,
        "real_LR_RL_read_smoke": real,
        "ecf_plugin_path": ecf_plugin_path,
        "Genesis_HDR_smoke": genesis,
        "Ours_v3_1_smoke": genesis,
        "V2E_official_full_pipeline_smoke": v2e,
        "smoke_root": str(smoke_root),
        "next_action": "wait for human approval of Phase 5",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("benchmark/protocol_v3.yaml"))
    parser.add_argument("--pair-manifest", type=Path, default=Path("benchmark/evaluation_pairs_v3.yaml"))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-root", type=Path, default=Path("/tmp/genesis_event_benchmark_phase4_7"))
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    repo = protocol.parents[1]
    try:
        if not args.preflight:
            raise RuntimeError("formal benchmark is disabled; use --preflight for readiness smokes")
        result = preflight(repo, protocol, args.pair_manifest.resolve(), args.smoke_root.resolve())
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        result = {
            "status": "STOP",
            "formal_benchmark_executed": False,
            "formal_metrics_executed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
