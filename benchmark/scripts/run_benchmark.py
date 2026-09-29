#!/usr/bin/env python3
"""Fail-closed entry point and preflight gate for benchmark protocol v2.

Phase 4.6 uses ``--preflight`` only.  The gate validates provenance and the
frozen execution contract but never computes the formal metrics.  No CLI flag
can override methods, metrics, windows, parameters, cadence, or seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import re


EXPECTED_CORE = "e79e656ac00f7847b98db9e12bf43b093f3afff1"
EXPECTED_PHASE45 = "46ce49b941d36d2c145bd9e369dc8ef53362b8cc"
EXPECTED_PROTOCOL_VERSION = "v2-frozen"
EXPECTED_DISABLED_METRICS = {
    "temporal_rate_error_per_5ms_bin",
    "chamfer_distance_cd",
    "gaussian_distance_gd",
    "composite_score",
    "ranking_score",
}


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to parse the frozen benchmark protocol") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("protocol root must be a mapping")
    return data


def current_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def git_status(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"STOP: {message}")


def check_core_integrity(repo: Path) -> dict[str, object]:
    current = current_commit(repo)
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", "--exit-code", EXPECTED_CORE, "--", "genesis_event_plugin/"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    require(diff.returncode == 0, "genesis_event_plugin/ differs from frozen core e79e656")
    return {"head": current, "frozen_core": EXPECTED_CORE, "core_diff": "empty"}


def check_approval(repo: Path, protocol: dict) -> dict[str, object]:
    path = repo / "benchmark/FORMAL_RUN_APPROVAL.yaml"
    approval = load_yaml(path)
    require(bool(approval.get("approved")), "formal approval file does not approve Phase 4.6")
    require(approval.get("approved_protocol") == "benchmark/protocol_v2.yaml", "approval targets another protocol")
    require(approval.get("frozen_core_commit") == EXPECTED_CORE, "approval frozen-core commit mismatch")
    require(approval.get("phase4_5_commit") == EXPECTED_PHASE45, "approval Phase 4.5 commit mismatch")
    require(protocol.get("formal_run_allowed") is False, "protocol must remain closed before final confirmation")
    return approval


def check_config_hashes(repo: Path) -> dict[str, str]:
    ledger_path = repo / "benchmark/configs/CONFIG_HASHES.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    require(ledger.get("algorithm") == "sha256", "config hash ledger algorithm mismatch")
    actual: dict[str, str] = {}
    for name, recorded in ledger.get("files", {}).items():
        path = repo / "benchmark/configs" / name
        expected = str(recorded).removeprefix("sha256:")
        actual[name] = sha256(path)
        require(actual[name] == expected, f"config hash mismatch: {name}")
    require(set(actual) == {"ours.yaml", "v2e.yaml", "esim.yaml"}, "unexpected config set")
    return actual


def check_pair_manifest(repo: Path, protocol: dict) -> dict[str, object]:
    manifest_path = repo / "benchmark/evaluation_pairs_v2.yaml"
    manifest = load_yaml(manifest_path)
    require(manifest.get("status") == "frozen_before_formal_execution", "pair manifest is not frozen")
    require(manifest.get("protocol") == "benchmark/protocol_v2.yaml", "pair manifest protocol mismatch")
    window = manifest.get("evaluation_window", {})
    expected_window = protocol["evaluation_windows"]
    require(float(window.get("duration_s")) == float(expected_window["duration_s"]), "window duration mismatch")
    require(list(window.get("normalized_fractions", [])) == list(expected_window["normalized_fractions"]), "window fractions mismatch")
    require(window.get("same_for_real_ours_v2e") is True, "pair windows are not shared")
    pairs = manifest.get("pairs", [])
    require([p.get("real_reference", {}).get("direction") for p in pairs] == ["LR", "RL"], "formal pair directions changed")
    for pair in pairs:
        require(pair.get("genesis_replay", {}).get("definition") == "benchmark/replays/checkerboard_translation_v2.yaml", "replay definition mismatch")
        require(pair.get("ours_output", {}).get("method_id") == "ours_v3_1_direct_real_v2", "Ours pair method mismatch")
        require(pair.get("v2e_output", {}).get("method_id") == "v2e_official_core", "V2E pair method mismatch")
        require(float(pair["real_reference"]["duration_s"]) == float(pair["genesis_replay"]["duration_s"]), f"duration mismatch in {pair.get('pair_id')}")
    return {"sha256": sha256(manifest_path), "pair_ids": [p["pair_id"] for p in pairs]}


def check_v2e_contract(repo: Path, protocol: dict) -> dict[str, object]:
    spec = protocol["methods"]["v2e"]["formal_input_cadence"]
    config = load_yaml(repo / "benchmark/configs/v2e.yaml")
    require(config["implementation"]["version"] == protocol["methods"]["v2e"]["version"], "V2E version mismatch")
    require(float(config["parameters"]["cutoff_hz"]) == float(spec["cutoff_hz"]), "V2E cutoff changed")
    source = Path("/home/科研/Eventbased_WAM/code/v2e_source/setup.py")
    require(source.exists(), "V2E official source setup.py is missing")
    text = source.read_text(encoding="utf-8")
    require(re.search(r'version\s*=\s*["\']1\.5\.1["\']', text) is not None, "V2E source version is not 1.5.1")
    return {
        "version": config["implementation"]["version"],
        "effective_cadence_hz": float(spec["effective_cadence_hz"]),
        "timestamp_resolution_s": float(spec["timestamp_resolution_s"]),
        "official_source": str(source),
    }


def preflight(repo: Path, protocol_path: Path, protocol: dict) -> dict[str, object]:
    require(protocol.get("protocol_version") == EXPECTED_PROTOCOL_VERSION, "protocol is not v2-frozen")
    require(protocol.get("formal_run_allowed") is False, "formal gate must remain disabled")
    require(not git_status(repo), "worktree is not clean; commit protocol changes before preflight")
    core = check_core_integrity(repo)
    approval = check_approval(repo, protocol)
    configs = check_config_hashes(repo)
    pairs = check_pair_manifest(repo, protocol)
    v2e = check_v2e_contract(repo, protocol)
    disabled = set(protocol["metrics"]["disabled"])
    require(disabled == EXPECTED_DISABLED_METRICS, "disabled metric set changed")
    replay = load_yaml(repo / "benchmark/replays/checkerboard_translation_v2.yaml")
    require(replay.get("formal_replay_gate", {}).get("require_hdr_overlay") is True, "HDR replay gate missing")
    require(replay.get("formal_replay_gate", {}).get("display_rgb_substitution_allowed") is False, "display RGB fallback is not closed")
    protocol_hash = sha256(protocol_path)
    protocol_sidecar = repo / "benchmark/protocol_v2.sha256"
    require(protocol_sidecar.read_text(encoding="utf-8").split()[0] == protocol_hash, "protocol v2 sidecar hash mismatch")
    pair_sidecar = repo / "benchmark/evaluation_pairs_v2.sha256"
    require(pair_sidecar.read_text(encoding="utf-8").split()[0] == pairs["sha256"], "pair manifest sidecar hash mismatch")
    return {
        "status": "preflight_pass",
        "formal_benchmark_run": False,
        "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
        "approval": {"approved": approval["approved"], "scope": approval["approval_scope"]},
        "core": core,
        "configs": configs,
        "pairs": pairs,
        "v2e": v2e,
        "disabled_metrics": sorted(disabled),
        "hdr_formal_replay": "required_but_not_executed_in_phase_4_6",
        "formal_metrics": "not_run",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight or run only the protocol-approved benchmark.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="validate v2 provenance without running metrics")
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    repo = protocol.parents[1]
    data = load_yaml(protocol)
    if args.preflight:
        report = preflight(repo, protocol, data)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    protocol_sha = sha256(protocol)
    print("Formal benchmark is disabled by protocol confirmation gate.")
    print(f"protocol={protocol}")
    print(f"protocol_sha256={protocol_sha}")
    print("No formal result files were created.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
