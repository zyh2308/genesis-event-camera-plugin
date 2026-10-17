#!/usr/bin/env python3
"""Exercise the real launcher gate with negative receipt/config/data controls."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("repo-root", "config", "bank-root", "v2e-root", "data-root", "model-root", "genesis-root", "receipt", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    for name in ("repo_root", "config", "bank_root", "v2e_root", "data_root", "model_root", "genesis_root", "receipt", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    gate = args.repo_root / "benchmark/sim2real/scripts/stage4_launch_gate.py"
    base = [sys.executable, str(gate), "--phase", "pilot", "--receipt", str(args.receipt),
            "--repo-root", str(args.repo_root), "--config", str(args.config),
            "--bank-root", str(args.bank_root), "--v2e-root", str(args.v2e_root),
            "--data-root", str(args.data_root), "--model-root", str(args.model_root),
            "--genesis-root", str(args.genesis_root), "--run-root", str(args.output / "should_not_exist"),
            "--generator", "eventgenesis", "--max-updates", "1",
            "--test-only-sentinel"]
    env = os.environ.copy(); env["STAGE4_GATE_TEST_MODE"] = "1"
    cases = []
    with tempfile.TemporaryDirectory(prefix="stage4_gate_negative_") as temp:
        tmp = Path(temp)
        valid = json.loads(args.receipt.read_text())
        case_inputs = []
        case_inputs.append(("missing_receipt", args.receipt.parent / "missing_preflight_result.json", args.config, args.data_root))

        config_copy = tmp / "stale_config.json"
        stale_config = json.loads(args.config.read_text()); stale_config["negative_control_marker"] = "changed"
        config_copy.write_text(json.dumps(stale_config))
        case_inputs.append(("stale_config_fingerprint", args.receipt, config_copy, args.data_root))

        stale_report = json.loads(json.dumps(valid))
        stale_report["checks"] = [row for row in stale_report["checks"] if row.get("check") != "correlation_operator_cuda"]
        report_missing_check = tmp / "missing_required_check.json"; report_missing_check.write_text(json.dumps(stale_report))
        case_inputs.append(("missing_required_check", report_missing_check, args.config, args.data_root))

        fail_report = json.loads(json.dumps(valid))
        row = next(row for row in fail_report["checks"] if row.get("check") == "renderer_segmentation_mapping")
        row["status"] = "FAIL"
        report_failed_check = tmp / "injected_failed_check.json"; report_failed_check.write_text(json.dumps(fail_report))
        case_inputs.append(("injected_failed_check", report_failed_check, args.config, args.data_root))

        data_copy = tmp / "stale_data"; data_copy.mkdir()
        adapter_manifest = json.loads((args.data_root / "training_bank_manifest.json").read_text())
        adapter_manifest["negative_control_marker"] = "changed"
        (data_copy / "training_bank_manifest.json").write_text(json.dumps(adapter_manifest))
        case_inputs.append(("stale_data_manifest_fingerprint", args.receipt, args.config, data_copy))

        for index, (name, receipt, config, data) in enumerate(case_inputs):
            sentinel = args.output / f"sentinel_{index}.txt"
            forbidden_run_root = args.output / "should_not_exist"
            cmd = base.copy()
            cmd[cmd.index(str(args.receipt))] = str(receipt)
            cmd[cmd.index(str(args.config))] = str(config)
            cmd[cmd.index(str(args.data_root))] = str(data)
            cmd += [str(sentinel)]
            proc = subprocess.run(cmd, env=env, text=True, capture_output=True)
            blocked = proc.returncode != 0 and not sentinel.exists() and not forbidden_run_root.exists()
            cases.append({"case": name, "exit_code": proc.returncode,
                          "launch_branch_reached": sentinel.exists(), "blocked_before_process": blocked,
                          "run_root_created": forbidden_run_root.exists(),
                          "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:], "command": cmd})
            if not blocked:
                raise RuntimeError(f"negative control did not block before trainer branch: {name}")
    result = {"schema": "stage4_launcher_negative_controls_v1", "status": "PASS",
              "cases": cases, "all_blocked_before_process": True}
    out = args.output / "launcher_negative_controls.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
