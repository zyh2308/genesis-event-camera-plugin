#!/usr/bin/env python3
"""Attach verified phase evidence to a fingerprinted preflight receipt."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from stage4_launch_gate import build_fingerprint


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--status", choices=("PASS", "FAIL", "NOT_RUN"), required=True)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--v2e-root", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--genesis-root", type=Path, required=True)
    args = p.parse_args()
    report_path = args.report.resolve()
    if not args.evidence.resolve().is_file():
        raise FileNotFoundError(f"preflight check evidence file is missing: {args.evidence}")
    report = json.loads(report_path.read_text())
    report["checks"] = [row for row in report.get("checks", []) if row.get("check") != args.name]
    report["checks"].append({"check": args.name, "status": args.status,
                             "evidence": str(args.evidence.resolve()), "command": args.command})
    current_fingerprint = build_fingerprint(
        args.repo_root.resolve(), args.config.resolve(), args.bank_root.resolve(), args.v2e_root.resolve(),
        args.data_root.resolve(), args.model_root.resolve(), args.genesis_root.resolve())
    if report.get("fingerprint") != current_fingerprint:
        raise RuntimeError("cannot append evidence: preflight receipt code/data/config/environment fingerprint is stale; rerun full checks")
    report["fingerprint"] = current_fingerprint
    statuses = {row["status"] for row in report["checks"]}
    report["overall_status"] = "PASS" if "FAIL" not in statuses and "NOT_RUN" not in statuses else "BLOCKED"
    tmp = report_path.with_suffix(report_path.suffix + ".partial")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(tmp, report_path)
    print(json.dumps({"updated": args.name, "status": args.status,
                      "overall_status": report["overall_status"],
                      "fingerprint_sha256": report["fingerprint"]["fingerprint_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
