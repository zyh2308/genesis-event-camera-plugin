#!/usr/bin/env python3
"""Fail-closed entry point for the protocol-approved formal benchmark.

The command accepts the protocol as the only experiment specification. It does
not accept ad-hoc metric, window, seed, or method parameters. Until a human
confirms the frozen protocol and enables execution, it exits without creating
formal result files.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


EXPECTED_COMMIT = "e79e656ac00f7847b98db9e12bf43b093f3afff1"


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run only a protocol-approved formal benchmark.")
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = args.protocol.resolve()
    repo = protocol.parents[1]
    data = load_yaml(protocol)
    protocol_sha = hashlib.sha256(protocol.read_bytes()).hexdigest()

    if current_commit(repo) != EXPECTED_COMMIT:
        raise SystemExit("STOP: repository is not at the frozen Ours commit")
    if data.get("protocol_version") != "v1-frozen":
        raise SystemExit("STOP: protocol is not marked v1-frozen")
    if not bool(data.get("formal_run_allowed", False)):
        print("Formal benchmark is disabled by protocol confirmation gate.")
        print(f"protocol={protocol}")
        print(f"protocol_sha256={protocol_sha}")
        print("No formal result files were created.")
        return 2

    # The evaluator is intentionally not silently inferred from CLI arguments.
    # This branch is a stop point until the protocol is human-confirmed and the
    # real/sim input runner has passed the alignment gate.
    raise SystemExit("STOP: formal evaluator implementation must be enabled only after protocol confirmation")


if __name__ == "__main__":
    raise SystemExit(main())

