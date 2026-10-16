#!/usr/bin/env python3
"""Publish a concise live training handoff from remote run records."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_last_update(path: Path):
    """Read the newest optimizer record without loading the full run log."""
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            stream.seek(max(0, size - 65536))
            lines = stream.readlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and "optimizer_update" in record:
            return record
    return None


def read_arm_status(run_root: Path, name: str):
    arm_root = run_root / "flow_dt1_seed0" / name
    checkpoint_status = read_json(arm_root / "live_status.json") or {}
    update = read_last_update(arm_root / "training_metrics.jsonl")
    if update is None:
        return checkpoint_status or None
    # Per-update metrics are fresher than the checkpoint sidecar, which is
    # intentionally written only at update 20 and each 100 updates.
    status = {**checkpoint_status, **update}
    status["completed_updates"] = update["optimizer_update"]
    status["status"] = checkpoint_status.get("status", "training")
    status["updated_utc"] = update.get("updated_utc", checkpoint_status.get("updated_utc"))
    return status


def render(args) -> str:
    bank_progress = read_json(args.bank_root / "generation_progress.json")
    v2e_root = args.run_root / "v2e_events"
    shard_progress = [read_json(v2e_root / f"gpu{gpu}" / "shard_progress.json") for gpu in (1, 2, 4, 5)]
    v2e_manifest = read_json(v2e_root / "v2e_complete_manifest.json")
    adapter_manifest = read_json(args.adapter_root / "training_bank_manifest.json")
    arms = {name: read_arm_status(args.run_root, name)
            for name in ("eventgenesis", "v2e")}
    now = datetime.now(timezone.utc).isoformat()
    prior_attempts = []
    for prior in sorted(args.run_root.parent.glob("20261016-stage4-flowdt1-seed0-v*")):
        if prior.resolve() == args.run_root.resolve():
            continue
        exit_path = prior / "exit_code.txt"
        if not exit_path.is_file():
            continue
        try:
            exit_code = exit_path.read_text().strip()
        except OSError:
            exit_code = "unreadable"
        finished_path = prior / "finished_utc.txt"
        finished = finished_path.read_text().strip() if finished_path.is_file() else "running/unknown"
        prior_attempts.append((prior, exit_code, finished))
    def progress_line():
        if bank_progress is None:
            return "未开始。"
        return (f"{bank_progress.get('completed_rollouts', 0)}/{bank_progress.get('total_rollouts', 32)} rollouts; "
                f"状态 `{bank_progress.get('status', 'unknown')}`; 更新时间 `{bank_progress.get('updated_utc', '—')}`")
    lines = [
        "# Stage 4 训练实时状态", "",
        f"最后刷新：`{now}`", "",
        "## 固定配置", "",
        f"- 分支：`{args.branch}`；代码提交：`{args.commit}`。",
        f"- 配置：`{args.config}`（SHA-256 `{args.config_sha}`）。",
        f"- 合成 bank：`{args.bank_root}`；计划 32×2 s，346×260，100 Hz endpoints，1 ms physics。",
        "- 真实数据评分/反馈：关闭。两组只用各自合成事件输入和共享合成灰度/标签。",
        "", "## 共享物理 bank", "", f"- 进度：{progress_line()}.",
    ]
    if prior_attempts:
        lines.extend(["", "## Previous attempts", ""])
        for prior, exit_code, finished in prior_attempts:
            lines.append(f"- `{prior}`：campaign exit code `{exit_code}`，结束/更新时间 `{finished}`；启动器和标准错误见 `/home/zhaoyuhan/Eventbased_WAM/logs/sim2real_stage4_flowdt1_seed0_tmux.log`。")
    if bank_progress and bank_progress.get("latest_rollout"):
        lines.append(f"- 最近 rollout：`{bank_progress['latest_rollout']}`；已完成数由 `generation_progress.json` 记录。")
    if bank_progress and bank_progress.get("active_rollout"):
        lines.append(f"- 当前 rollout：`{bank_progress['active_rollout']}` endpoint {bank_progress.get('active_endpoint')}/{bank_progress.get('active_total_endpoints')}；近期吞吐 {bank_progress.get('active_endpoint_rate_hz')} endpoint/s。")
    if (args.bank_root / "bank_manifest.json").is_file():
        m = read_json(args.bank_root / "bank_manifest.json") or {}
        lines.append(f"- 完整 shared HDF5 SHA-256：`{m.get('bank_sha256', '—')}`；split SHA-256：`{m.get('split_sha256', '—')}`。")
    lines.extend(["", "## V2E 25 μs 导出", ""])
    if v2e_manifest:
        lines.append(f"- 完成：32 个 rollout；manifest `{v2e_root / 'v2e_complete_manifest.json'}`，SHA-256 `{args.hash_file(v2e_root / 'v2e_complete_manifest.json')}`。")
    else:
        for gpu, shard in zip((1, 2, 4, 5), shard_progress):
            if shard:
                active = f"，当前 `{shard['active_rollout']}`" if shard.get("active_rollout") else ""
                lines.append(f"- GPU {gpu}：{shard.get('completed_rollouts', 0)}/{shard.get('requested_rollouts', 8)}，状态 `{shard.get('status')}`{active}。")
            else:
                lines.append(f"- GPU {gpu}：尚未开始。")
    lines.extend(["", "## 30 ms STE-FlowNet seed 0", ""])
    if adapter_manifest:
        lines.append(f"- 两组 adapter：{adapter_manifest.get('train_windows_per_generator')} train / {adapter_manifest.get('validation_windows_per_generator')} synthetic-validation windows；manifest `{args.adapter_root / 'training_bank_manifest.json'}` SHA-256 `{args.hash_file(args.adapter_root / 'training_bank_manifest.json')}`。")
    else:
        lines.append("- Adapter 尚未完成；尚无 Stage 4 optimizer update。")
    for name in ("eventgenesis", "v2e"):
        status = arms[name]
        if status is None:
            lines.append(f"- **{name}**：尚未训练。")
        else:
            lines.append(
                f"- **{name}**：`{status.get('status')}`，GPU `{status.get('physical_gpu_id', '—')}`，"
                f"optimizer updates `{status.get('completed_updates', status.get('optimizer_update', 0))}/{status.get('total_updates', 8910)}`；"
                f"loss `{status.get('loss', '—')}`（photo `{status.get('photometric_loss', '—')}`；smooth `{status.get('smoothness_loss', '—')}`）；"
                f"有效 batch `{status.get('effective_batch', '—')}`；checkpoint `{status.get('checkpoint', '—')}`；"
                f"SHA-256 `{status.get('checkpoint_sha256', '—')}`；更新时间 `{status.get('checkpoint_updated_utc', status.get('updated_utc', '—'))}`。"
            )
            if status.get("update20_restore_verified"):
                lines.append(f"  - update 20 checkpoint reload/optimizer/scheduler probe 通过；prediction max abs `{status.get('restore_prediction_max_abs')}`。")
    lines.extend(["", "## 运行与证据", "",
                  f"- Run root：`{args.run_root}`。运行日志、PID、exit code 和 checkpoint 均留在服务器 run root。",
                  f"- 最近刷新器 PID：`{args.watcher_pid}`。",
                  "- 当前结果仅证明训练链路和更新确实发生；不构成 generator 优劣或 Sim-to-Real 结论。", ""])
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", type=Path, required=True)
    p.add_argument("--bank-root", type=Path, required=True)
    p.add_argument("--adapter-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha", required=True)
    p.add_argument("--branch", required=True)
    p.add_argument("--commit", required=True)
    p.add_argument("--poll-seconds", type=int, default=15)
    args = p.parse_args()
    args.watcher_pid = __import__("os").getpid()
    args.hash_file = lambda path: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    output = args.repo_root / "benchmark/sim2real/TRAINING_LIVE_STATUS.md"
    while True:
        output.write_text(render(args))
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    main()
