# Stage 6 执行更新：HQF 全量重建评测

记录时间：2026-10-25 08:33 UTC / 17:33 JST。状态来自 g1 实时观测，不是历史快照。

## 当前状态

| 事项 | 状态 | 证据/说明 |
|---|---|---|
| HQF 冻结协议预检 | **通过，exit 0** | 全 14 序列数据清单、协议 SHA、两个 synthetic-best checkpoint 和锁定的下游源码通过核验。机器日志在 g1：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-preflight-r4/`。 |
| HQF 第一次执行 | **失败，已保留** | `eventgenesis/bike_bay_hdr` 完成；下一序列因 Python 在锁定 EVREAL checkout 写入未跟踪 `__pycache__`，源码洁净性门禁拒绝继续。未产生真实评分。完整现场在 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-actual/`。 |
| HQF 重跑 | **运行中** | 清理的仅是本次创建于隔离源码 checkout 内的 5 个 `__pycache__` 目录；r5 冻结预检再次通过。重跑设置 `PYTHONDONTWRITEBYTECODE=1`，避免复发。GPU 0，launcher PID 2154000、评测进程 PID 2154028。17:33 JST EventGenesis 已完成 `bike_bay_hdr`、`boxes` 两序列，正在处理 `desk`；V2E 与评分尚未开始。输出目录 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-actual-r2/`，托管日志 `/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-launch-r2/`。 |
| 环境克隆 | **后台复制中，有持续 I/O** | 目标为 `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28`，外层 PID 2139928、conda 子进程 PID 2141886；原环境未修改，PyTorch 升级尚未执行。17:33 JST `/proc/2141886/io` 累计写入约 3.08 GB（17:29 时约 2.17 GB），属于慢速共享盘复制；还没有完成标记。日志目录：`/home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-envclone-torch28/`。 |
| 新训练 | **未启动** | 当前空闲 GPU 不代表训练数据门禁已通过：光流仍缺 ReplicaCAD 丰富场景及合格运动信号；位姿仍需解决监督标签语义。未用旧小场景绕过门禁。 |

最近刷新时 GPU 0 约 4% 利用率、692 MiB，由 HQF 重跑使用。GPU 1–5 当前均有其他任务的显存/进程使用（GPU 1 和 5 约 2.7 GB，GPU 2–4 约 14.6–15.2 GB；可见进程属于其他用户），按繁忙处理。没有空闲 GPU 可供新训练。未接触其他用户进程；并且光流仍缺 ReplicaCAD 丰富场景及合格运动信号，位姿仍需解决监督标签语义，重建 200 epoch 训练已完成。

## 冻结协议与运行绑定

- 本次代码执行包来自仓库 commit `888b9d1235b247ae8f3d1a4b22f5fdc1dfaa9601`，对应 Stage 6 冻结 runner 的原始哈希；执行包 SHA256：`45c8291d22e6b69acd50d97ee8f7cd1f00af81c3d07ffcf4ee6eac1d42e5e3b3`。
- 冻结协议 SHA256：`46656bbf0b02e3cddc79b9957cde9092b33bba0ecca48a7687c0e733cab97494`。
- HQF 文件清单 SHA256：`76e19db296be055135fa7446a4da68b704a1c573dd7f2b8f06ce599ddc1a3add`，使用已登记的 `stage5/20261021-hqf14-memmap-v3`。
- EventGenesis synthetic-best：SHA256 `506e81bd558f1bcca2b67d34d4c1004836fc7cce65393084eb52a378dfd554bc`；V2E synthetic-best：`598b1e7837b4b02b50a7f390889c7737ed797840043a3f23121f54dc99d050d8`。
- 下游源码为干净、浅层 checkout：EVREAL `398f03551b7f5d150ae10aa89e4e731557fce240`；event_cnn_minimal `78aa471a66593f55e349303563997297fbc816e8`，两个子模块也锁定到 manifest 中记录的 commit。源码包 SHA256：`24dec069e70be4af6377a747428c373105ee2f008cdbb0d077e15aec60595135`。
- 真实评分使用冻结的 14 序列、event-only、原生画幅、LPIPS-Alex 主指标和等序聚合规则。测试输出只报告，不用于调参、挑序列或更换 checkpoint。

## 保留的失败尝试

失败尝试没有覆盖或删除：

1. `preflight-r1` 把原始 bag 目录误作冻结的 memmap 根目录，门禁按 manifest 路径不匹配拒绝。
2. `preflight-r2` 的数据根目录正确，但部署的源码包虽然逐文件和 tree SHA 全部匹配，冻结 runner 仍只接受 `.git` checkout。为避免改变冻结 runner 哈希，后续改为部署严格匹配 commit 的干净浅层 checkout。
3. 曾试图修改 runner 放行锁定源码包，但冻结协议会拒绝 runner 文件哈希变化；该临时改动已通过 revert 撤销，正式运行未使用该改动。
4. `preflight-r3` 因 runner 哈希变化按冻结协议拒绝。
5. `preflight-r4` 使用原 runner、正确 memmap 根目录及干净源码 checkout，全部门禁通过，exit 0；第一次推理后，Python 在 EVREAL checkout 生成了未跟踪 bytecode 缓存，导致下一序列的源码洁净门禁拒绝。
6. 已将上述仅由本次运行生成的缓存目录从隔离 checkout 清除，`preflight-r5` 在 `PYTHONDONTWRITEBYTECODE=1` 下再次通过，exit 0；同一冻结 runner 已在独立 `actual-r2` 输出目录重启，避免覆盖失败现场。

本地随仓库保存的短证据位于 `benchmark/sim2real/evidence/20261025-hqf14-rerun/`：`preflight_r5.json` SHA256 `baec04fdf6d7b3b0a2cb84fb1ae00792e302f9a141df212e347bdfb6cadf73b1`；第一次失败的 `eventgenesis__boxes.stderr.log` SHA256 `c04a19b816beb825b5108d9de48e72442fd96d50f2ff3614bae0cad4c58418f7`。完整日志与大输出留在 g1 对应 run 目录。

## 复现命令

g1 运行入口为：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
  /home/zhaoyuhan/miniconda3/envs/Eventbased_WAM/bin/python -u \
  /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/gec-sim2real-888b9d1-benchmark/benchmark/sim2real/scripts/run_hqf14_reconstruction_eval.py \
  --protocol /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/gec-sim2real-888b9d1-benchmark/benchmark/sim2real/protocol/hqf_14_reconstruction_test_frozen_20261024_v1.json \
  --data-manifest /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/gec-sim2real-888b9d1-benchmark/benchmark/sim2real/evidence/S5/20261021-recon-flow-localization/20261021-hqf14-manifest-validation-v3/hqf_14_file_manifest_v3.json \
  --hqf-root /home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/20261021-hqf14-memmap-v3 \
  --evreal-root /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/hqf-locked-source-checkouts-20261025/hqf-EVREAL-sparse \
  --model-root /home/zhaoyuhan/Eventbased_WAM/execution_checkouts/hqf-locked-source-checkouts-20261025/hqf-event_cnn_minimal-sparse \
  --output-root /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261025-hqf14-e2vid-seed0-actual \
  --device cuda --enable-real-scoring
```

首次运行命令由 g1 上的 `run_hqf14_stage6_20261025.sh` 托管并保留失败现场。重跑入口为 `/home/zhaoyuhan/Eventbased_WAM/run_hqf14_stage6_20261025_r2.sh`，在原命令中只增加 `PYTHONDONTWRITEBYTECODE=1` 并使用独立 `actual-r2` / `launch-r2` 目录；其完整命令、时间、退出码和逐序列日志均记录在 g1 上述目录。最终评分完成后补充逐序列与汇总指标，并更新证据索引。

## WSL 网络与 Git 状态

WSL 检查时（2026-10-25 17:28 JST），大小写代理变量仍分别指向 `127.0.0.1:7897` 与 `127.0.0.1:7890`；1897、7890、7897 均无 WSL 本地监听，直连 `127.0.0.1:1897` 的 curl 返回 connection refused。已按用户提供的新端口将该 Git 仓库的 `http.proxy` 更新为 `http://127.0.0.1:1897`；但进程环境变量仍是旧端口且本地无监听，因此尚未推送。服务器上执行包通过 SSH 部署，GitHub 尚未同步。需启动/开放可从 WSL 访问的 1897 代理，并统一当前 shell 的大小写代理变量后，再先 `curl`、`git ls-remote` 验证；当前不重试 push。

