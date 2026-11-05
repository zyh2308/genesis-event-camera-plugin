# MVSEC 本地来源核查及恢复传输建议

仅进行了本地目录、字节数和 SHA256 只读核查；未 SSH、未启动传输、未读候选权重、未评分。机器收据为同目录 `local_mvsec_data_hash_audit.json`。7 个文件的完整 SHA256 重算用时 42.10 秒，**7/7 与事前登记一致**。

注册文件 `stage9_mvsec_indoor_unscored_registration.json` 为 22,075,595 字节，SHA256 `46e0141bde84b8c3c3cc96615cec247f44bb8488ae33bdaf626d4f23d8dca02f`，保持原样，不重新选窗或改写源数据路径。

## 可直接补传的本地来源

统一目录：`/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/indoor_flying/`。这里不是 `MVSEC-public-v2`；不能因后者在本机不存在就判断文件未下载。

| 文件 | 本地及登记字节数 | 完整 SHA256 | 主会话最近提供的 g1 观测 |
| --- | ---: | --- | --- |
| indoor_flying1_data.hdf5 | 1700819167 | `9e8315403758cf000c80e4884c0d1d898b4362d6026cdcf2ae668971c5160eab` | 540147712 字节，不能作为完整文件使用 |
| indoor_flying1_gt.hdf5 | 3370770337 | `cc3c6bc8fe6b99b944dcb867ca0dca3b143998b2c694d18ce8819ddcadd7015d` | 未找到 |
| indoor_flying2_data.hdf5 | 2532552285 | `a547babf4c20cbd558fd59d936cec588d3308761b8b2cb436a11e348ecc3d608` | 未找到 |
| indoor_flying2_gt.hdf5 | 4284255701 | `a76038611c37a4492a6de9ccf8e2fca2c43e4364f03c976345e1da65dee5dcfb` | 3931013120 字节，不能作为完整文件使用 |
| indoor_flying3_data.hdf5 | 2625560959 | `9706cd43d95b7249f56f385af622524e2893514166af9b651711c924d24240e9` | 未找到 |
| indoor_flying3_gt.hdf5 | 4454252233 | `b22e4ff90d853d1237a40aff375b61171dd95b1623bb399644120c54c5c006e5` | 大小相符，仍须远端完整 SHA256 |
| indoor_flying_calib.zip | 1294736 | `2bc61565a2e53af174c7afc956c9148dea96c55a765b95f06e558e7d726d1f94` | 未确认 |

6 HDF5 加校准共 18,969,505,418 字节。若 g1 第 3 条 GT 完整 SHA256 相同，其余 5 HDF5 加校准为 14,515,253,185 字节，可通过已有残件减少传输，但最后仍应完整校验。不把大小异常直接称为损坏：它们可能是未完成或仍在传输的文件，需要主会话核对写入进程。

HDF5 的已注册正式目标统一为：`/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2/indoor_flying/`。

**校准的登记 `path_on_g1` 遗留了本地 `/home/科研/...` 路径。** 这是登记的路径元数据问题，不能凭此断言远端校准已在位，也不应为了匹配字段擅自创建另一用户 home。可将相同校准字节安全传到 g1 正式数据目录并另存路径解析回执；保留原登记不变。当前桥接按注册的事件体/GT/mask 哈希逐窗复核，实际不读取校准 ZIP；若将来评价入口需要它，应在研究冻结前显式登记解析方式。

## 已授权的直接续传入口

主会话后续明确允许完成已有下载残件，要求尽量续传。已据此新增 `resume_mvsec_transfer.py`：默认仅离线计划；只有 `--execute --writer-check-confirmed` 同时给出才连接固定 g1。它对每个旧目标完整记录 hash：正确文件跳过；不足长度的文件仅在与本地同长度前缀 SHA 一致时续传；同长错 hash、超长、不同前缀或 symlink 保留原样并拒绝。写入限于本页固定 ASCII 输入目录，采用 `rsync --partial --append-verify` 和远端 rsync 期间的 `flock`；不修改训练或结果。这里的 `--append-verify` 会直接完成已授权的短文件，下面 staging 方案仅作为异常前缀时的备用，**不必为了正确残件重新传一份**。

CPU 自测 10 项和实际注册离线计划均退出 0，收据为 `transfer_launcher_cpu_selftest.json`、`transfer_launcher_offline_plan.json`、`transfer_launcher_source_receipt.json`。运行进程 PID、每次 ssh/rsync 的 stdout/stderr/PID/exit、旧/新 hash 和最终 `exit.json` 均存新 `--out`。下列执行应由主会话在确认没有其他写入者后运行：

```bash
python3 '/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/mainline_audit_20261105/flow/resume_mvsec_transfer.py' \
  --execute --writer-check-confirmed \
  --out '/mnt/c/Users/zhaoyuhan/Documents/ChatGPT/New project/outputs/mainline_audit_20261105/flow/transfer_run_20261105_v1'
```

必须使用新的 `--out`；再次接续可选新目录，旧记录保留。支持重复 `--file` 选择登记文件，`--bwlimit-kib` 限速。launcher 依赖现有 WSL rsync、远端 rsync/flock 和 Windows OpenSSH，不安装或修改其配置。独立执行须用现有已授权的后台启动方式，避免与终端生命周期绑定；本子会话没有启动它。

## 异常前缀时的 staging 备用方案

1. 核实已有传输及写入进程、磁盘空间。若存在正在写入同一文件的传输，不并发启动第二个写入者。先完整 hash 核验 g1 `indoor_flying3_gt.hdf5`，相同就保留复用。
2. 在 g1 正式数据目录旁新建唯一 staging 目录，例如 `MVSEC-public-v2.incoming-20261105`，不要用现有正式文件作 `--inplace` 目标。若希望复用旧残件，在确认无写入者后用 reflink/copy 将残件复制到 staging；旧文件保留，不能将未核实文件直接发布。
3. 从上述本地完整来源恢复到 staging。可使用已有 SSH 配置的 g1 别名和 rsync；示例仅供主会话适配现有代理/限速，不是已经执行的事实：

```bash
SRC='/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/indoor_flying/'
DST='/home/zhaoyuhan/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-public-v2.incoming-20261105/indoor_flying/'
rsync -rt --partial --append-verify --protect-args --info=progress2 \
  "$SRC" "g1:$DST"
```

`--append-verify` 允许续传 staging 中的短文件，并在内容校验不符时重新传输；不与 `--ignore-existing` 同用，否则短文件会被跳过。若双方 rsync 不具备此能力，沿用已验证的 resumable 工具，仍保持 staging、非覆盖和完整 SHA256 这三个条件。

4. 完成后按上表核对 g1 staging 中每个文件的完整 SHA256 和字节数。只将合格文件发布到注册的正式目标：目标不存在时采用不可覆盖发布；目标是旧残件时先保留旧字节到明确的 quarantine 名称，再以同文件系统原子替换为已验证版本，记录两者 hash，不删除旧证据。若目标在此期间改变或被其他作业使用，停止该文件的发布并重新核实。
5. 注册 JSON 也须保留精确字节及 SHA256，复制到桥接 README 指定的新位置，再运行无评分 `prepare`。评价仍须等完整 15 权重矩阵、研究冻结及 hash 绑定授权；完成数据传输本身不授权评分。

## outdoor_day1 尚不能由本地恢复

对 `/home/科研/Eventbased_WAM/` 做了只读同名文件搜索，未发现任何 `*outdoor_day1*` 文件。两个已有下载清单分别为：

- `/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-download-v2/public_download_manifest.json`：2026-10-20 的 `outdoor_day1_data.hdf5` 下载失败。
- `/home/科研/Eventbased_WAM/data/sim2real/stage5/real/MVSEC-outdoor-retry-20261023/public_download_manifest.json`：2026-10-22 重试仍因 Google Drive 下载配额失败。

失败记录中的官方 data 文件 ID 为 `1JLIrw2L24zIQBmqaWvef7G2t9tsMY3H0`；不是已验证可用下载。本轮未发网络请求，也未启动下载。未发现对应 GT 或 calibration 文件，因此不能仅修复 data 后声称 outdoor 已齐。

这只证明已检查的 WSL 项目范围内不存在，不涵盖尚未检查的 Windows 各盘。后续应由主会话确认是否有其他既有完整副本，或重新走官方 HDF5 获取链路，补齐 data+GT+calibration 并事前注册窗口。不能静默从 required4 去掉 outdoor，也不能未经几何/时间审计用 ROS bag 替换 HDF5 来拼出结果。
