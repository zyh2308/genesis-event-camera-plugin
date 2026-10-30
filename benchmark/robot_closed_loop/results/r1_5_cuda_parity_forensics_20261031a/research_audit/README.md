# CUDA parity 审查入口

先读 REVIEW.md。audit.py 独立复核保存数组、哈希及 CPU 判定分项，audit.json 为结果，mismatch_pixels_cpu_trace.csv 保存 106 点的 CPU 侧线索。

共享入口：`/home/科研/Eventbased_WAM/notes/cuda_parity_review_2026-10-30/README.md`。

已有 WSL Python：`/home/科研/Eventbased_WAM/repositories/genesis-event-camera-plugin-bounce-catch-20261024/.venv-robot/bin/python`。复制本目录脚本到新的审查目录后运行 audit.py；输出在脚本目录，避免覆盖本次证据。不安装依赖，不使用 CUDA/renderer，不启动服务器任务。
