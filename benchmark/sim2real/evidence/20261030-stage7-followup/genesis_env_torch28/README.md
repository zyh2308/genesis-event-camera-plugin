# Isolated Genesis/PyTorch Environment Update

This is setup evidence only. It does not report a Genesis scene render, model training, or an experiment result.

## Result

- The shared `Eventbased_WAM` conda environment was left unchanged at its prior PyTorch version.
- Only the cloned environment `/home/zhaoyuhan/miniconda3/envs/Eventbased_WAM_torch28` was updated.
- g1 run03 completed at `2026-10-30T12:06:35Z` with exit code 0. CPU import checks reported PyTorch `2.8.0+cu126` / CUDA build `12.6`, torchvision `0.23.0+cu126`, torchaudio `2.8.0+cu126`, Genesis `1.2.2`, and its bundled pyrender path. Genesis import no longer emitted the earlier unsupported-PyTorch warning.
- No CUDA context, renderer, or real data was used by the import check.
- pip reported pre-existing missing optional requirements for `tensordict` (`orjson`) and `robomimic` (`egl_probe`, `tensorboardX`). Genesis and its bundled pyrender imported successfully; these packages were not added because they are not needed for the current scene-import probe.

## Artifact provenance

All large wheels remain outside Git under:

`/home/zhaoyuhan/Eventbased_WAM/external_sources/pytorch_wheels/stage7-20261030/`

| Wheel | SHA256 |
|---|---|
| torch 2.8.0+cu126, CPython 3.11 | `542cdbf042aaf5d6ddbed43cb8cd8c4df1e586bebb1338f5dcba14fa52830d3c` |
| torchvision 0.23.0+cu126, CPython 3.11 | `b697f5ca72399822d6454929cfb8835298527881e0ba58982f80ff0401331a92` |
| torchaudio 2.8.0+cu126, CPython 3.11 | `77859ec6c6a92ca3b666a3f02363a5631e9676c027fb4dce99a461a089c9962c` |
| triton 3.4.0, CPython 3.11 | `7b70f5e6a41e52e48cfc087436c8a28c17ff98db369447bcaff3b887a3ab4467` |

Torch/vision/audio came from the official [PyTorch CUDA 12.6 wheel index](https://download.pytorch.org/whl/cu126/); the Triton wheel and hash came from [PyPI Triton 3.4.0](https://pypi.org/project/triton/3.4.0/). Each file was hash-checked before installation. The CUDA 12.6 wheel set matches the official [PyTorch previous-version installation table](https://docs.pytorch.org/get-started/previous-versions/).

## Runs and transfer findings

- g1 `20261030-stage7-genesis-torch28-setup-run01`: direct PyTorch CDN install attempt, stopped by this session at exit 143 after observing roughly 58 KB/s for the 822 MB torch wheel.
- g1 `20261030-stage7-genesis-torch28-setup-run02`: local verified torch wheel was used, but PyTorch-index torchvision/torchaudio/Triton transfers measured roughly 26–30 KB/s. This attempt was stopped at exit 143 before pip reached installation; the cloned environment still reported PyTorch 2.5.1 afterward.
- Attempts to send the wheel from WSL to g1 over Windows SSH/SCP reached only about 0.35 MB/s. Their partial sizes and paths are recorded in run02 `transfer_attempts.log`; only those task-owned partial files were removed.
- g3 direct HTTPS downloads were substantially faster: the 821,911,765-byte torch wheel completed and matched its expected hash; the three remaining wheels were then downloaded from direct official links and checked against the PyTorch index/PyPI SHA256 values. Download logs and hashes are in the g3 run directories below.
- g1 `20261030-stage7-genesis-torch28-setup-run03`: installed the four checked wheels plus CUDA runtime dependencies in the cloned environment; installation and CPU imports exited 0. Full logs, package snapshots, timestamps, and exit codes are at the g1 run path.

## Resource boundary

A later sequential refresh at `2026-10-30T12:13:18Z` on g1 and `12:13:21Z` on g3 again found other-user processes on all GPUs. The full per-GPU snapshot, UUIDs, utilization, memory, and visible process owners are in `resource_refresh_20261030T121318Z.txt`. Earlier, g3 GPU2 briefly appeared unused at `12:07:21Z`, but a `12:09:10Z` recheck found two other-user V2E processes on it. No reservation was made and no renderer/training job was started. The first actual ReplicaCAD render still requires a fresh free-GPU check and a reservation.

## Reproduction

From the repository root, the setup launchers used were:

```bash
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_g3_download.sh
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_g3_slow_wheels.sh
bash benchmark/sim2real/evidence/20261030-stage7-followup/genesis_env_torch28/start_run03.sh
```

The environment installation driver rechecks CPU/memory, confirms no process is using the cloned environment, validates wheel hashes, records GPU ownership without allocating a GPU, and performs CPU-only imports. The corresponding remote command output is retained at the run paths below.

## CPU-only importer API check

At `2026-10-30T12:16:14Z`, the cloned environment exposed Genesis `gs.morphs.Mesh` controls for GLB/Z-up parsing, coordinate convention, transforms, visualization, and collision. CUDA was hidden. This was package API introspection only; it did not load a ReplicaCAD file or initialize a renderer. The command, limitations, and remote log hashes are recorded in `genesis_api_inspection_20261030T121600Z.txt`.
