# R0.5 Direct-HDR GT-shadow reference

This nine-episode run used code commit `2100645b0473c294760789bc4475b644f261d9cf` and config SHA256 `dcc681fa57ed85b38871455b9ddbb736af57a5c37fab5b6bffb9356b2d86c0cd`.

The controller receives a 10 ms delayed current-GT state only to create complete reference trajectories. The paired event streams are sensor/evaluator inputs and do not control the robot. Do not count these episodes as event-controller successes.

This repository copy includes the per-episode CSV/JSON/JSONL metrics and state/control traces, HDR key snapshots, and GIFs. The nine larger `direct_hdr_events_1khz_2khz.npz` files remain in the original server result directory:

`g1:/home/zhaoyuhan/Eventbased_WAM/repositories/gec-robot-r0_5-39b235a/benchmark/robot_closed_loop/results/r0_5_direct_hdr_gt_shadow_20261016/`

Use `SHA256SUMS.local-artifacts` to verify files included here. The original `SHA256SUMS` is the full server checksum list and includes the raw event arrays; it was verified on g1.
