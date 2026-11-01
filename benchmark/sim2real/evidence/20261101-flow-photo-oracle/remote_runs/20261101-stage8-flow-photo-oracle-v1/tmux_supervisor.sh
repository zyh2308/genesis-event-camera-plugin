#!/usr/bin/env bash
set +e
bash /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-flow-photo-oracle-v1/run_stage7_gt_zero_reverse_photometric_g1.sh
RC=$?
printf '%s\n' "$RC" > /home/zhaoyuhan/Eventbased_WAM/runs/sim2real/20261101-stage8-flow-photo-oracle-v1/tmux_exit_code.txt
sleep 5
exit "$RC"
