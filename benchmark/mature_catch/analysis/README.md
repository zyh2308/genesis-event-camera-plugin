# Fixed-policy event-catch analysis

`analyze_paired.py` (72 episodes), `analyze_cadence.py` (44 episodes), `audit_*` and `make_*_server.py` run on g1 with `--root` pointing at the corresponding archived dataset. They resolve explicit `RECOVERY_MAP.json` paths and preserve all raw result flags.

`summarize_*.py` accept `--data` and `--output`; they generate Windows Chinese plots/CSVs using Microsoft YaHei. `plot_input_motion.py` accepts the same arguments. `plot_catch_outcomes.py` takes `--results` and `--output`.

`render_archived_demo.py --run <actual_episode_root> --output <new_folder>` renders stored Genesis states and checks replay positions to 1e-5m; it never advances physics. The event-panel script displays only packets with actual availability time no later than the control time.

Other presentation/orchestration scripts preserve the executed task directory layout, environment paths and receipts. Large inputs remain at the server locations in the result handoff. The actor snapshot and locked configuration are recorded in each dataset protocol; this directory does not change the estimator, policy weights or frozen simulator core.
