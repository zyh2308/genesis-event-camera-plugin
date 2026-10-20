# Incomplete replay attempt; superseded by 20261021b

This directory preserves the partial episode 004 artifacts from a failed diagnostic attempt. It is **not an accepted result** and has no complete summary or output manifest. The run stopped when the scorer asked for a query at `0.5905 s`, just beyond the final trace timestamp represented in binary floating point (`0.5904999999...`). The scorer was corrected in commit `a5388db102da884bf3e05da0f19f55d1d67dbcc3` to use the complete recorded trace for query-time ground truth. The corrected, successful two-episode replay is [`20261021b`](../r1_5_surface_raster_cached_replay_20261021b/README.md).

Partial files are retained only for provenance. Do not use their partial event stream or frame metrics for model, event, or localization conclusions.
