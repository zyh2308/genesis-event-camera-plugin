# R1.5 cached evidence CPU replay

This run replays the saved center and +55 mm episodes on CPU. It compares the frozen 100 Hz path, endpoint identity/exact candidates, and log-radiance with exact cached-render fills where timestamps match. Ten previously saved 1 kHz direct HDR images provide image references; candidate event streams use only actual state rows from the saved trace. Unmatched render requests are counted and filled from the nearer endpoint solely to keep the event emulator defined.

The query table includes the 0.5 ms read phases. These are learned-model diagnostics on candidate event streams; rows using unresolved render requests are not evidence that the full render-request path is accurate. The cached direct event sequence is only 9 ms long and is excluded from 25 ms model inference. No simulator core, physics, or renderer was changed or initialized.
