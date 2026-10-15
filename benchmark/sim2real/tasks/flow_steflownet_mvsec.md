# Optical-flow route: STE-FlowNet → MVSEC

Status: synthetic feasibility loop passed on g1. No MVSEC data were downloaded or tested.

- Pinned source: `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`.
- Model remains the official event-only STE-FlowNet with its native polarity-separated Gaussian-weighted counts; grayscale supervision is synthetic only.
- The actual adapter fixed the generic five-slice edge cases, retained remainder events, guarded empty/tied times, rejected uint8 overflow, center-padded to 256×256, and bypassed the stock training-time real validation path.
- The optional correlation extension was absent. A benchmark-local pure PyTorch `F.unfold` equivalent for the official 1×1/9×9 call was used; official photometric loss and smoothness objective were kept. One matched update plus synthetic validation passed for each event source.
- Official MVSEC data/flow access was verified; the task data were not downloaded. Full results, memory, logs, and hashes are in `reports/stage1_feasibility.md` and the Stage-1 evidence bundle.
- MVSEC is test-only; no model score or sequence selection from real results in this phase.
