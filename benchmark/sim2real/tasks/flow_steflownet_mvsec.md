# Optical-flow route: STE-FlowNet → MVSEC

Status: Stage-2 paired multi-rollout training adapters are implemented; g1 run is pending. No MVSEC data were downloaded or tested.

- Pinned source: `ruizhao26/STE-FlowNet@eb4498c4126788c8ee8c8cbfabfbe9b5b77548f6`.
- Model remains the official event-only STE-FlowNet with its native polarity-separated Gaussian-weighted counts; grayscale supervision is synthetic only.
- The actual adapter fixed the generic five-slice edge cases, retained remainder events, guarded empty/tied times, rejected uint8 overflow, center-padded to 256×256, and bypassed the stock training-time real validation path.
- Stage 2 keeps the official STE-FlowNet photometric/smoothness objective, with four train and two separate validation physics rollouts. dt1 is one adjacent endpoint interval; dt4 spans four adjacent endpoint intervals. Labels are the same physical forward displacement for both event sources.
- The optional native correlation extension was absent on the pinned Stage-1 environment. Stage 2 selects the benchmark-local pure PyTorch F.unfold implementation of the official 1×1-kernel/9×9-patch operation and compares it numerically to an explicit 81-shift reference at the model feature size. Its measured run-time and peak memory will be recorded with training evidence.
- The real-data manifest pins the official MVSEC HDF5 event/timestamp fields, flow-ground-truth timestamp association helper, and STE-FlowNet dt1/dt4 reference entries. Real sequences and GT were not downloaded or scored; the official validity support and primary EPE mask still need research freeze before testing.
- MVSEC is test-only; no model score or sequence selection from real results in this phase.
