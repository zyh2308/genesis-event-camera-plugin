# E2VID temporal-consistency geometry preflight

Local WSL reference run on 2026-10-19 (Asia/Tokyo). This is an early, independent numerical check; it is not the server model/data preflight and does not authorize training by itself.

Commands:

```bash
python benchmark/sim2real/scripts/audit_e2vid_tc_geometry.py --device cpu --output benchmark/sim2real/evidence/S5/20261019-tc-aligntrue-preflight/geometry_cpu.json
python benchmark/sim2real/scripts/audit_e2vid_tc_geometry.py --device cuda:0 --output benchmark/sim2real/evidence/S5/20261019-tc-aligntrue-preflight/geometry_cuda.json
```

Both CPU and CUDA passed on a non-square 9x17 field. Zero flow produced exact identity and zero TC loss; the known integer translation had zero error and correct sign; subpixel bilinear sampling differed from the closed-form ramp by `4.7684e-7`; in-bounds and zero-padding boundaries matched the explicit rule; the independent sampler agreed exactly. The deliberately retained old implicit `align_corners=False` path produced `0.71875` max error on the checkerboard, so the fixture detects the reviewed bug.

Server-side E2VID outputs, gradients, loader, augmentation, recurrence, and restart remain separate mandatory gates.
