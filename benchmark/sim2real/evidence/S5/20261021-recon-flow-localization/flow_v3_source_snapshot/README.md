# Flow v3 clean execution snapshot

- Source commit: `7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`.
- Clean source root on g1: `/home/zhaoyuhan/Eventbased_WAM/repositories/gec-sim2real-exec-7c55e3e5/`.
- Tar archive on the WSL execution host: `/tmp/gec-sim2real-exec-7c55e3e5.tar`, 31,498,240 bytes, SHA-256 `de704c2bfaaa26374e833dc406a5b10616d19cb337c3a9fda2ee4b27a29b4498`.
- `SOURCE_FILE_HASHES.json` contains 1,324 tracked-file SHA-256 entries and binds the clean tree to commit `7c55e3e5c2b253a13ebae5e2ba99e6f158809bc9`; its SHA-256 is `a6c8b6f716ff6558114a78a5e8183c5766e2b71ee73fe877dcac32db3f5553bb`.
- The remote generator script SHA-256 is `858bf659115b3f04e0e6441ed08fbe995f5da27ea9cc0d068f2831e1162194b1`; the audit script SHA-256 is `a9de9481ba9992cccb789592ebc87999052cf1cc75ea6665c19e36e958906706`.
- Pilot config: `configs/stage5_flow_scene_coverage_pilot_v3.json`, SHA-256 `a64ca5000cbc40e8059450d4c6819b3206ffb0fca6715f17547733fdc4e4febe`.

The archive itself is retained on the execution host; the repository stores the source file manifest, its checksums, and run-specific records so the research session can verify the deployed code without accessing a mutable checkout.
