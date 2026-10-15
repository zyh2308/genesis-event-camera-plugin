Rejected camera candidate: position `[0.65, -0.25, 0.75]`, lookat `[0.65, 0, 0.02]`, code commit `a0db01337cbf9266ede5518cdfe034b2e0a995f1`, config SHA256 `eb20e525fa63ddc89887be244086c406aad9ae17e98dafb59d301dfeb333d05f`.

The intercept-pose rendering is valid: the target-only reference has 771 pixels, while only 13 target pixels remain visible with Panda at the legal pre-grasp home pose (98.31% rendered occlusion; projected short side 2 px). This camera pose is unusable and was rejected.

The nine `nominal_path_view_samples_robot_held_home` entries are invalid. The robot-scene target was repositioned with `set_qpos()` and rendered without advancing the scene, so those masks reused the prior target render. The target-only reference changed as expected, exposing the stale render. Do not use those path-sample values. Benchmark preflight code now advances one physics step before sampling; reruns use the corrected implementation.

This camera-only failure has no effect on the separate GT-oracle physical success batches.
