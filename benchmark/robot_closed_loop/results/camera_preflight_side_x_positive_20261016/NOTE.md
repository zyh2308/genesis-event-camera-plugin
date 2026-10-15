Accepted camera candidate: position `[1.15, 0, 0.22]`, lookat `[0.65, 0, 0.02]`, resolution 320x240, vertical FOV 30 degrees. Server code commit `062c3811684a854110e01cfb500222083f0f6deb`; exact used config is preserved as `config_used.json` (SHA256 `124d87188e9a383a054fb7e6e6eaf106c6d331ad0abfaf2a8a959364eafdc549`).

Corrected target-only reference measurement at intercept center: 1060 target pixels, 1060 visible with Panda at legal home pose, 0% occlusion, projected short side 30 px. Across nine static path samples with Panda held at home, projected short side ranged 30–55 px. Launch and midpoint samples had 0% occlusion; the three intercept-y samples ranged from 0% to 6.65% occlusion. HDR output was finite float32 `[240,320,3]`.

This is static geometry coverage with the robot held at nominal pre-grasp home. It is not a measurement of the moving Panda trajectory; assess that qualitatively in the direct-HDR/representative rollout video.
