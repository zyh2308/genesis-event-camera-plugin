# Corrected camera/HDR preflight

Server run on code commit `9c83cbd040388e50e49489e6d83b1d8431b863cd`, config SHA256 `39c7aff627b2caa72a98fea2e8d4067218267d6f0b23507ecd59f0a7008bb70a`, Genesis 1.2.2 + Torch 2.8.0+cpu. It passed the corrected preflight: target visible with the robot, bbox `[50,97,93,144]`, projected short side 43 px, 859 target pixels, reported occlusion 0.0 against the neutral-qpos comparison, and finite HDR `[240,320,3]` with max 1.1211. The RGB image was visually checked. This is a camera candidate check, not a frozen scene config.
