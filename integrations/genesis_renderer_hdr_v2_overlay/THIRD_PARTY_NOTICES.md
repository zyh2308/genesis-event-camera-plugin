# Third-party notices for the HDR renderer overlay

The files under this directory are modified copies of renderer and camera
files from [Genesis World](https://github.com/Genesis-Embodied-AI/genesis-world).
Genesis World is distributed under the Apache License 2.0. The corresponding
license text is preserved as `LICENSE-APACHE-2.0.txt` in this directory.

The overlay is intentionally separated from the event-plugin implementation:
the plugin code is MIT-licensed as described in the repository root, while the
overlay must retain the upstream Genesis licensing terms. The overlay is not a
complete Genesis distribution and should be applied only to a matching Genesis
source tree/version.

The event emulator also contains code adapted from V2E; its attribution and
license information are recorded in the repository-level `NOTICE.md`.
