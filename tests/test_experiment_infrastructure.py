"""Dependency-light checks for the experiment infrastructure layer."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile

import numpy as np

from eventgenesis_experiment.adapter import QualityAdapter, RobotAdapter
from eventgenesis_experiment.config import load_config
from eventgenesis_experiment.runner import ExperimentRunner


class _Scene:
    def __init__(self):
        self.cur_t = 0.0


class _Camera:
    res = (16, 9)
    fov = 23.6
    intrinsics = np.asarray([[10.0, 0.0, 8.0], [0.0, 10.0, 4.5], [0.0, 0.0, 1.0]])


def run() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "eventgenesis_default.yaml"
    config = load_config(config_path)
    # Keep the integration test independent of the optional 1280x720 measured
    # noise map while preserving the production configuration contract.
    sensor = replace(config.sensor, preset="clean", pixel_noise_map_path=None,
                     correlated_noise_alpha=0.0, correlated_noise_scale=0.0,
                     resolution_xy=(16, 9))
    config = replace(config, sensor=sensor,
                     scene=replace(config.scene, resolution_xy=(16, 9)))
    scene, camera = _Scene(), _Camera()
    frame_index = {"value": 0}

    def frame_provider(_scene, _camera):
        value = 1.0 if frame_index["value"] == 0 else 2.0
        frame_index["value"] += 1
        return {
            "radiance": np.full((9, 16), value, dtype=np.float32),
            "depth": np.ones((9, 16), dtype=np.float32),
            "seg": np.zeros((9, 16), dtype=np.int32),
        }

    def motion_provider(_scene, _camera):
        return {"camera_to_world": np.eye(4), "object_to_world": {}}

    with tempfile.TemporaryDirectory() as directory:
        adapter = QualityAdapter(frame_provider, motion_provider,
                                 state_provider=lambda *_: {"camera_pose": np.eye(4)})
        runner = ExperimentRunner(config, adapter, dataset_root=directory,
                                  config_path=config_path, run_id="test-run")
        runner.attach(scene, camera)
        runner.start_episode()
        first = runner.capture()
        scene.cur_t = 0.01
        second = runner.capture()
        runner.end_episode()
        runner.close()
        assert first.shape == (0, 4)
        assert second.ndim == 2 and second.shape[1] == 4
        root = Path(directory)
        metadata = json.loads((root / "metadata" / "metadata.json").read_text())
        rows = [json.loads(line) for line in (root / "metadata" / "state.jsonl").read_text().splitlines()]
        manifest = json.loads((root / "manifest.json").read_text())
        assert metadata["experiment_kind"] == "quality"
        assert len(rows) == 2 and rows[1]["timestamp_s"] == 0.01
        assert rows[1]["event_index"]["end"] >= rows[1]["event_index"]["start"]
        assert "events/aer_events.h5" in manifest["files_sha256"]

    robot = RobotAdapter(frame_provider, motion_provider,
                         state_provider=lambda *_: {"qpos": [0.0]})
    assert robot.kind == "robot"


if __name__ == "__main__":
    run()
