#!/usr/bin/env python3
"""Verify that the v2 Genesis adapter consumes native HDR in one provider call."""

import json
import numpy as np


def main():
    import genesis as gs
    from genesis_event_plugin import GenesisPhysicsEventPlugin
    from genesis_event_plugin.genesis_adapter import make_frame_provider, camera_to_world_cv

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001, gravity=(0, 0, 0)),
        vis_options=gs.options.VisOptions(ambient_light=(0.2, 0.2, 0.2)),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        gs.morphs.Box(pos=(0, 0, 0.15), size=(0.3, 0.3, 0.3)),
        surface=gs.surfaces.Default(color=(0.8, 0.2, 0.1, 1.0)),
    )
    cam = scene.add_camera(res=(64, 48), pos=(0.8, 0.8, 0.6), lookat=(0, 0, 0.15), fov=50, GUI=False)
    scene.build()
    context = cam._rasterizer._context
    context.ambient_light = np.asarray((10.0, 10.0, 10.0), dtype=np.float32)
    context._scene.ambient_light = context.ambient_light
    context.jit.set_light(context._scene, context._scene.light_nodes, context.ambient_light)

    payload = make_frame_provider(cam, native_radiance=True)(scene, cam)
    radiance = np.asarray(payload["radiance"])
    result = {k: {"dtype": str(np.asarray(v).dtype), "shape": list(np.asarray(v).shape)} for k, v in payload.items()}
    result["radiance_max"] = float(radiance.max())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert radiance.dtype == np.float32
    assert radiance.shape == (48, 64, 3)
    assert np.isfinite(radiance).all() and (radiance > 1.0).any()

    # Exercise the actual v2 plugin boundary: native HDR is provided in the
    # frame mapping and the plugin stays in explicit linear mode.
    motion_provider = lambda scene_obj, cam_obj: {
        "camera_to_world": camera_to_world_cv(cam_obj),
        "object_to_world": {},
    }
    plugin = GenesisPhysicsEventPlugin(
        input_space="linear", frame_provider=make_frame_provider(cam, native_radiance=True),
        motion_state_provider=motion_provider, device="cpu",
    )
    plugin.attach(scene, cam).start_episode()
    first_events = plugin.capture()
    scene.step()
    second_events = plugin.capture()
    assert first_events.shape[1] == 4 and second_events.shape[1] == 4
    gs.destroy()


if __name__ == "__main__":
    main()
