#!/usr/bin/env python3
"""
Plan A scaffold: Genesis 仿真器 vs 真实事件相机 — PECS CD/GD 评估
============================================================

IMPORTANT: this file is an experiment scaffold, not a valid PECS reproduction.
The scene builders currently lack commanded motion, checkerboard texture,
matched camera/lens/light calibration and synchronization. It intentionally
fails closed so placeholder output cannot be used as a research result.

Target PECS 6 序列实验：
  R_360_H  — 旋转圆盘, 360 rpm, 高光照
  R_360_L  — 旋转圆盘, 360 rpm, 低光照
  R_60_H   — 旋转圆盘, 60 rpm,  高光照
  T_1.0_H  — 平移棋盘格, 1.0 m/s, 高光照
  T_0.6_H  — 平移棋盘格, 0.6 m/s, 高光照
  T_1.0_L  — 平移棋盘格, 1.0 m/s, 低光照

用法:
  # 在服务器上执行
  cd ~/Eventbased_WAM/code/genesis_event_plugin
  
  # 方式1: 命令行
  PYTHONPATH=. python genesis_event_plugin/evaluation/plan_a_runner.py \
      --real-dir /path/to/real/events/ \
      --presets moderate noisy \
      --output results/
  
  # 方式2: Python API
  from genesis_event_plugin.evaluation.plan_a_runner import run_plan_a
  reports = run_plan_a(real_dir='/data/real/', presets=['moderate'])

依赖:
  - Genesis (genesis-world) + 事件插件
  - 真实事件数据 (.raw 或 .h5)
  - scipy, h5py, numpy
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .evaluator import EventSimEvaluator, EvaluationReport, batch_evaluate
from .event_reader import read_events


# ════════════════════════════════════════════════════════
# Plan A 场景列表
# ════════════════════════════════════════════════════════

PLAN_A_SCENES = [
    {
        'name': 'R_360_H',
        'type': 'rotate',
        'speed_rpm': 360,
        'light': 'high',
        'description': '旋转圆盘 360rpm 高光照',
        'duration_s': 5.0,
    },
    {
        'name': 'R_360_L',
        'type': 'rotate',
        'speed_rpm': 360,
        'light': 'low',
        'description': '旋转圆盘 360rpm 低光照',
        'duration_s': 5.0,
    },
    {
        'name': 'R_60_H',
        'type': 'rotate',
        'speed_rpm': 60,
        'light': 'high',
        'description': '旋转圆盘 60rpm 高光照',
        'duration_s': 5.0,
    },
    {
        'name': 'T_1.0_H',
        'type': 'translate',
        'speed_mps': 1.0,
        'light': 'high',
        'description': '平移棋盘格 1.0m/s 高光照',
        'duration_s': 3.0,
    },
    {
        'name': 'T_0.6_H',
        'type': 'translate',
        'speed_mps': 0.6,
        'light': 'high',
        'description': '平移棋盘格 0.6m/s 高光照',
        'duration_s': 3.0,
    },
    {
        'name': 'T_1.0_L',
        'type': 'translate',
        'speed_mps': 1.0,
        'light': 'low',
        'description': '平移棋盘格 1.0m/s 低光照',
        'duration_s': 3.0,
    },
]


PLAN_A_BLOCKERS = (
    "rotating/translation actors are not actuated",
    "checkerboard/disk geometry and texture are placeholders",
    "camera intrinsics, lens, illumination and spectral response are not calibrated",
    "real/sim spatial and temporal synchronization is not implemented",
)


def _require_valid_plan_a_implementation():
    details = "; ".join(PLAN_A_BLOCKERS)
    raise NotImplementedError(
        "Plan A is a non-claimable scaffold and is disabled until the matched "
        f"physical protocol is implemented: {details}."
    )


# ════════════════════════════════════════════════════════
# Genesis 场景生成
# ════════════════════════════════════════════════════════

def generate_genesis_events(
    scene_config: dict,
    preset: str = 'moderate',
    output_dir: str = '/tmp/plan_a_output',
    cam_resolution: tuple = (240, 320),
    dt: float = 0.001,  # 1ms sim step
) -> np.ndarray:
    """
    Generate events using Genesis + event plugin for a given Plan A scene.
    
    Args:
        scene_config: scene dict from PLAN_A_SCENES
        preset: DVS noise preset
        output_dir: temp output directory
        cam_resolution: (H, W)
        dt: physics step (seconds)
    
    Returns:
        events: (N, 4) [x, y, p, t] float32
    """
    _require_valid_plan_a_implementation()
    try:
        import genesis as gs
        from genesis_event_plugin import GenesisEventPlugin
    except ImportError as e:
        warnings.warn(f"Genesis not available: {e}. Returning empty events.")
        return np.empty((0, 4), dtype=np.float32)
    
    gs.init(backend=gs.cpu, logging_level='warning')
    
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2, 1, 2),
            camera_lookat=(0, 0, 0),
        ),
        show_viewer=False,
    )
    
    # Camera
    cam = scene.add_camera(
        res=cam_resolution[::-1],  # (W, H)
        pos=(0, 0, 1.5),
        lookat=(0, 0, 0),
        fov=60,
    )
    
    # Scene type
    scene_type = scene_config['type']
    
    if scene_type == 'rotate':
        _build_rotate_scene(scene, scene_config)
    elif scene_type == 'translate':
        _build_translate_scene(scene, scene_config)
    
    scene.build()
    
    # Plugin
    plugin = GenesisEventPlugin(
        output_dir=output_dir,
        preset=preset,
        interpolation_mode='analytic',
        U_fixed=5,
        output_event_frames=False,
        output_rgb_frames=False,
    )
    plugin.attach(scene, cam, [])
    
    # Run simulation
    duration = scene_config['duration_s']
    n_steps = int(duration / dt)
    
    plugin.start_episode()
    all_events = []
    
    for step in range(n_steps):
        scene.step()
        events = plugin.capture()
        if len(events) > 0:
            all_events.append(events)
    
    plugin.end_episode()
    plugin.close()
    
    if all_events:
        events = np.concatenate(all_events, axis=0)
        # Sort by time and convert to [x,y,p,t]
        events = events[events[:, 0].argsort()]
        events = events[:, [1, 2, 3, 0]]  # [t,x,y,p] → [x,y,p,t]
    else:
        events = np.empty((0, 4), dtype=np.float32)
    
    return events


def _build_rotate_scene(scene, cfg):
    """Build rotating disk scene."""
    import genesis as gs
    
    # Plane (background)
    scene.add_entity(gs.morphs.Plane())
    
    # Rotating disk with checkerboard pattern
    # For now, use a textured box as proxy
    disk = scene.add_entity(
        gs.morphs.Cylinder(
            pos=(0, 0, 0.1),
            radius=0.15,
            height=0.01,
        ),
        surface=gs.surfaces.Default(
            color=(0.8, 0.8, 0.8, 1.0),
        ),
    )
    
    # TODO: Add checkerboard texture and rotation motor
    # For full implementation, use the Genesis-RL patterns
    warnings.warn("Rotate scene is simplified. Full checkerboard + motor TBD.")


def _build_translate_scene(scene, cfg):
    """Build translating checkerboard scene."""
    import genesis as gs
    
    scene.add_entity(gs.morphs.Plane())
    
    # Checkerboard as textured box on linear rail
    board = scene.add_entity(
        gs.morphs.Box(
            pos=(0, 0, 0.1),
            size=(0.3, 0.3, 0.005),
        ),
    )
    
    warnings.warn("Translate scene is simplified. Full checkerboard + rail TBD.")


# ════════════════════════════════════════════════════════
# Main Runner
# ════════════════════════════════════════════════════════

def run_plan_a(
    real_dir: str,
    presets: List[str] = None,
    output_dir: str = 'plan_a_results',
    subsample: int = 5000,
    scenes: List[str] = None,
) -> Dict[str, List[EvaluationReport]]:
    """
    Run full Plan A evaluation.
    
    Args:
        real_dir: directory containing real event files named like 'R_360_H.raw'
        presets: DVS presets to evaluate (default: ['moderate', 'noisy'])
        output_dir: where to save results
        subsample: max events for CD/GD computation
        scenes: which scenes to run (default: all 6)
    
    Returns:
        {preset: [EvaluationReport for each scene]}
    """
    _require_valid_plan_a_implementation()
    if presets is None:
        presets = ['moderate', 'noisy']
    
    all_results = {}
    evaluator = EventSimEvaluator(subsample=subsample)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for preset in presets:
        print(f"\n{'='*70}")
        print(f"  Preset: {preset}")
        print(f"{'='*70}")
        
        preset_reports = []
        
        for scene_cfg in PLAN_A_SCENES:
            name = scene_cfg['name']
            if scenes and name not in scenes:
                continue
            
            print(f"\n  [{name}] {scene_cfg['description']}...")
            
            # 1. Generate Genesis events
            sim_events = generate_genesis_events(
                scene_cfg, preset=preset,
                output_dir=f'{output_dir}/genesis_{preset}_{name}',
            )
            print(f"    Generated {len(sim_events):,} events")
            
            # 2. Load real events (if available)
            real_path = Path(real_dir) / f"{name}.raw"
            real_events = np.empty((0, 4), dtype=np.float32)
            if real_path.exists():
                try:
                    real_events = read_events(str(real_path), max_events=subsample * 2)
                    print(f"    Loaded {len(real_events):,} real events")
                except Exception as e:
                    print(f"    ⚠️  Failed to load real events: {e}")
            else:
                print(f"    ⚠️  No real data: {real_path} not found")
            
            # 3. Evaluate
            if len(sim_events) > 0 and len(real_events) > 0:
                report = evaluator.full_evaluation(
                    sim_events, real_events,
                    preset=preset, scene=name,
                )
                preset_reports.append(report)
                print(report.summary())
            elif len(sim_events) > 0:
                # Internal-only evaluation
                report = evaluator.full_evaluation(
                    sim_events, sim_events,  # self-compare
                    preset=preset, scene=name,
                )
                report.external = None
                preset_reports.append(report)
                print(f"    Internal only (no real data)")
                print(report.summary())
        
        all_results[preset] = preset_reports
        
        # Save intermediate results
        result_file = Path(output_dir) / f"results_{preset}.json"
        with open(result_file, 'w') as f:
            json.dump(
                {preset: [r.to_dict() for r in preset_reports]},
                f, indent=2,
            )
        print(f"\n  Results saved to {result_file}")
    
    # Final summary
    _print_final_summary(all_results)
    
    return all_results


def _print_final_summary(all_results):
    """Print comparison table across presets."""
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY: Genesis vs PECS Baselines")
    print("=" * 70)
    
    print(f"\n{'Scene':<12s}", end="")
    for preset in all_results:
        print(f"  {preset:>20s}", end="")
    print(f"  {'PECS':>12s}  {'V2E':>12s}")
    print("-" * 70)
    
    for scene_cfg in PLAN_A_SCENES:
        name = scene_cfg['name']
        print(f"{name:<12s}", end="")
        
        for preset in all_results:
            reports = all_results[preset]
            matching = [r for r in reports if r.scene_name == name and r.external]
            if matching:
                cd = matching[0].external.cd
                print(f"  CD={cd:>5.1f} GD={matching[0].external.gd:>5.2f}", end="")
            else:
                print(f"  {'N/A':>20s}", end="")
        
        # PECS baselines
        from .pecs_metrics import PECS_BASELINES, V2E_BASELINES
        pecs = PECS_BASELINES.get(name, (0, 0))
        v2e = V2E_BASELINES.get(name, (0, 0))
        print(f"  GD={pecs[0]:>5.2f} CD={pecs[1]:>6.1f}", end="")
        print(f"  GD={v2e[0]:>5.2f} CD={v2e[1]:>6.1f}")
    
    print("=" * 70)


# ════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Plan A: Genesis simulator vs real events (PECS CD/GD)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full Plan A with all presets
  python plan_a_runner.py --real-dir /data/plan_a_real/

  # Single scene, single preset
  python plan_a_runner.py --real-dir /data/real/ --presets moderate --scenes R_360_H

  # Just generate and self-evaluate (no real data needed)
  python plan_a_runner.py --self-eval --presets moderate clean noisy
        """,
    )
    parser.add_argument('--real-dir', default='',
                       help='Directory with real event files')
    parser.add_argument('--presets', nargs='+', default=['moderate'],
                       help='DVS presets to evaluate')
    parser.add_argument('--scenes', nargs='+', default=None,
                       help='Which scenes to run (default: all 6)')
    parser.add_argument('--output', default='plan_a_results',
                       help='Output directory')
    parser.add_argument('--subsample', type=int, default=5000,
                       help='Max events for CD/GD')
    parser.add_argument('--self-eval', action='store_true',
                       help='Run without real data (internal metrics only)')
    
    args = parser.parse_args()
    
    if args.self_eval:
        # Internal evaluation only
        evaluator = EventSimEvaluator(subsample=args.subsample)
        for preset in args.presets:
            print(f"\n{'='*70}")
            print(f"  Self-Evaluation: preset={preset}")
            print(f"{'='*70}")
            for scene_cfg in PLAN_A_SCENES:
                if args.scenes and scene_cfg['name'] not in args.scenes:
                    continue
                print(f"\n  [{scene_cfg['name']}] Generating...")
                sim = generate_genesis_events(scene_cfg, preset=preset)
                report = evaluator.full_evaluation(
                    sim, sim, preset=preset, scene=scene_cfg['name'],
                )
                report.external = None
                print(report.summary())
    else:
        run_plan_a(
            real_dir=args.real_dir,
            presets=args.presets,
            output_dir=args.output,
            subsample=args.subsample,
            scenes=args.scenes,
        )


if __name__ == '__main__':
    main()
