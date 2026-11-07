"""Combine the same recorded episode's physical scene and available event input."""
from pathlib import Path
import argparse
import json
import hashlib
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=Path, default=root/'evidence/demo/event_success_whole_and_hand.mp4')
parser.add_argument('--events', type=Path, default=root/'evidence/demo/event_control_input_raw.mp4')
parser.add_argument('--records', type=Path, default=root/'evidence/hybrid_v8_runtime_v9_trials.json')
parser.add_argument('--case', default='hybrid_normal_v9')
parser.add_argument('--initialization', default='RGB')
parser.add_argument('--camera-label',default='')
parser.add_argument('--method-label',default='')
parser.add_argument('--output', type=Path, default=Path('C:/Users/zhaoyuhan/Desktop/成熟接取任务_20261107/07_完整事件接取_全景近景事件三联'))
args = parser.parse_args()
scene_path = args.scene
event_path = args.events
scene = [f for f in imageio.get_reader(scene_path)]
events = [f for f in imageio.get_reader(event_path)]
records = json.loads(args.records.read_text(encoding='utf-8'))
trace = records[args.case+'/episode_000/trace.json']
result = records[args.case+'/episodes.json'][0]
assert result['success'] and not result['terminated'] and result['hand_contact_steps'] > 0
assert len(scene) == len(events) == len(trace)
speed = float(np.linalg.norm(result['initial']['velocity'][:2]))
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 27)
small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 21)
frames = []
for i, scene_frame in enumerate(scene):
    # Scene replay shows each step's end. Input frame i+1 is the next control
    # observation at that same physical time; the terminal hold panel is repeated.
    event_index = min(i+1, len(events)-1)
    state = trace[event_index]['event_state'] or {}
    phase = {'single_RGB_initialized_static_target_prediction': '单帧初始化／静止预测',
             'single_RGBD_initialized_static_target_prediction': '单帧 RGB-D 初始化／静止预测',
             'known_radius_event_contour_and_causal_motion_fit': '事件跟踪／含短时预测',
             'event_acquisition_then_proprioceptive_grasp_propagation': '手部状态保持／事件采样已停止'}
    canvas = Image.new('RGB', (1440, 480), 'white')
    canvas.paste(Image.fromarray(scene_frame), (0, 78))
    canvas.paste(Image.fromarray(events[event_index]).resize((480, 360)), (960, 78))
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 8), f'Genesis 接取任务：单帧 {args.initialization} 初始化＋事件跟踪', font=font, fill='#19374d')
    draw.text((1220, 12), f'{trace[i]["t"]:.2f} s  原速 1×', font=small, fill='#19374d')
    for x, label in [(15, '底盘与机械臂'+('｜'+args.camera_label if args.camera_label else '')), (495, '手部接取近景'), (975, '事件输入与目标位置估计'+('｜'+args.method_label if args.method_label else ''))]:
        draw.text((x, 46), label, font=small, fill='#19374d')
    draw.line((480, 78, 480, 438), fill='white', width=2)
    draw.line((960, 78, 960, 438), fill='white', width=2)
    draw.text((15, 449), f'水平来球 {speed:.1f} m/s｜原接取权重｜接触 {result["first_touch"]:.2f} s，保持至 {result["sim_time"]:.2f} s', font=small, fill='#19374d')
    draw.text((975, 449), phase.get(state.get('source'), '无可用观测'), font=small, fill='#167d86')
    frames.append(canvas)
dest = args.output
dest.parent.mkdir(parents=True, exist_ok=True)
with imageio.get_writer(dest.with_suffix('.mp4'), fps=25, codec='libx264',
                        quality=8, macro_block_size=None) as writer:
    for frame in frames:
        writer.append_data(np.asarray(frame))
slow = []
for i, frame in enumerate(frames):
    if result['first_touch']-.24 <= trace[i]['t'] <= result['first_touch']+.40:
        frame = frame.copy()
        draw = ImageDraw.Draw(frame)
        draw.rectangle((1175, 4, 1439, 41), fill='white')
        draw.text((1200, 10), '接触慢放 0.25×', font=small, fill='#a35316')
        slow.append(frame)
sequence = [f.resize((1080, 360), Image.Resampling.LANCZOS) for f in frames+slow]
palette = Image.new('RGB', (360, 120*6))
for k, index in enumerate(np.linspace(0, len(sequence)-1, 6).astype(int)):
    palette.paste(sequence[index].resize((360, 120)), (0, 120*k))
palette = palette.quantize(colors=112)
indexed = [f.quantize(palette=palette) for f in sequence]
indexed[0].save(dest.with_suffix('.gif'), save_all=True, append_images=indexed[1:],
                loop=0, duration=[40]*len(frames)+[160]*len(slow), disposal=2)
dest.with_suffix('.json').write_text(json.dumps(dict(
    episode=args.case+'/episode_000', initialization=args.initialization,
    camera_label=args.camera_label,method_label=args.method_label,
    source_scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
    source_event_panel_sha256=hashlib.sha256(event_path.read_bytes()).hexdigest(),
    scene_times='trace[i].t', event_control_times='trace[i+1].t_before; final hold panel repeats last input',
    frame_rate=25, actual_physics_replay=True, new_physics_run=False,
    target_GT_used_for_overlay=False, GIF_labeled_extra_slow_segment=True,
), ensure_ascii=False, indent=2), encoding='utf-8')
frames[min(28, len(frames)-1)].resize((1080, 360)).save(dest.with_suffix('.preview.png'))
print(dest)
