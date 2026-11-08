"""Compose three declared representative/counterexample Genesis replays locally.

Requires completed actual group scores before annotating a group; no placeholders.
Never opens SSH or advances physics. Original full source videos remain intact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent
SOURCE = DELIVERY/'event_demo_window10'
W, H, FPS, VIDEO_Y = 1320, 740, 25, 185
BG, INK, MUTED = '#f6f7f9', '#142234', '#526071'
BLUE, RED, GREEN = '#1765b1', '#a83938', '#15744c'
PLANS = {
    'dev_v01': dict(case='v01_normal25', stage='contact_paired25_windows10', scope='开发全组',
        title='Genesis事件闭环｜开发来球 v01', left='dev_v01_ours', right='dev_v01_v2e',
        winner='ours', stem='window10_dev_v01_side_by_side'),
    'unseen_u01': dict(case='u01_normal25', stage='contact_windows10_unseen25', scope='新来球全组',
        title='Genesis事件闭环｜新来球 u01', left='unseen_u01_ours', right='unseen_u01_v2e',
        winner='ours', stem='window10_unseen_u01_side_by_side'),
    'unseen_u02': dict(case='u02_normal25', stage='contact_windows10_unseen25', scope='新来球全组',
        title='Genesis事件闭环｜新来球反例 u02', left='unseen_u02_ours', right='unseen_u02_v2e',
        winner='v2e', stem='window10_unseen_u02_counterexample'),
}


def read(path):
    if not path.is_file():
        raise ValueError(f'Actual completed artifact is not available: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype('C:/Windows/Fonts/'+('msyhbd.ttc' if bold else 'msyh.ttc'), size)


def completed_table(plan):
    path = DELIVERY/plan['stage']/'CONTACT_PAIRED_COMPARISON.json'
    data = read(path)
    methods = ('ours', 'v2e') if plan['stage'] == 'contact_paired25_windows10' else ('ours', 'v2e', 'init_only', 'state_reference')
    if (data.get('status') != 'PASS' or data.get('all_planned_successes_and_failures_retained') is not True or
        len(data.get('rows', [])) != 6*len(methods)):
        raise ValueError('The entire actual six-seed group must be complete before annotating its demo')
    if len(methods) == 4 and (data.get('independent_confirmation') is not True or
                            data.get('all_reference_outcomes_retained') is not True):
        raise ValueError('New-bank group must include all24 audited actor outcomes and all reference outcomes')
    expected = None
    indexed = {m: {} for m in methods}
    for row in data['rows']:
        method, case = row['method'], row['case_id']
        if method not in indexed or case in indexed[method] or type(row.get('seed')) is not int:
            raise ValueError('Unexpected/duplicate actual episode or invalid seed')
        for field in ('verified_catch', 'raw_success', 'terminated', 'final_contact'):
            if type(row.get(field)) is not bool:
                raise ValueError('Actual boolean physical outcome fields required')
        touches = row['hand_contact_steps']
        if type(touches) is not int or touches < 0:
            raise ValueError('Actual nonnegative contact count required')
        if row['verified_catch'] != (row['raw_success'] and not row['terminated'] and touches > 0 and row['final_contact']):
            raise ValueError('Conflicting physical catch result')
        indexed[method][case] = row
    for method, by_case in indexed.items():
        mapping = {case: row['seed'] for case, row in by_case.items()}
        if len(mapping) != 6 or len(set(mapping.values())) != 6:
            raise ValueError('Every input group requires six distinct actual seeds')
        if expected is None:
            expected = mapping
        if mapping != expected:
            raise ValueError('The completed groups are not case/seed matched')
        scores = dict(n=6, verified_catches=sum(r['verified_catch'] for r in by_case.values()),
                      contact_episodes=sum(r['hand_contact_steps'] > 0 for r in by_case.values()))
        if any(type(data['scores'][method].get(k)) is not int or data['scores'][method][k] != v for k, v in scores.items()):
            raise ValueError('Full group scores disagree with actual episodes')
    fixed = data['fixed_parameters']
    if (fixed.get('endpoint_hz') != 25 or fixed.get('observation_window_ms') != 10 or
        fixed.get('causal_flight_fit_enabled') is not True or fixed.get('camera_pitch_degrees') != 7):
        raise ValueError('Expected locked7degree/25Hz/shared10ms/shared-flight-fit configuration')
    rows = {m: indexed[m][plan['case']] for m in ('ours', 'v2e')}
    if (rows[plan['winner']]['verified_catch'] is not True or
        rows['v2e' if plan['winner'] == 'ours' else 'ours']['verified_catch'] is not False):
        raise ValueError('Actual representative/counterexample outcome differs from the declared plan')
    return data, rows, dict(path=str(path.resolve()), sha256=sha(path))


def load_video(label, row):
    path = SOURCE/label/'whole_and_hand.mp4'
    manifest_path = SOURCE/label/'replay_manifest.json'
    if not manifest_path.is_file():
        manifest_path = SOURCE/label/'manifest.json'
    manifest = read(manifest_path)
    if not path.is_file():
        raise ValueError('Actual source video missing: '+str(path))
    if manifest.get('source_event_control') is not True:
        raise ValueError('This demo requires an event-controlled archived trajectory')
    if manifest['source_run'].rstrip('/') != row['output'].rstrip('/'):
        raise ValueError('Replay source does not match the actual scored episode')
    cap = cv2.VideoCapture(str(path))
    metadata = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS), frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if (metadata['width'], metadata['height'], metadata['fps']) != (1280, 480, 25.):
        cap.release()
        raise ValueError('Expected original1280x480 whole/hand video at25fps')
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr[:, 640:], cv2.COLOR_BGR2RGB)))
    cap.release()
    if len(frames) != manifest['frames'] or len(frames) != metadata['frames'] or len(frames) < 2:
        raise ValueError('Decoded actual frame count differs from source receipt')
    if 'frame_times_s' in manifest:
        times = np.asarray(manifest['frame_times_s'], dtype=float)
    else:
        times = np.r_[manifest['time_start_s']+np.arange(len(frames)-1)/FPS, manifest['time_end_s']]
    if len(times) != len(frames) or not np.all(np.diff(times) > 0) or np.max(np.diff(times)) > 1/FPS+1e-7:
        raise ValueError('Cannot establish the actual regular-control plus short-terminal-step source clock')
    return dict(frames=frames, times=times, end_s=float(times[-1]),
        receipt=dict(path=str(path.resolve()), sha256=sha(path), metadata=metadata,
                     manifest_path=str(manifest_path.resolve()), manifest_sha256=sha(manifest_path), manifest=manifest,
                     frame_clock_source='explicit frame_times_s' if 'frame_times_s' in manifest else
                     '40ms full control steps; exact first/final timestamps from actual replay manifest'))


def preflight(key):
    plan = PLANS[key]
    data, rows, source_table = completed_table(plan)
    left, right = load_video(plan['left'], rows['ours']), load_video(plan['right'], rows['v2e'])
    if abs(left['times'][0]-right['times'][0]) > 1e-8:
        raise ValueError('The two source resets do not share the same first control timestamp')
    return plan, data, rows, source_table, left, right


def outcome_label(row):
    if row['verified_catch']:
        return '完成抓持'
    if row['terminated']:
        return '接触后提前终止' if row['hand_contact_steps'] else '未接触提前终止'
    return '接触后未达标' if row['hand_contact_steps'] else '未接触'


def probe(path):
    wsl_path = '/mnt/c/'+str(path.resolve()).replace('\\', '/')[3:]
    cmd = ['wsl.exe', '-d', 'Ubuntu-22.04', '--exec', '/usr/bin/ffprobe', '-v', 'error', '-count_frames',
           '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames:format=duration',
           '-of', 'json', wsl_path]
    return json.loads(subprocess.run(cmd, capture_output=True, check=True).stdout.decode('utf-8'))


def compose(args, item):
    plan, data, rows, source_table, left, right = item
    output = args.output
    mp4, gif, poster, manifest_path = [output/(plan['stem']+s) for s in ('.mp4', '.gif', '_poster.png', '_manifest.json')]
    if not args.overwrite and any(p.exists() for p in (mp4, gif, poster, manifest_path)):
        raise ValueError('Demo output already exists; use --overwrite only for these named files')
    scores = {m: data['scores'][m]['verified_catches'] for m in ('ours', 'v2e')}
    case_short = plan['case'].split('_')[0]
    subtitle = f'{plan["scope"]}：Ours {scores["ours"]}/6，V2E {scores["v2e"]}/6｜代表回合 {case_short}'
    duration_source = max(left['end_s'], right['end_s'])
    clock = np.arange(1, int(math.ceil(duration_source*FPS-1e-8))+1)/FPS
    if abs(clock[-1]-duration_source) > 1e-7:
        clock = np.r_[clock[clock < duration_source], duration_source]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18',
           '-preset', 'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    poster_index = min(len(clock)-1, int(np.searchsorted(clock, min(1.76, duration_source))))
    for i, t in enumerate(clock):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((18, 13), plan['title'], fill=INK, font=font(35, True))
        d.text((20, 65), subtitle, fill=BLUE, font=font(25, True))
        d.text((20, 108), '共同10 ms观察窗 · 25 Hz输入/控制 · 冻结核心与原权重不变', fill=MUTED, font=font(21))
        d.text((1125, 111), '0.5× 慢放', fill=BLUE, font=font(21, True))
        d.line((16, 145, 1304, 145), fill='#cdd5df', width=2)
        for x, source, method, label in ((16, left, 'ours', 'Ours'), (664, right, 'v2e', 'V2E')):
            row = rows[method]
            color = GREEN if row['verified_catch'] else RED
            d.rectangle((x, 158, x+5, 180), fill=color)
            d.text((x+16, 152), label+'：'+outcome_label(row), fill=color, font=font(25, True))
            k = int(np.searchsorted(source['times'], t+1e-8, side='right')-1)
            if k < 0:
                raise ValueError('A future source frame would be required')
            im.paste(source['frames'][min(k, len(source['frames'])-1)], (x, VIDEO_Y))
            if t > source['end_s']+1e-8:
                d.rounded_rectangle((x+10, VIDEO_Y+437, x+397, VIDEO_Y+474), radius=5, fill='#fff1d5')
                d.text((x+20, VIDEO_Y+442), '原回合已结束｜保持末帧', fill='#825300', font=font(22, True))
                text = ('提前终止于 ' if row['terminated'] else '回合结束于 ')+f'{source["end_s"]:.3f} s'
            else:
                text = f'仿真时间 {t:.2f} s'
            d.text((x+9, 674), text, fill=MUTED, font=font(21))
        d.text((20, 712), '同一来球 · Genesis已归档控制轨迹渲染重放 · 全组分数来自完整六种子结果', fill=MUTED, font=font(17))
        if i == poster_index:
            im.save(poster)
        frame = im.tobytes()
        encoder.stdin.write(frame)
        encoder.stdin.write(frame)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('MP4 encoding failed')
    gif_cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-i', str(mp4), '-filter_complex',
        'fps=12.5,scale=1100:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
        '-loop', '0', str(gif)]
    subprocess.run(gif_cmd, check=True)
    probes = {f.name: probe(f) for f in (mp4, gif)}
    assert int(probes[mp4.name]['streams'][0]['nb_read_frames']) == len(clock)*2
    assert abs(float(probes[mp4.name]['format']['duration'])-len(clock)*2/FPS) < .02
    assert int(probes[gif.name]['streams'][0]['nb_read_frames']) > 1
    with Image.open(gif) as g:
        assert g.info['loop'] == 0
        gif_frames = g.n_frames
        gif_duration = 0
        for j in range(gif_frames):
            g.seek(j)
            gif_duration += g.info.get('duration', 0)
    receipt = dict(title=plan['title'], subtitle=subtitle, case=plan['case'], seed=rows['ours']['seed'],
        full_group_scores=scores, full_group_source=source_table,
        aggregate_scope='six matched development cases' if plan['scope']=='开发全组' else 'six matched newly registered cases; every input/reference actor result complete',
        representative_outcomes={m: rows[m] for m in ('ours', 'v2e')},
        source_movies=dict(ours=left['receipt'], v2e=right['receipt']), original_full_videos_preserved=True,
        transforms=dict(crop_xywh=[640, 0, 640, 480], playback_speed=.5,
            frame_interpolation='none; latest source state at or before common reset-relative timestamp',
            short_terminal_steps='exact source manifest terminal time; frozen thereafter with visible ended banner',
            simulation_or_control_changes=False, image_recoloring=False),
        verification=dict(ffprobe=probes, gif_frames=gif_frames, gif_duration_ms=gif_duration, gif_loop=0),
        commands=dict(mp4=cmd, gif=gif_cmd), script_sha256=sha(Path(__file__)),
        outputs={f.name: sha(f) for f in (mp4, gif, poster)})
    manifest_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    sums = output/(plan['stem']+'_SHA256SUMS.txt')
    sums.write_text(''.join(f'{sha(f)}  {f.name}\n' for f in (mp4, gif, poster, manifest_path)), encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETED_ACTUAL_GROUP', stem=plan['stem'],
        source_group_scores=scores, mp4_duration_s=len(clock)*2/FPS, gif_frames=gif_frames,
        outputs=[str(f) for f in (mp4, gif, poster, manifest_path, sums)]), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--group', choices=list(PLANS), action='append', help='Default: all3; select a complete group to make it separately')
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    keys = args.group or list(PLANS)
    if len(set(keys)) != len(keys):
        p.exit(2, 'Group requested twice.\n')
    try:
        items = [preflight(key) for key in keys]
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        p.exit(2, f'NOT_READY: {error}\nNo requested demo has been generated.\n')
    if args.check_only:
        print(json.dumps(dict(status='READY', groups=keys, completed_group_tables=True, source_videos_verified=True,
                             artifacts_generated=False), ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for item in items:
        compose(args, item)


if __name__ == '__main__':
    main()
