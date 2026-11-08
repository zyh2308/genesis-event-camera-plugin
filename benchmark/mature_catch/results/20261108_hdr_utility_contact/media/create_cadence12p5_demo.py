"""Final single12.5Hz u01 replay comparison; existing three demos stay intact."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
sys.dont_write_bytecode = True

import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
import create_window10_three_demos as media

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent.parent
sys.path.insert(0, str(LOCAL/'policy_review'))
from plot_cadence25_vs12p5 import validate_pair

STEM = 'cadence12p5_u01_side_by_side'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE.parent/'event_demo_cadence12p5')
    p.add_argument('--output', type=Path, default=HERE)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    paths = SimpleNamespace(
        hz25=HERE.parent/'contact_windows10_unseen25/CONTACT_PAIRED_COMPARISON.json',
        hz12p5=HERE.parent/'contact_windows10_cadence12p5/CONTACT_PAIRED_COMPARISON.json',
        bank=LOCAL/'FIT_UNSEEN_BANK.json',
        development=HERE.parent/'contact_paired25_windows10/CONTACT_PAIRED_COMPARISON.json')
    try:
        old, old_i, new, new_i, cases, seeds, launches, group_sources, fields = validate_pair(paths)
        case = 'u01_normal25'
        rows = {method: new_i[method][case] for method in ('ours', 'v2e')}
        if rows['ours']['verified_catch'] is not True or rows['v2e']['verified_catch'] is not False:
            raise ValueError('The declared u01 representative pair does not match actual outcomes')
        # Reuse the unchanged shared local video loader/clock validation only.
        media.SOURCE = args.source
        left, right = media.load_video('ours', rows['ours']), media.load_video('v2e', rows['v2e'])
        if abs(left['times'][0]-right['times'][0]) > 1e-8:
            raise ValueError('Source videos do not share the reset-relative first timestamp')
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        p.exit(2, f'NOT_READY: {error}\nNo12.5Hz demo or placeholder has been produced.\n')
    scores = {method: new['scores'][method]['verified_catches'] for method in ('ours', 'v2e')}
    if args.check_only:
        print(json.dumps(dict(status='READY', full_six_case_scores=scores, actual_sources_valid=True,
                              artifacts_generated=False), ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    mp4, gif, poster, manifest_path, sums_path = [args.output/(STEM+s) for s in
        ('.mp4', '.gif', '_poster.png', '_manifest.json', '_SHA256SUMS.txt')]
    if not args.overwrite and any(f.exists() for f in (mp4, gif, poster, manifest_path, sums_path)):
        p.exit(2, 'This final demo already exists; --overwrite required to regenerate it.\n')
    source_end = max(left['end_s'], right['end_s'])
    clock = np.arange(1, int(math.ceil(source_end*25-1e-8))+1)/25
    if abs(clock[-1]-source_end) > 1e-7:
        clock = np.r_[clock[clock < source_end], source_end]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1320x740', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '18', '-preset',
        'medium', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)]
    encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    subtitle = f'六条来球全组：Ours {scores["ours"]}/6，V2E {scores["v2e"]}/6｜代表回合 u01'
    poster_index = min(len(clock)-1, int(np.searchsorted(clock, min(1.76, source_end))))
    for i, t in enumerate(clock):
        im = Image.new('RGB', (1320, 740), media.BG)
        d = ImageDraw.Draw(im)
        d.text((18, 13), 'Genesis事件闭环｜12.5 Hz稀疏输入', fill=media.INK, font=media.font(35, True))
        d.text((20, 65), subtitle, fill=media.BLUE, font=media.font(25, True))
        d.text((20, 108), '输入12.5 Hz · 控制25 Hz · 共同10 ms观察窗', fill=media.MUTED, font=media.font(21))
        d.text((1125, 111), '0.5× 慢放', fill=media.BLUE, font=media.font(21, True))
        d.line((16, 145, 1304, 145), fill='#cdd5df', width=2)
        for x, source, method, label in ((16, left, 'ours', 'Ours'), (664, right, 'v2e', 'V2E')):
            row = rows[method]
            color = media.GREEN if row['verified_catch'] else media.RED
            d.rectangle((x, 158, x+5, 180), fill=color)
            d.text((x+16, 152), label+'：'+media.outcome_label(row), fill=color, font=media.font(25, True))
            k = int(np.searchsorted(source['times'], t+1e-8, side='right')-1)
            assert k >= 0
            im.paste(source['frames'][min(k, len(source['frames'])-1)], (x, 185))
            if t > source['end_s']+1e-8:
                d.rounded_rectangle((x+10, 622, x+397, 659), radius=5, fill='#fff1d5')
                d.text((x+20, 627), '原回合已结束｜保持末帧', fill='#825300', font=media.font(22, True))
                text = ('提前终止于 ' if row['terminated'] else '回合结束于 ')+f'{source["end_s"]:.3f} s'
            else:
                text = f'仿真时间 {t:.2f} s'
            d.text((x+9, 674), text, fill=media.MUTED, font=media.font(21))
        d.text((20, 712), '原权重与冻结核心不变 · Genesis归档控制轨迹渲染重放 · 同一来球', fill=media.MUTED, font=media.font(17))
        if i == poster_index:
            im.save(poster)
        rgb = im.tobytes()
        encoder.stdin.write(rgb)
        encoder.stdin.write(rgb)
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError('Final MP4 encode failed')
    gif_cmd = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-i', str(mp4), '-filter_complex',
        'fps=12.5,scale=1100:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
        '-loop', '0', str(gif)]
    subprocess.run(gif_cmd, check=True)
    probes = {f.name: media.probe(f) for f in (mp4, gif)}
    assert int(probes[mp4.name]['streams'][0]['nb_read_frames']) == len(clock)*2
    assert abs(float(probes[mp4.name]['format']['duration'])-len(clock)*2/25) < .02
    assert int(probes[gif.name]['streams'][0]['nb_read_frames']) > 1
    with Image.open(gif) as g:
        assert g.info['loop'] == 0
        frames, duration_ms = g.n_frames, 0
        for j in range(frames):
            g.seek(j)
            duration_ms += g.info.get('duration', 0)
    helpers = [HERE/'create_window10_three_demos.py', LOCAL/'policy_review/plot_cadence25_vs12p5.py',
               LOCAL/'policy_review/plot_actual_observation_window_comparison.py', LOCAL/'policy_review/plot_unused_six_four_inputs.py']
    receipt = dict(title='Genesis事件闭环｜12.5 Hz稀疏输入', subtitle=subtitle,
        case=case, seed=seeds[case], full_group_scores=scores, full_group_sources=group_sources,
        full_group_scope='Same six previously evaluated seed blocks at12.5Hz; not six additional independent cases',
        source_movies=dict(ours=left['receipt'], v2e=right['receipt']), representative_outcomes=rows,
        actual_source_image_hz=12.5, output_control_hz=25, event_observation_window_ms=10,
        original_full_source_videos_preserved=True, existing_three_demo_outputs_untouched=True,
        transforms=dict(crop_xywh=[640, 0, 640, 480], playback_speed=.5, frame_interpolation='none',
                        synchronized_on='actual reset-relative control-state times, including short terminal step',
                        ended_side='last frame frozen with explicit ended banner and exact terminal time',
                        physics_control_or_image_content_changed=False),
        verification=dict(ffprobe=probes, gif_frames=frames, gif_duration_ms=duration_ms, gif_loop=0),
        commands=dict(mp4=cmd, gif=gif_cmd), script_sha256=media.sha(Path(__file__)),
        helper_sha256={f.name: media.sha(f) for f in helpers}, outputs={f.name: media.sha(f) for f in (mp4, gif, poster)})
    manifest_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    sums_path.write_text(''.join(f'{media.sha(f)}  {f.name}\n' for f in (mp4, gif, poster, manifest_path)), encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETED_ACTUAL12P5_RESULT', scores=scores,
        mp4_duration_s=len(clock)*2/25, gif_frames=frames,
        outputs=[str(f) for f in (mp4, gif, poster, manifest_path, sums_path)]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
