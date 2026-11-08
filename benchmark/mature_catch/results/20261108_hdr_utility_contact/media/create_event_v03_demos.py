"""Two local presentations of archived event-controlled Genesis trajectories."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'event_demo_v03'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
W, H, FPS = 1320, 740, 25
VIDEO_Y, XL, XR = 185, 16, 664
BG, INK, MUTED = '#f6f7f9', '#142234', '#526071'
BLUE, RED, GREEN = '#1765b1', '#a83938', '#15744c'


def font(size, bold=False):
    name = 'msyhbd.ttc' if bold else 'msyh.ttc'
    return ImageFont.truetype('C:/Windows/Fonts/' + name, size)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(name):
    path = SOURCE / name / 'whole_and_hand.mp4'
    manifest_path = path.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['source_event_control'] is True
    cap = cv2.VideoCapture(str(path))
    assert (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (1280, 480)
    assert cap.get(cv2.CAP_PROP_FPS) == FPS
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr[:, 640:], cv2.COLOR_BGR2RGB)))
    cap.release()
    assert len(frames) == manifest['frames']
    # Full control intervals are 40 ms; the final frame may be an early
    # physics-step termination at e.g.1.356 or1.322 rather than next40 ms.
    times = [manifest['time_start_s'] + i/FPS for i in range(len(frames)-1)] + [manifest['time_end_s']]
    assert all(a < b for a, b in zip(times, times[1:]))
    assert 0 < times[-1] - times[-2] <= 1/FPS + 1e-8
    return dict(frames=frames, times=np.asarray(times), end_s=manifest['time_end_s'],
                receipt=dict(path=str(path), sha256=sha(path), manifest_path=str(manifest_path),
                             manifest_sha256=sha(manifest_path), manifest=manifest,
                             frame_times_source='40 ms full control intervals; exact terminal time from source manifest'))


sources = {key: load_source(key) for key in ('ours_before', 'ours_flightfit', 'v2e_flightfit')}
plans = [
    dict(stem='event_v03_ours_shared_estimator', left='ours_before', right='ours_flightfit',
         title='事件闭环｜共同估计改进，同一来球',
         subtitle='Ours 全组六条：2/6 → 4/6｜核心与原权重不变',
         note='只改共同状态估计器；此处展示 Ours 的同来球前后对照',
         left_label='原估计：接触后提前终止', right_label='共同运动估计：完成抓持',
         left_success=False, right_success=True,
         footer='真实事件控制 · 仅改共同状态估计器 · g3 Genesis 真实轨迹渲染重放',
         aggregate=dict(ours_before='2/6', ours_after='4/6')),
    dict(stem='event_v03_representative_ours_v2e', left='ours_flightfit', right='v2e_flightfit',
         title='代表回合｜同一来球，事件闭环',
         subtitle='全组：Ours 4/6，V2E 5/6｜本回合不能代表总体',
         note='两方法共用状态估计器、场景、原策略权重和成功标准',
         left_label='Ours：完成抓持', right_label='V2E：接触后提前终止',
         left_success=True, right_success=False,
         footer='代表回合 v03 · 真实事件控制 · g3 Genesis 真实轨迹渲染重放',
         aggregate=dict(ours='4/6', v2e='5/6')),
]


def probe(path):
    wsl_path = '/mnt/c/' + str(path.resolve()).replace('\\', '/')[3:]
    cmd = ['wsl.exe', '-d', 'Ubuntu-22.04', '--exec', '/usr/bin/ffprobe', '-v', 'error',
           '-count_frames', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration',
           '-of', 'json', wsl_path]
    return json.loads(subprocess.run(cmd, check=True, capture_output=True).stdout.decode('utf-8'))


for plan in plans:
    left, right = sources[plan['left']], sources[plan['right']]
    mp4, gif, poster = [HERE / (plan['stem'] + suffix) for suffix in ('.mp4', '.gif', '_poster.png')]
    command = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo',
               '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
               '-an', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(mp4)]
    p = subprocess.Popen(command, stdin=subprocess.PIPE)
    times = np.arange(1, 64) / FPS
    for i, t in enumerate(times):
        im = Image.new('RGB', (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((18, 13), plan['title'], fill=INK, font=font(35, True))
        d.text((20, 65), plan['subtitle'], fill=BLUE, font=font(27, True))
        d.text((20, 108), plan['note'], fill=MUTED, font=font(21))
        d.text((1125, 111), '0.5× 慢放', fill=BLUE, font=font(21, True))
        d.line((16, 145, 1304, 145), fill='#cdd5df', width=2)
        for x, item, key in ((XL, left, 'left'), (XR, right, 'right')):
            color = GREEN if plan[key+'_success'] else RED
            d.rectangle((x, 158, x+5, 180), fill=color)
            d.text((x+16, 152), plan[key+'_label'], fill=color, font=font(25, True))
            source_i = int(np.searchsorted(item['times'], t+1e-8, side='right') - 1)
            assert source_i >= 0
            source_i = min(source_i, len(item['frames'])-1)
            im.paste(item['frames'][source_i], (x, VIDEO_Y))
            ended = t > item['end_s'] + 1e-8
            if ended:
                d.rounded_rectangle((x+10, VIDEO_Y+437, x+397, VIDEO_Y+474), radius=5, fill='#fff1d5')
                d.text((x+20, VIDEO_Y+442), '原回合已结束｜保持末帧', fill='#825300', font=font(22, True))
                d.text((x+9, 674), f'提前终止于 {item["end_s"]:.3f} s', fill=MUTED, font=font(21))
            else:
                d.text((x+9, 674), f'仿真时间 {t:.2f} s', fill=MUTED, font=font(21))
        d.text((20, 712), plan['footer'], fill=MUTED, font=font(17))
        if i == 43:
            im.save(poster)
        frame = im.tobytes()
        p.stdin.write(frame)
        p.stdin.write(frame)
    p.stdin.close()
    assert p.wait() == 0
    gif_command = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error', '-i', str(mp4), '-filter_complex',
                   'fps=12.5,scale=1100:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
                   '-loop', '0', str(gif)]
    subprocess.run(gif_command, check=True)
    probes = {f.name: probe(f) for f in (mp4, gif)}
    assert int(probes[mp4.name]['streams'][0]['nb_read_frames']) == 126
    assert abs(float(probes[mp4.name]['format']['duration']) - 5.04) < .01
    assert int(probes[gif.name]['streams'][0]['nb_read_frames']) == 63
    with Image.open(gif) as g:
        assert g.info['loop'] == 0
        gif_frames = g.n_frames
        gif_duration_ms = 0
        for j in range(gif_frames):
            g.seek(j)
            gif_duration_ms += g.info.get('duration', 0)
    manifest = dict(
        title=plan['title'], case='v03_normal25', sources=dict(left=left['receipt'], right=right['receipt']),
        aggregate_results=plan['aggregate'],
        aggregate_scope='All six paired development cases, not an independent formal test or selected-success subset',
        aggregate_source='Root verified actual six-case experiment results; demo source manifests identify representative v03 trajectories',
        event_control=True, simulation='Archived actual g3 Genesis physics, local video composition only',
        source_video_preserved=True,
        transforms=dict(crop_xywh=[640, 0, 640, 480], playback_speed=.5, source_clock_start_s=.04,
                        clock='Reset-relative control-state timestamps; short final step uses exact source terminal time',
                        output_clock='40 ms source-state samples; each displayed twice at25fps',
                        sampling='Last available source frame, never a future source frame', interpolation='none',
                        terminated_side='Hold last frame; explicit ended banner and true terminal timestamp',
                        image_recoloring=False),
        verification=dict(ffprobe=probes, gif_loop=0, gif_frames=gif_frames, gif_duration_ms=gif_duration_ms),
        outputs=[dict(path=str(f), sha256=sha(f), size_bytes=f.stat().st_size) for f in (mp4, gif, poster)],
        commands=dict(mp4=command, gif=gif_command), script_sha256=sha(Path(__file__)),
    )
    (HERE / (plan['stem'] + '_manifest.json')).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(stem=plan['stem'], mp4_duration_s=5.04, gif_frames=gif_frames, gif_duration_ms=gif_duration_ms), ensure_ascii=False), flush=True)
