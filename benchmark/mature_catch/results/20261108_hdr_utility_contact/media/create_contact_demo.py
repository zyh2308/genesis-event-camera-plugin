"""Compose existing Genesis replay videos; no simulation or remote access."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'contact_demo'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FONT = Path('C:/Windows/Fonts/msyh.ttc')
BOLD = Path('C:/Windows/Fonts/msyhbd.ttc')
W, H = 1320, 708
FPS = 25
SLOW = 2
Y_VIDEO = 162
X_LEFT, X_RIGHT = 16, 664
BG = '#f6f7f9'
INK = '#142234'
MUTED = '#526071'
BLUE = '#1765b1'
RED = '#a83938'
GREEN = '#15744c'


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(name):
    path = SOURCE / name / 'whole_and_hand.mp4'
    cap = cv2.VideoCapture(str(path))
    meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    assert meta['width'] == 1280 and meta['height'] == 480
    assert abs(meta['fps'] - FPS) < 1e-6
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr[:, 640:1280], cv2.COLOR_BGR2RGB)))
    cap.release()
    assert len(frames) == meta['frames']
    manifest_path = path.parent / 'manifest.json'
    source_manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    return frames, dict(path=str(path), sha256=sha(path), metadata=meta,
                        source_manifest_path=str(manifest_path),
                        source_manifest_sha256=sha(manifest_path),
                        source_manifest=source_manifest)


left, left_meta = load_source('baseline')
right, right_meta = load_source('combined')
assert len(left) == 35 and len(right) == 63
mp4 = HERE / 'contact_calibration_h01_side_by_side.mp4'
gif = HERE / 'contact_calibration_h01_side_by_side.gif'
poster = HERE / 'contact_calibration_h01_poster.png'

encode_command = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error',
                  '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                  '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
                  '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                  '-movflags', '+faststart', str(mp4)]
encoder = subprocess.Popen(encode_command, stdin=subprocess.PIPE)
for i in range(len(right)):
    canvas = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((18, 14), '共同场景校准｜同一来球，原权重不变', fill=INK, font=font(36, True))
    d.text((20, 67), 'GT 参考输入 · 非事件方法比较', fill=MUTED, font=font(24))
    d.text((1127, 69), '0.5× 慢放', fill=BLUE, font=font(22, True))
    d.line((16, 109, 1304, 109), fill='#cdd5df', width=2)
    d.rectangle((X_LEFT, 124, X_LEFT + 5, 151), fill=RED)
    d.text((X_LEFT+16, 120), '原参数：触球后未留住', fill=RED, font=font(26, True))
    d.rectangle((X_RIGHT, 124, X_RIGHT + 5, 151), fill=GREEN)
    d.text((X_RIGHT+16, 120), '共同校准：完成抓持', fill=GREEN, font=font(26, True))
    canvas.paste(left[min(i, len(left)-1)], (X_LEFT, Y_VIDEO))
    canvas.paste(right[i], (X_RIGHT, Y_VIDEO))
    d = ImageDraw.Draw(canvas)
    if i >= len(left):
        d.rounded_rectangle((X_LEFT+10, Y_VIDEO+437, X_LEFT+397, Y_VIDEO+474), radius=5, fill='#fff1d5')
        d.text((X_LEFT+20, Y_VIDEO+442), '原回合已结束｜保持末帧', fill='#825300', font=font(22, True))
    else:
        d.text((X_LEFT+9, 651), f'仿真时间 {0.04*(i+1):.2f} s', fill=MUTED, font=font(21))
    d.text((X_RIGHT+9, 651), f'仿真时间 {0.04*(i+1):.2f} s', fill=MUTED, font=font(21))
    if i >= len(left):
        d.text((X_LEFT+9, 651), '终止于 1.40 s', fill=MUTED, font=font(21))
    d.text((20, 683), 'g3 · Genesis 真实物理轨迹渲染重放 · h01 / seed 2026110821', fill=MUTED, font=font(16))
    if i == 43:
        canvas.save(poster)
    frame_bytes = canvas.tobytes()
    for _ in range(SLOW):
        encoder.stdin.write(frame_bytes)
encoder.stdin.close()
if encoder.wait() != 0:
    raise RuntimeError('FFmpeg MP4 encoder failed')

gif_command = [FFMPEG, '-y', '-hide_banner', '-loglevel', 'error', '-i', str(mp4),
               '-filter_complex', 'fps=12.5,scale=1100:-2:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a',
               '-loop', '0', str(gif)]
subprocess.run(gif_command, check=True)


def wsl_path(path):
    return '/mnt/c/' + str(path.resolve()).replace('\\', '/')[3:]


probes = {}
for path in (mp4, gif):
    cmd = ['wsl.exe', '-d', 'Ubuntu-22.04', '--exec', '/usr/bin/ffprobe', '-v', 'error',
           '-count_frames', '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,nb_read_frames,duration:format=duration',
           '-of', 'json', wsl_path(path)]
    result = subprocess.run(cmd, capture_output=True, check=True)
    probes[path.name] = json.loads(result.stdout.decode('utf-8'))
    (HERE / (path.stem + '_' + path.suffix[1:] + '_ffprobe.json')).write_text(
        json.dumps(probes[path.name], ensure_ascii=False, indent=2), encoding='utf-8')
assert int(probes[mp4.name]['streams'][0]['nb_read_frames']) == 126
assert abs(float(probes[mp4.name]['format']['duration']) - 5.04) < .01
assert int(probes[gif.name]['streams'][0]['nb_read_frames']) > 1
with Image.open(gif) as im:
    assert im.info.get('loop') == 0
    gif_frames = im.n_frames
    gif_duration_ms = 0
    for i in range(gif_frames):
        im.seek(i)
        gif_duration_ms += im.info.get('duration', 0)

manifest = dict(
    title='共同场景校准｜同一来球，原权重不变',
    interpretation='GT reference controller; shared contact-scene calibration; not an event-method comparison',
    sources=dict(baseline=left_meta, combined=right_meta),
    transforms=dict(crop_xywh=[640, 0, 640, 480],
                    synchronized_by='same reset-relative 40 ms control-state timestamps, starting .04 s',
                    playback_speed=.5, source_frames_per_output=0.5,
                    interpolation='none', baseline_last_frame='held after true terminal time 1.40 s with visible ended label',
                    combined_terminal_time_s=2.52, image_recoloring=False,
                    annotations='Chinese titles and disclosure; no ball or robot edits'),
    outputs=[dict(path=str(p), sha256=sha(p), size_bytes=p.stat().st_size) for p in (mp4, gif, poster)],
    verification=dict(ffprobe=probes, gif_loop=0, gif_frames=gif_frames, gif_duration_ms=gif_duration_ms),
    commands=dict(mp4=encode_command, gif=gif_command),
    script_sha256=sha(Path(__file__)),
)
(HERE / 'contact_calibration_media_manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(dict(mp4=str(mp4), gif=str(gif), poster=str(poster),
                     mp4_duration_s=5.04, gif_frames=gif_frames, gif_duration_ms=gif_duration_ms), ensure_ascii=False, indent=2))
