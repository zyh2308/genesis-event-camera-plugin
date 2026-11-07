"""Label actual recordings with each panel's terminal time and observed outcome."""
from pathlib import Path
import argparse
import json
import hashlib
import base64

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--receipt', type=Path, default=root/'remote_queue_control/058_three_condition_event_demos.response.json')
parser.add_argument('--output', type=Path, default=Path('C:/Users/zhaoyuhan/Desktop/成熟接取任务_20261107/05_三条件事件接取_成功与失败'))
parser.add_argument('--title', default='单帧初始化＋事件跟踪｜三个条件的开发回合')
args = parser.parse_args()
reply = json.loads(args.receipt.read_text(encoding='utf-8'))
assert reply['ok'], reply
data = json.loads(reply['output'])
raw = base64.b64decode(data['base64'])
assert hashlib.sha256(raw).hexdigest() == data['sha256']
source = root/'evidence/demo'/(args.output.name+'.raw.mp4')
source.write_bytes(raw)
dest = args.output
font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 16)
names = ['正常照明 · 1.7 m/s', '高速 · 2.2 m/s', '高速逆光 · 2.2 m/s']
reader = imageio.get_reader(source)
frames = []
for index, frame in enumerate(reader):
    time_s = index / 25.
    canvas = Image.new('RGB', (960, 340), 'white')
    canvas.paste(Image.fromarray(frame), (0, 76))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), args.title, font=font, fill='#19374d')
    for j, (title, result) in enumerate(zip(names, data['panels'])):
        color = '#167d86' if result['success'] else '#9a6970'
        draw.text((j*320+10, 46), title + ('：成功' if result['success'] else '：失败'),
                  font=small, fill=color)
        own_time = min(time_s, result['sim_time'])
        text = f'{own_time:.2f} s'
        if time_s > result['sim_time']:
            text += '  已结束，画面定格'
        draw.rectangle((j*320+3, 79, j*320+305, 104), fill='white')
        draw.text((j*320+10, 81), text, font=small, fill=color)
    draw.text((12, 319), '保留原接取权重和评分；已知开发来球，不代表总体成功率。失败后不虚构后续动作。',
              font=small, fill='#425666')
    frames.append(canvas)
reader.close()
with imageio.get_writer(dest.with_suffix('.mp4'), fps=25, codec='libx264',
                        quality=8, macro_block_size=None) as writer:
    for frame in frames:
        writer.append_data(np.asarray(frame))
palette = Image.new('RGB', (320, 120*6))
for k, index in enumerate(np.linspace(0, len(frames)-1, 6).astype(int)):
    palette.paste(frames[index].resize((320, 120)), (0, k*120))
palette = palette.quantize(colors=96)
indexed = [frame.quantize(palette=palette) for frame in frames]
indexed[0].save(dest.with_suffix('.gif'), save_all=True, append_images=indexed[1:],
                duration=40, loop=0, disposal=2)
dest.with_suffix('.json').write_text(json.dumps({k:v for k,v in data.items() if k!='base64'},
                                               ensure_ascii=False, indent=2), encoding='utf-8')
print(dest)
