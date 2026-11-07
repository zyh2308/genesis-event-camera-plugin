"""Label an actual archived task video; retain simulated timing and source receipt."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', required=True)
    parser.add_argument('--footer', required=True)
    parser.add_argument('--first-touch', type=float, default=1.4)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 22)
    small = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 17)
    reader = imageio.get_reader(args.input)
    fps = float(reader.get_meta_data()['fps'])
    frames = []
    for i, frame in enumerate(reader):
        picture = Image.fromarray(frame).resize((960, 360), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (960, 440), 'white')
        canvas.paste(picture, (0, 44))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), args.title, font=font, fill='#19374d')
        draw.text((12, 412), args.footer, font=small, fill='#19374d')
        phase = '跟踪来球' if (i + 1) / fps < args.first_touch else '接触、保持'
        label = f'{(i + 1) / fps:.2f} s  |  {phase}'
        draw.rectangle((717, 49, 953, 77), fill='#ffffff')
        draw.text((726, 51), label, font=small, fill='#19374d')
        frames.append(canvas)
    reader.close()
    with imageio.get_writer(args.output.with_suffix('.mp4'), fps=fps,
                            codec='libx264', quality=8, macro_block_size=None) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))
    # Repeat the original contact interval at quarter speed, explicitly labeled.
    slow = []
    for i, frame in enumerate(frames):
        if args.first_touch - .24 <= (i + 1) / fps <= args.first_touch + .40:
            frame = frame.copy()
            draw = ImageDraw.Draw(frame)
            draw.rectangle((714, 5, 959, 40), fill='white')
            draw.text((729, 9), '接触慢放 0.25×', font=small, fill='#a35316')
            slow.append(frame)
    sequence = frames + slow
    palette = Image.new('RGB', (320, 120 * 6))
    for k, index in enumerate(np.linspace(0, len(frames)-1, 6).astype(int)):
        palette.paste(frames[index].resize((320, 120)), (0, 120*k))
    palette = palette.quantize(colors=96)
    indexed = [frame.quantize(palette=palette) for frame in sequence]
    indexed[0].save(args.output.with_suffix('.gif'), save_all=True,
                    append_images=indexed[1:], loop=0,
                    duration=[round(1000/fps)]*len(frames)+[round(4000/fps)]*len(slow),
                    disposal=2)
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        source=str(args.input.resolve()), fps=fps, frames=len(frames),
        source_timing_retained_in_mp4=True,
        GIF_extra_slow_segment=True, title=args.title, footer=args.footer
    ), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
