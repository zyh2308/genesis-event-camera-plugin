"""Label full recorded episodes. No action synthesis or outcome-based trimming."""
import argparse
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont

FONT='C:/Windows/Fonts/msyh.ttc'
large=ImageFont.truetype(FONT,28);medium=ImageFont.truetype(FONT,22);small=ImageFont.truetype(FONT,18)
DARK='#203c50';GREEN='#157a6e';RED='#a64246'

def load_frames(record,media):
    run=media/'cases'/record['case_id']/record['mode']
    video=run/'episode_000/genesis.mp4'
    if video.exists():
        reader=imageio.get_reader(video);frames=[Image.fromarray(x).resize((480,360)) for x in reader];reader.close()
        return frames,str(video),False
    image=run/'migration_audit/sensor_episode_000/initial_rgb.png'
    if image.exists():return [Image.open(image).convert('RGB').resize((480,360))],str(image),True
    raise FileNotFoundError('Missing actual video/initial image: '+str(run))

def panel(canvas,draw,record,frames,initialization_failed,index,x,y):
    i=min(index,len(frames)-1);canvas.paste(frames[i],(x,y))
    result=record['result'];ended=index>=len(frames)
    # Native recordings contain the reset frame followed by one frame per trace row.
    trace=record.get('trace') or []
    physical_time=(0. if index==0 else trace[min(index-1,len(trace)-1)]['t']) if trace else min(index/25.,result['sim_time'])
    label=('初始化失败，无物理执行' if initialization_failed else ('已结束，画面定格' if ended else f'{physical_time:.2f} s'))
    draw.rectangle((x+4,y+4,x+384,y+33),fill='white');draw.text((x+12,y+6),label,font=small,fill=DARK)
    if ended or initialization_failed:
        draw.rectangle((x+4,y+326,x+475,y+356),fill='white');draw.text((x+12,y+328),'回合已结束；此后没有新的动作',font=small,fill=GREEN if result['success'] else RED)

def save_movie(frames,path,gif=False):
    with imageio.get_writer(path,fps=25,codec='libx264',quality=8,macro_block_size=None) as w:
        for frame in frames:w.append_data(np.asarray(frame))
    if gif:
        selected=frames[::2];palette=Image.new('RGB',(360,120*6))
        for j,k in enumerate(np.linspace(0,len(selected)-1,6).astype(int)):palette.paste(selected[k].resize((360,120)),(0,120*j))
        palette=palette.quantize(colors=96);indexed=[f.quantize(palette=palette) for f in selected]
        indexed[0].save(path.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--media',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.data.read_text(encoding='utf-8'));records={(r['case_id'],r['mode']):r for r in data['records']};conditions=['normal','fast','backlit'];names={'normal':'普通','fast':'高速','backlit':'逆光'};allframes=[];provenance=[]
    movie=a.output/'01_六个新来球_全部18条.mp4';writer=imageio.get_writer(movie,fps=25,codec='libx264',quality=8,macro_block_size=None)
    for block in range(1,7):
        row=[records[(f's{block:02d}_{c}','event')] for c in conditions];loaded=[load_frames(r,a.media) for r in row]
        initial=row[0]['result'].get('initial') or records[(row[0]['case_id'],'state_reference')]['result']['initial'];xyz=initial['position']
        for i in range(64):
            canvas=Image.new('RGB',(1440,496),'white');draw=ImageDraw.Draw(canvas)
            draw.text((18,7),'固定7°机位｜新来球完整验证',font=large,fill=DARK)
            draw.text((610,14),f"Seed {row[0]['seed']}  起点({xyz[0]:+.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) m",font=medium,fill=DARK)
            for j,(r,(frames,source,initial_failed)) in enumerate(zip(row,loaded)):
                result=r['result'];color=GREEN if result['success'] else RED
                title=f"{names[r['condition']]} · 水平 {r['horizontal_speed_m_s']:.1f} m/s · "+('成功' if result['success'] else '失败')
                draw.text((480*j+12,52),title,font=medium,fill=color);panel(canvas,draw,r,frames,initial_failed,i,480*j,88)
            draw.text((18,464),'首帧RGB-D后用事件和自身状态；原权重、估计器、冻结仿真器不变。全部成功与失败均保留。',font=small,fill=DARK);writer.append_data(np.asarray(canvas))
            if i%2==0:allframes.append(canvas.resize((960,330)))
            if block==2 and i==35:canvas.save(a.output/'01_六个新来球_预览.png')
        provenance.extend(dict(case=r['case_id'],mode='event',source=source,initialization_failed=failed,success=r['result']['success'],terminal_s=r['result']['sim_time']) for r,(_,source,failed) in zip(row,loaded))
    writer.close()
    palette=allframes[35].quantize(colors=96);indexed=[f.quantize(palette=palette) for f in allframes]
    indexed[0].save(movie.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=80,loop=0,disposal=2)
    del allframes,indexed
    # Select the lowest-seed event failure in each condition and show its matched reference.
    chosen=[]
    for condition in conditions:
        candidates=sorted([r for r in records.values() if r['mode']=='event' and r['condition']==condition],key=lambda x:x['seed'])
        failures=[r for r in candidates if not r['result']['success']]
        chosen.append((failures or candidates)[0])
    comparisons=[]
    pairloaded=[(load_frames(r,a.media),load_frames(records[(r['case_id'],'state_reference')],a.media)) for r in chosen]
    for i in range(64):
        canvas=Image.new('RGB',(1440,944),'white');draw=ImageDraw.Draw(canvas);draw.text((18,8),'同来球对照｜事件状态 / 提供真实位置的原策略参考',font=large,fill=DARK)
        for col,(r,loads) in enumerate(zip(chosen,pairloaded)):
            draw.text((col*480+12,54),f"{names[r['condition']]}  seed {r['seed']}  {r['horizontal_speed_m_s']:.1f} m/s",font=medium,fill=DARK)
            for row,mode in enumerate(['event','state_reference']):
                rec=records[(r['case_id'],mode)];frames,src,failed=loads[row];y=130+row*400;color=GREEN if rec['result']['success'] else RED
                text=('事件状态' if row==0 else '真值含噪参考')+'：'+('成功' if rec['result']['success'] else '失败')
                draw.text((col*480+12,y-34),text,font=medium,fill=color);panel(canvas,draw,rec,frames,failed,i,col*480,y)
        draw.text((18,920),'参考使用同一原权重，仅替换目标状态；不是最优控制上界。按每种条件最低seed失败选择，全部结果见主录像。',font=small,fill=DARK);comparisons.append(canvas)
    save_movie(comparisons,a.output/'02_事件与真值参考_配对.mp4',gif=True);comparisons[35].save(a.output/'02_事件与真值参考_预览.png')
    receipt=dict(scope='labels over actual full recordings; early termination is explicitly frozen; no physics or action generated',all_cases=provenance,paired_selected_cases=[r['case_id'] for r in chosen],selection='all 18 in seed order; paired overview takes lowest-seed event failure per condition, otherwise lowest-seed case',fps=25)
    (a.output/'DEMO_PROVENANCE.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(dict(all_event_cases=len(provenance),output=str(a.output)),ensure_ascii=False))

if __name__=='__main__':main()
