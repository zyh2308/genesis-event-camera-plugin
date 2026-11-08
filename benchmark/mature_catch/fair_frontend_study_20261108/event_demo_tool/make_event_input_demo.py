"""One fixed development pair: actual capture footage plus causally available raw events."""
from pathlib import Path,PurePosixPath
from bisect import bisect_right
import argparse,json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'media_development_tool'))
import make_development_demo as development
shared=development.shared;fair=shared.fair
SCHEMA='fair_fixed_dev_pair_raw_event_demo_v1'
TITLE='事件输入与实际接取｜开发示例'
WINDOW_S=.040


def prepare_plan(stage_dir,development_plan,development_receipt,output):
    stage=fair.load_stage(stage_dir,'validation');fair.validate_crossover(stage,fair.DEFAULT_FRONTENDS,4)
    original=development.read_plan(development_plan)
    base=shared.validate_receipts(original,Path(development_plan).parent,development_receipt)
    first=original['blocks'][0]
    fair.require(first['seed']==2026110841,'The event demo is fixed to the first development seed41')
    panels=[p for p in first['panels'] if p['endpoint_hz']==12.5]
    fair.require([p['method'] for p in panels]==list(fair.METHODS),'Fixed pair must contain both sources at12.5Hz')
    marker=fair.read_json(Path(stage_dir)/'FAIR_COLLECTION_COMPLETE.json');manifest_path=Path(stage_dir)/'SHA256SUMS'
    fair.require(fair.sha(manifest_path)==marker['raw_manifest_sha256'],'Completed raw manifest changed')
    manifest={}
    for line in manifest_path.read_text(encoding='utf-8-sig').splitlines():
        digest,relative=line.split(None,1);manifest[relative.lstrip('*').strip()]=digest
    protocol=fair.read_json(Path(stage_dir)/'EVALUATION_PROTOCOL.json');remote_root=PurePosixPath(protocol['root'])
    artifacts=[];existing={}
    for panel in panels:
        panel['event_artifacts']={}
        for identifier in panel['artifacts'].values():
            item=base[identifier];existing[identifier]=dict(path=str(item['path'].resolve()),remote=item['remote'],sha256=item['sha256'],kind=item['kind'])
        for kind,name in [('raw_events','raw_events.npz'),('packet_audit','packet_audit.json')]:
            remote=shared.remote_child(panel['actor_output'],'migration_audit/sensor_episode_000/'+name)
            relative=str(PurePosixPath(remote).relative_to(remote_root))
            fair.require(relative in manifest,'Exact final actor raw artifact missing from completed manifest')
            identifier=f"seed41:hz12p5:{panel['method']}:{kind}"
            artifacts.append(dict(artifact_id=identifier,method=panel['method'],kind=kind,remote=remote,
                local_relative=f"assets/{panel['method']}/{name}",expected_sha256=manifest[relative],manifest_relative=relative))
            panel['event_artifacts'][kind]=identifier
    output=Path(output);fair.require(not (output/'EVENT_DEMO_PLAN.json').exists(),'Use a fresh event-demo directory')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(schema=SCHEMA,title=TITLE,stage='validation',seed=first['seed'],physical_case_id=first['physical_case_id'],
        endpoint_hz=12.5,only_fixed_first_development_pair=True,not_blind_or_generalization=True,panels=panels,
        artifacts=artifacts,existing_source_artifacts=existing,window_s=WINDOW_S,
        event_display_rule='At each actual video control time select only latest parent with available_s<=time. Display its final40ms event-time window. Hold between packets and label original packet/window/availability.',
        intensity_rule='Shared fixed ON/OFF colors and three events per pixel saturation; no per-method normalization',
        no_GT_box_mask=True,no_new_physical_render=True,
        source_inputs=[dict(path=str(Path(p).resolve()),sha256=fair.sha(p)) for p in
            [Path(development_plan),Path(development_receipt),manifest_path,Path(stage_dir)/'FAIR_COLLECTION_COMPLETE.json']])
    path=output/'EVENT_DEMO_PLAN.json';shared.write_json(path,plan)
    shared.write_json(output/'PLAN_SHA256.json',dict(schema=SCHEMA,plan_sha256=fair.sha(path)))
    shared.write_json(output/'EXTRA_DOWNLOAD_FETCH_PLAN.json',[dict(remote=a['remote'],
        local=str(shared.local_child(output,a['local_relative'])),expected_sha256=a['expected_sha256']) for a in artifacts])
    return plan


def read_plan(path):
    plan=fair.read_json(path);seal=fair.read_json(Path(path).with_name('PLAN_SHA256.json'))
    fair.require(plan.get('schema')==SCHEMA and seal.get('plan_sha256')==fair.sha(path),'Fixed event-demo plan seal mismatch')
    fair.require(plan.get('seed')==2026110841 and plan.get('endpoint_hz')==12.5 and len(plan.get('panels',[]))==2,
        'Keep the fixed first development pair without outcome selection')
    return plan


def actual_inputs(plan,assets_root,receipt_path):
    extra=shared.validate_receipts(plan,assets_root,receipt_path)
    for artifact in plan['artifacts']:
        fair.require(extra[artifact['artifact_id']]['sha256']==artifact['expected_sha256'],
            'Downloaded raw events/packet audit disagrees with the completed experiment manifest')
    existing={}
    for identifier,item in plan['existing_source_artifacts'].items():
        path=Path(item['path']);fair.require(path.is_file() and fair.sha(path)==item['sha256'],
            'Previously downloaded actual scene recording changed or is missing')
        existing[identifier]=dict(item,path=path)
    return {**existing,**extra}


def load_raw(panel,inputs):
    import numpy as np
    refs=panel['event_artifacts'];packets=fair.read_json(inputs[refs['packet_audit']]['path'])
    fair.require(bool(packets),'The fixed actual event-control pair must contain captured parent packets')
    with np.load(inputs[refs['raw_events']]['path'],allow_pickle=False) as archive:events=archive['events']
    fair.require(events.ndim==2 and events.shape[1]==4 and np.isfinite(events).all(),'Invalid actual raw event array')
    fair.require(np.isin(events[:,3],[-1,1]).all() and ((events[:,1]>=0)&(events[:,1]<640)&(events[:,2]>=0)&(events[:,2]<480)).all(),
        'Raw event polarity/resolution contract differs from the actual sensor')
    fair.require(np.equal(events[:,1:3],np.floor(events[:,1:3])).all(),'Raw event pixel coordinates must remain integer pixels')
    offsets=[0];availability=[];times=[]
    for packet in packets:
        count=int(packet['events']);t=float(packet['time_s']);available=float(packet['available_s'])
        fair.require(count>=0 and packet['method']==panel['method'] and float(packet['endpoint_rate_hz'])==12.5,
            'Packet audit source/rate disagrees with the fixed actual actor')
        fair.require(abs(available-t-.01)<1e-8,'Actual parent availability must retain its recorded10ms causal delay')
        fair.require(not times or (t>times[-1] and available>availability[-1]),'Actual parent packet order is not increasing')
        offsets.append(offsets[-1]+count);availability.append(available);times.append(t)
        chunk=events[offsets[-2]:offsets[-1]]
        fair.require(not len(chunk) or chunk[:,0].max()<=t+1e-6,'Raw events occur after their parent endpoint')
    fair.require(offsets[-1]==len(events),'Raw packet event counts do not exactly cover the original event array')
    return dict(events=events,packets=packets,availability=availability,times=times,offsets=offsets,cache={})


def latest_packet(raw,query):
    # Tolerance is only for sub-nanosecond archive roundoff, never10ms latency.
    return bisect_right(raw['availability'],float(query)+1e-10)-1


def event_picture(raw,packet_index):
    import numpy as np
    from PIL import Image
    if packet_index in raw['cache']:return raw['cache'][packet_index]
    background=np.asarray([8.,17.,27.]);on=np.asarray([44.,207.,205.]);off=np.asarray([245.,103.,113.])
    counts=np.zeros((2,480,640),dtype=np.int32)
    if packet_index<0:
        selected=np.empty((0,4));start=end=None
    else:
        end=raw['times'][packet_index];start=end-WINDOW_S
        # Only prefixes from parents already available to the displayed query.
        prefix=raw['events'][:raw['offsets'][packet_index+1]]
        selected=prefix[(prefix[:,0]>start)&(prefix[:,0]<=end+1e-9)]
        for polarity,channel in [(1,0),(-1,1)]:
            subset=selected[selected[:,3]==polarity]
            if len(subset):np.add.at(counts[channel],(subset[:,2].astype(int),subset[:,1].astype(int)),1)
    alpha=np.minimum(counts.astype(float)/3.,1.)
    rgb=background+alpha[0,:,:,None]*(on-background)+alpha[1,:,:,None]*(off-background)
    image=Image.fromarray(np.clip(rgb,0,255).astype('uint8'))
    result=dict(image=image,event_count=len(selected),window_start_s=start,window_end_s=end,
        maximum_event_s=None if not len(selected) else float(selected[:,0].max()),
        packet_index=packet_index,parent_time_s=None if packet_index<0 else raw['times'][packet_index],
        available_s=None if packet_index<0 else raw['availability'][packet_index])
    raw['cache'][packet_index]=result;return result


def canvas(plan,scene,raw,index,font_path):
    from PIL import Image,ImageDraw,ImageOps
    image=Image.new('RGB',(1280,1280),'white');draw=ImageDraw.Draw(image)
    large,medium,small=shared.fonts(str(font_path));dark='#203c50';gray='#526374';green='#157a6e';red='#a64246'
    draw.text((18,8),TITLE,font=large,fill=dark)
    draw.text((18,48),f"固定开发 seed {plan['seed']} · 12.5 Hz · 首帧RGB-D后事件控制 · 原权重/冻结核心",font=medium,fill=dark)
    frame_audit=[]
    for col,(panel,data,event_data) in enumerate(zip(plan['panels'],scene,raw)):
        x=640*col;name='Ours' if panel['method']=='ours' else 'V2E';color=green if panel['verified_catch'] else red
        verdict='严格成功' if panel['verified_catch'] else '严格失败'
        time=data['elapsed'][min(index,len(data['elapsed'])-1)];query=data.get('absolute_start_s',0.)+time
        ended=index>=len(data['frames']);packet_index=latest_packet(event_data,query);picture=event_picture(event_data,packet_index)
        draw.text((x+14,93),f'{name} 实际接取 · {verdict}',font=medium,fill=color)
        draw.text((x+14,124),f"{panel['frontend_id']} | 模拟{time:.2f}s"+('；已结束，定格' if ended else ''),font=small,fill=gray)
        frame=data['frames'][min(index,len(data['frames'])-1)];sized=ImageOps.contain(frame,(640,480))
        image.paste(sized,(x+(640-sized.width)//2,154+(480-sized.height)//2))
        draw.text((x+14,652),f"{name} 原始事件 · 40ms窗 · {picture['event_count']:,}条",font=medium,fill=dark)
        if packet_index<0:label='尚无事件包到达；未提前显示后续事件'
        else:
            prior_index=latest_packet(event_data,data.get('absolute_start_s',0.)+data['elapsed'][min(max(index-1,0),len(data['elapsed'])-1)])
            status='；保留' if prior_index==packet_index and index>0 else '；新到达'
            label=f"包{picture['parent_time_s']:.2f}/到达{picture['available_s']:.2f}s | 窗{picture['window_start_s']:.2f}–{picture['window_end_s']:.2f}s"+status
        draw.text((x+14,683),label,font=small,fill=gray);image.paste(picture['image'],(x,714))
        frame_audit.append(dict(method=panel['method'],video_index=min(index,len(data['frames'])-1),query_s=query,
            scene_ended=ended,packet_index=packet_index,event_count=picture['event_count'],
            window_start_s=picture['window_start_s'],window_end_s=picture['window_end_s'],
            parent_time_s=picture['parent_time_s'],available_s=picture['available_s'],maximum_event_s=picture['maximum_event_s']))
    draw.line((640,86,640,1198),fill='#dae1e8',width=2)
    draw.text((18,1210),'青=ON，红=OFF；同40ms窗长与像素强度（3事件饱和）。仅显示已到达包，不读GT框/掩码。',font=small,fill=dark)
    draw.text((18,1242),'上为旁观相机实录，下为事件相机输入。包间/停止采集后保留最新包；开发单例不代表泛化。',font=small,fill=gray)
    return image,frame_audit


def render(plan_path,receipt_path,assets_root,output,font_path,check_only=False):
    plan=read_plan(plan_path);inputs=actual_inputs(plan,assets_root,receipt_path)
    scene=[shared.validate_panel(panel,inputs,decode=True) for panel in plan['panels']]
    raw=[load_raw(panel,inputs) for panel in plan['panels']]
    fair.require(all(not data['initialization_failed'] for data in scene),'This fixed pair must retain its actual full control recordings')
    checks=dict(status='PASS',actual_pair=1,seed=plan['seed'],source_frames=[len(d['frames']) for d in scene],
        raw_event_counts=[len(d['events']) for d in raw],actual_parent_packets=[len(d['packets']) for d in raw])
    if check_only:return dict(checks,videos_written=0)
    fair.require(Path(font_path).is_file(),'Actual readable Chinese font required')
    output=Path(output);fair.require(not (output/'EVENT_DEMO_PROVENANCE.json').exists(),'Use a fresh event-demo output directory')
    output.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    import numpy as np
    movie=output/'开发示例_seed41_12p5Hz_事件输入与实际接取.mp4';count=max(len(d['frames']) for d in scene)
    gifs=[];gif_indices=[];palette=None;frame_audit=[]
    with imageio.get_writer(movie,fps=25,codec='libx264',quality=8,macro_block_size=None,output_params=['-hide_banner']) as writer:
        for index in range(count):
            frame,audit=canvas(plan,scene,raw,index,font_path);writer.append_data(np.asarray(frame));frame_audit.append(dict(frame=index,panels=audit))
            if index%2==0 or index==count-1:
                small=frame.resize((960,960))
                if palette is None:
                    # Build the palette with both actual fixed ON/OFF colors, so
                    # later nonempty events retain their shared polarity colors.
                    palette=small.copy();palette.paste((44,207,205),(0,0,60,60));palette.paste((245,103,113),(60,0,120,60));palette=palette.quantize(colors=128)
                gifs.append(small.quantize(palette=palette));gif_indices.append(index)
            if index in [25,37,count-1]:frame.save(output/(f'frame_{index:03d}_preview.png'))
    durations=[(b-a)*40 for a,b in zip(gif_indices,gif_indices[1:])]+[40]
    gif=movie.with_suffix('.gif');gifs[0].save(gif,save_all=True,append_images=gifs[1:],duration=durations,loop=0,disposal=2)
    fair.require(all(p['available_s'] is None or p['available_s']<=p['query_s']+1e-10 for f in frame_audit for p in f['panels']),
        'A displayed parent packet preceded its real availability')
    shared_active_until=min(data['availability'][-1] for data in raw)
    common_window_checks=0
    for audit in frame_audit:
        a,b=audit['panels']
        if a['window_end_s'] is not None and b['window_end_s'] is not None and max(a['query_s'],b['query_s'])<=shared_active_until+1e-10:
            fair.require(abs(a['query_s']-b['query_s'])<1e-8 and abs(a['window_end_s']-b['window_end_s'])<1e-8,
                'Both active sources must show the same actual40ms event-time window')
            common_window_checks+=1
    shared.write_json(output/'EVENT_AVAILABILITY_FRAME_AUDIT.json',frame_audit)
    outputs=[movie,gif]+sorted(output.glob('*_preview.png'))+[output/'EVENT_AVAILABILITY_FRAME_AUDIT.json']
    provenance=dict(schema=SCHEMA,title=TITLE,stage='validation',not_blind_or_generalization=True,seed=plan['seed'],endpoint_hz=12.5,
        fixed_pair_without_outcome_selection=True,no_new_physical_render=True,no_GT_box_or_mask=True,
        window_s=WINDOW_S,same_absolute_window_while_both_active=True,common_active_window_frames_checked=common_window_checks,
        stopped_source='Hold its own last available parent window and explicitly label its older time',
        common_colors_and_fixed_saturation=plan['intensity_rule'],no_episode_trimming=True,
        causal_parent_availability_pass=True,output_frames=count,output_fps=25,gif_duration_ms=sum(durations),checks=checks,
        source_inputs={key:dict(value,path=str(value['path'].resolve())) for key,value in inputs.items()},
        plan_sha256=fair.sha(plan_path),download_receipt_sha256=fair.sha(receipt_path),code_sha256=fair.sha(__file__),
        shared_media_code_sha256=fair.sha(shared.__file__),shared_validation_code_sha256=fair.sha(fair.__file__),
        outputs=[dict(path=str(p.resolve()),sha256=fair.sha(p)) for p in outputs])
    shared.write_json(output/'EVENT_DEMO_PROVENANCE.json',provenance)
    return dict(checks,mp4=str(movie.resolve()),gif=str(gif.resolve()),output_frames=count)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--stage',type=Path,required=True);p.add_argument('--development-plan',type=Path,required=True)
    p.add_argument('--development-receipt',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('render');p.add_argument('--plan',type=Path,required=True);p.add_argument('--receipt',type=Path)
    p.add_argument('--assets-root',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,default=shared.DEFAULT_FONT)
    p.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='plan':
            plan=prepare_plan(args.stage,args.development_plan,args.development_receipt,args.output)
            result=dict(status='PASS',actual_pair=1,extra_files=len(plan['artifacts']),output=str(args.output.resolve()))
        else:result=render(args.plan,args.receipt or args.plan.with_name('EXTRA_DOWNLOAD_FETCH_PLAN.receipt.json'),
            args.assets_root or args.plan.parent,args.output,args.font,args.check_only)
        print(json.dumps(result,ensure_ascii=False))
    except (fair.DataError,KeyError,ValueError,TypeError,OSError) as error:
        parser.exit(2,f'NOT_READY: {error}\nOnly real fixed development pair recordings/events are accepted.\n')


if __name__=='__main__':main()
