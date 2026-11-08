"""Plan downloads, then label actual complete blind recordings; never synthesize scenes."""
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'figures_tool'))
import make_fair_figures as fair

SCHEMA='fair_blind_actual_media_v1'
CATEGORIES=('ours_only','v2e_only','both_success','both_fail','initialization_failure')
CATEGORY_NAMES=dict(ours_only='仅 Ours 成功',v2e_only='仅 V2E 成功',both_success='双方成功',
    both_fail='双方失败',initialization_failure='含初始化失败')
TITLE='盲测代表回合｜固定7°原权重'
DEFAULT_FONT=Path('C:/Windows/Fonts/msyh.ttc')


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8',newline='\n')


def pair_categories(ours,v2e):
    a,b=ours['verified_catch'],v2e['verified_catch']
    labels=['both_success' if a and b else 'ours_only' if a else 'v2e_only' if b else 'both_fail']
    if not ours['initialization_valid'] or not v2e['initialization_valid']:labels.append('initialization_failure')
    return labels


def protocol_blocks(protocol,blind):
    mapping={}
    for case in protocol['cases']:
        physical=str(case['physical_case_id']);seed=int(case['seed']);index=int(case['block_index'])
        fair.require(1<=index<=32,'Invalid registered blind block_index')
        if (physical,seed) in mapping:fair.require(mapping[physical,seed]==index,'Inconsistent physical block index')
        mapping[physical,seed]=index
    fair.require(set(mapping)==set(blind['blocks']) and len(set(mapping.values()))==32,
        'Protocol must identify all 32 blind blocks with unique registered block indices')
    return mapping


def representative_pairs(blind,indices):
    selected={};decisions=[]
    ordered=sorted(blind['blocks'],key=lambda b:(indices[b],b[0],b[1]))
    for rate in blind['rates']:
        for category in CATEGORIES:
            eligible=[b for b in ordered if category in pair_categories(blind['indexed'][b,rate,'ours'],blind['indexed'][b,rate,'v2e'])]
            chosen=eligible[0] if eligible else None
            decisions.append(dict(endpoint_hz=rate,category=category,eligible_blocks=len(eligible),
                selected_block_index=None if chosen is None else indices[chosen],
                selected_physical_case_id=None if chosen is None else chosen[0],
                selected_seed=None if chosen is None else chosen[1]))
            if chosen is None:continue
            logical=rate,chosen
            if logical not in selected:
                pair_id=f"hz{rate:g}_b{indices[chosen]:03d}".replace('.','p')
                selected[logical]=dict(pair_id=pair_id,endpoint_hz=rate,physical_case_id=chosen[0],seed=chosen[1],
                    block_index=indices[chosen],selection_categories=[],panels=[])
            selected[logical]['selection_categories'].append(category)
    pairs=sorted(selected.values(),key=lambda p:(p['endpoint_hz'],p['block_index']))
    fair.require(len(pairs)<=20,'Representative selection exceeded20 paired recordings')
    return pairs,decisions


def remote_child(root,relative):
    path=PurePosixPath(str(root));child=PurePosixPath(relative)
    fair.require(path.is_absolute() and '..' not in path.parts and not child.is_absolute() and '..' not in child.parts,
        'Actor artifact must have a literal absolute server path and safe relative child')
    return str(path/child)


def local_child(root,relative):
    base=Path(root).resolve();child=(base/relative).resolve()
    fair.require(child!=base and base in child.parents,'Local artifact path escapes the chosen asset directory')
    return child


def prepare_plan(stage_dir,selection_path,protocol_path,output):
    stage=fair.load_stage(stage_dir,'blind')
    blind=fair.validate_blind(stage,selection_path,32,fair.BLIND_RATES)
    protocol=fair.read_json(protocol_path);indices=protocol_blocks(protocol,blind)
    fair.require(protocol.get('frontend_selection_sha256')==fair.sha(selection_path),
        'Blind protocol does not bind this exact frontend selection file')
    pairs,decisions=representative_pairs(blind,indices);artifacts=[]
    for pair in pairs:
        physical=(pair['physical_case_id'],pair['seed'])
        for method in fair.METHODS:
            row=blind['indexed'][physical,pair['endpoint_hz'],method]
            initialized=row['initialization_valid'];folder=f"assets/{pair['pair_id']}/{method}"
            definitions=[('trace','episode_000/trace.json','trace.json'),
                ('result','episode_000/result.json','result.json'),('actor_protocol','protocol.json','actor_protocol.json')]
            definitions.append(('video','episode_000/genesis.mp4','genesis.mp4') if initialized else
                ('initial_image','migration_audit/sensor_episode_000/initial_rgb.png','initial_rgb.png'))
            refs={}
            for kind,remote_relative,name in definitions:
                identifier=f"{pair['pair_id']}:{method}:{kind}";relative=f'{folder}/{name}'
                artifacts.append(dict(artifact_id=identifier,pair_id=pair['pair_id'],method=method,kind=kind,
                    remote=remote_child(row['output'],remote_relative),local_relative=relative,required=True))
                refs[kind]=identifier
            pair['panels'].append(dict(method=method,frontend_id=row['frontend_id'],endpoint_hz=row['endpoint_hz'],
                verified_catch=row['verified_catch'],initialization_valid=initialized,sim_time_s=row['sim_time_s'],
                hand_contact_steps=row['hand_contact_steps'],terminated=row['terminated'],raw_success=row['raw_success'],
                seed=pair['seed'],checkpoint_sha256=protocol['checkpoint_sha256'],
                actor_output=row['output'],artifacts=refs))
    output=Path(output);fair.require(not (output/'DEMO_DOWNLOAD_PLAN.json').exists(),'Use a fresh media plan output directory')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(schema=SCHEMA,title=TITLE,scope='actual complete Genesis recordings only; representative media is not an aggregate estimate',
        independent_blind_blocks=32,total_completed_actors=len(stage['rows']),rates=blind['rates'],
        selection_rule='At every registered rate and category take the minimum registered block_index; retain both sources; deduplicate pair keys; maximum20 pairs.',
        categories=list(CATEGORIES),decisions=decisions,selected_pair_count=len(pairs),pairs=pairs,artifacts=artifacts,
        locked_frontends=blind['selected'],inputs=[dict(name=Path(p).name,path=str(Path(p).resolve()),sha256=fair.sha(p))
            for p in stage['inputs']+[Path(selection_path),Path(protocol_path)]],
        media_rules=dict(original_episode_never_trimmed=True,shorter_episode_holds_final_frame_with_label=True,
            initialization_failure_real_image_only=True,initialization_failure_zero_physics_steps=True,
            no_synthetic_scene_images=True,show_source_frontend_rate_verdict_time=True))
    path=output/'DEMO_DOWNLOAD_PLAN.json';write_json(path,plan)
    write_json(output/'PLAN_SHA256.json',dict(schema=SCHEMA,plan_sha256=fair.sha(path)))
    export_fetch_plan(path,output,output/'DOWNLOAD_FETCH_PLAN.json')
    return plan


def read_plan(path):
    path=Path(path);plan=fair.read_json(path);seal=fair.read_json(path.with_name('PLAN_SHA256.json'))
    fair.require(plan.get('schema')==SCHEMA and seal.get('plan_sha256')==fair.sha(path),'Media plan seal mismatch')
    fair.require(plan['selected_pair_count']==len(plan['pairs'])<=20,'Invalid selected media pair count')
    return plan


def export_fetch_plan(plan_path,assets_root,output):
    plan=read_plan(plan_path)
    rows=[dict(remote=item['remote'],local=str(local_child(assets_root,item['local_relative']))) for item in plan['artifacts']]
    write_json(output,rows)
    return rows


def validate_receipts(plan,assets_root,receipt_path):
    receipts=fair.read_json(receipt_path);fair.require(isinstance(receipts,list),'Download receipt must be a list of remotely hashed files')
    indexed={}
    for row in receipts:
        logical=(str(row['remote']),str(Path(row['local']).resolve()))
        fair.require(logical not in indexed,'Duplicate remote/local artifact in transfer receipt');indexed[logical]=row
    inputs={}
    for artifact in plan['artifacts']:
        path=local_child(assets_root,artifact['local_relative']);logical=(artifact['remote'],str(path))
        fair.require(logical in indexed,f'Missing remote-verified transfer receipt for {artifact["artifact_id"]}')
        receipt=indexed[logical];fair.require(path.is_file(),f'Actual source artifact has not been downloaded: {path}')
        fair.require(path.stat().st_size==int(receipt['size']) and fair.sha(path)==receipt['sha256'],f'Downloaded source failed remote hash check: {path}')
        inputs[artifact['artifact_id']]=dict(path=path,remote=artifact['remote'],sha256=receipt['sha256'],kind=artifact['kind'])
    return inputs


def elapsed_times(trace):
    fair.require(bool(trace),'A non-initialization-failure episode must have actual control steps')
    start=float(trace[0]['t_before']);times=[0.]+[float(row['t'])-start for row in trace]
    fair.require(all(math.isfinite(t) for t in times) and all(b>a for a,b in zip(times,times[1:])),
        'Actual trace time is not finite and strictly increasing')
    intervals=[b-a for a,b in zip(times,times[1:])]
    fair.require(all(abs(dt-.04)<=1e-5 for dt in intervals),'Unexpected recording/control interval; refuse implicit respeeding')
    return times,start


def validate_panel(panel,inputs,decode=True):
    refs={kind:inputs[identifier] for kind,identifier in panel['artifacts'].items()}
    result=fair.read_json(refs['result']['path']);trace=fair.read_json(refs['trace']['path'])
    protocol=fair.read_json(refs['actor_protocol']['path'])
    fair.require(protocol.get('backend')=='Genesis' or 'genesis' in str(protocol.get('backend','')).lower(),
        'Demo must be an actual Genesis actor recording')
    fair.require(float(protocol.get('sensor_pitch_offset_degrees',float('nan')))==7.,'Actual actor did not use fixed7degree sensor pitch')
    fair.require(protocol.get('event_generator')==panel['method'] and protocol.get('event_control') is True,
        'Actual actor event source/control mode differs from the demo panel')
    fair.require(protocol.get('checkpoint_sha256')==panel['checkpoint_sha256'], 'Actual actor did not preserve the registered original weights')
    expected=fair.selected_frontend({'CATCHIT_EVENT_OBSERVATION_WINDOW_MS':protocol['event_observation_window_ms'],
        'CATCHIT_EVENT_FLIGHT_FIT':int(protocol['event_flight_fit']),
        'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS':protocol['event_confirmation_max_gap_ms']})
    fair.require(expected==panel['frontend_id'],'Actual actor frontend differs from selected demo record')
    final=bool(trace and trace[-1]['hand_contact'])
    strict=bool(result['success'] and not result['terminated'] and int(result['hand_contact_steps'])>0 and final)
    fair.require(strict==panel['verified_catch'] and len(trace)==int(result['steps']), 'Actual video actor scoring differs from the collector')
    fair.require(int(result['seed'])==panel['seed'] and int(result['hand_contact_steps'])==panel['hand_contact_steps']
        and bool(result['success'])==panel['raw_success'] and bool(result['terminated'])==panel['terminated']
        and abs(float(result['sim_time'])-panel['sim_time_s'])<1e-8, 'Actual actor episode metadata differs from the collector')
    if not panel['initialization_valid']:
        fair.require(not trace and int(result['steps'])==0 and float(result['sim_time'])==0. and not strict,
            'Initialization-failure panel must truly execute zero physics steps')
        data=dict(refs=refs,trace=trace,elapsed=[0.],recorded_steps=0,initialization_failed=True)
        if decode:
            from PIL import Image
            data['frames']=[Image.open(refs['initial_image']['path']).convert('RGB')]
        return data
    times,start=elapsed_times(trace)
    data=dict(refs=refs,trace=trace,elapsed=times,recorded_steps=len(trace),initialization_failed=False,absolute_start_s=start)
    if decode:
        import imageio.v2 as imageio
        from PIL import Image
        reader=imageio.get_reader(refs['video']['path'])
        try:
            metadata=reader.get_meta_data();frames=[Image.fromarray(frame).convert('RGB') for frame in reader]
        finally:reader.close()
        fair.require(abs(float(metadata['fps'])-25.)<1e-5,'Unexpected original video frame rate')
        fair.require(len(frames)==len(trace)+1,'Original reset+every-control-step video frame budget is incomplete')
        data.update(frames=frames,original_fps=metadata['fps'])
    return data


@lru_cache(maxsize=4)
def fonts(font_path):
    from PIL import ImageFont
    return tuple(ImageFont.truetype(font_path,size) for size in (28,22,17))


def make_canvas(pair,loaded,index,font_path):
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    canvas=Image.new('RGB',(1280,720),'white');draw=ImageDraw.Draw(canvas)
    large,medium,small=fonts(str(font_path))
    dark='#203c50';green='#157a6e';red='#a64246';gray='#526374'
    draw.text((18,8),TITLE,font=large,fill=dark)
    names=' / '.join(CATEGORY_NAMES[c] for c in pair['selection_categories'])
    draw.text((18,48),f"Block {pair['block_index']:02d} · seed {pair['seed']} · {pair['endpoint_hz']:g} Hz · {names}",font=medium,fill=dark)
    for col,(panel,data) in enumerate(zip(pair['panels'],loaded)):
        x=640*col;color=green if panel['verified_catch'] else red
        verdict='严格成功' if panel['verified_catch'] else '严格失败'
        draw.text((x+14,89),f"{'Ours' if panel['method']=='ours' else 'V2E'} · {verdict}",font=medium,fill=color)
        draw.text((x+14,120),f"来源 {panel['method']}  |  前端 {panel['frontend_id']}  |  {panel['endpoint_hz']:g} Hz",font=small,fill=gray)
        frame=data['frames'][min(index,len(data['frames'])-1)]
        resized=ImageOps.contain(frame,(640,480));canvas.paste(resized,(x+(640-resized.width)//2,154+(480-resized.height)//2))
        ended=index>=len(data['frames']);time=data['elapsed'][min(index,len(data['elapsed'])-1)]
        label='初始化失败：0 物理步，真实初始化截图' if data['initialization_failed'] else f'模拟时间 {time:.2f} s'
        draw.rectangle((x+7,162,x+632,195),fill='white');draw.text((x+15,165),label,font=small,fill=dark)
        if ended and not data['initialization_failed']:
            draw.rectangle((x+7,590,x+632,625),fill='white')
            draw.text((x+15,596),'回合已结束；之后画面定格，没有新动作',font=small,fill=color)
    draw.line((640,84,640,637),fill='#dae1e8',width=2)
    draw.text((18,650),'首帧 RGB-D 初始化，随后仅事件 + 自身状态；固定 7°机位、原权重、冻结仿真器。',font=small,fill=dark)
    draw.text((18,681),'完整实录，无结果导向裁切；按每类最小 block 选取，含双方有利和失败例。',font=small,fill=gray)
    return canvas


def render_plan(plan_path,assets_root,receipt_path,output,font_path,check_only=False):
    plan=read_plan(plan_path);inputs=validate_receipts(plan,assets_root,receipt_path)
    fair.require(Path(font_path).is_file(),'An actual Chinese font is required for readable labels')
    checked=[]
    # Decode/check every requested real source before producing any demo.
    for pair in plan['pairs']:
        loaded=[validate_panel(panel,inputs,decode=True) for panel in pair['panels']]
        checked.append(pair)
        # Never retain the whole study's decoded clips in memory at once.
        del loaded
    if check_only:
        return dict(status='PASS',actual_pairs=len(checked),actual_inputs=len(inputs),videos_written=0)
    output=Path(output);fair.require(not (output/'DEMO_PROVENANCE.json').exists(),'Use a fresh rendered demo output directory')
    output.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    import numpy as np
    provenance=[]
    for pair in checked:
        loaded=[validate_panel(panel,inputs,decode=True) for panel in pair['panels']]
        last=max(len(d['frames']) for d in loaded)-1
        # Both initialization failures have no physical trajectory. Two seconds
        # merely holds their actual screenshots; no generated motion is inserted.
        total_frames=max(last+1,50 if all(d['initialization_failed'] for d in loaded) else 1)
        stem=f"{pair['pair_id']}_"+'_'.join(pair['selection_categories']);movie=output/(stem+'.mp4')
        gifs=[];gif_indices=[]
        with imageio.get_writer(movie,fps=25,codec='libx264',quality=8,macro_block_size=None,output_params=['-hide_banner']) as writer:
            for index in range(total_frames):
                canvas=make_canvas(pair,loaded,index,font_path);writer.append_data(np.asarray(canvas))
                if index%2==0 or index==total_frames-1:
                    gifs.append(canvas.resize((960,540)));gif_indices.append(index)
        palette=gifs[len(gifs)//2].quantize(colors=128);indexed=[image.quantize(palette=palette) for image in gifs]
        durations=[(b-a)*40 for a,b in zip(gif_indices,gif_indices[1:])]+[40]
        indexed[0].save(movie.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=durations,loop=0,disposal=2)
        make_canvas(pair,loaded,min(total_frames//2,total_frames-1),font_path).save(movie.with_suffix('.png'))
        outputs=[dict(path=str(p.resolve()),sha256=fair.sha(p)) for p in [movie,movie.with_suffix('.gif'),movie.with_suffix('.png')]]
        provenance.append(dict(pair=pair,panels=[dict(method=p['method'],recorded_control_steps=d['recorded_steps'],
            initialization_failed=d['initialization_failed'],source_frames=len(d['frames']),
            final_physical_elapsed_s=d['elapsed'][-1],source={k:dict(v,path=str(v['path'].resolve())) for k,v in d['refs'].items()})
            for p,d in zip(pair['panels'],loaded)],output_fps=25,output_frames=total_frames,
            gif_fps_approx=12.5,gif_duration_ms=sum(durations),original_video_frames_retained_in_mp4=True,outputs=outputs))
    receipt=dict(schema=SCHEMA,title=TITLE,plan_sha256=fair.sha(plan_path),download_receipt_sha256=fair.sha(receipt_path),
        code_sha256=fair.sha(__file__),font_sha256=fair.sha(font_path),selected_pair_count=len(checked),
        footage_scope='actual actor Genesis recordings or actual zero-step initialization images; no re-rendered/synthetic scenes',
        selection=plan['selection_rule'],decisions=plan['decisions'],representative_media_not_success_estimate=True,
        no_outcome_based_trimming=True,early_termination_hold_explicitly_labeled=True,records=provenance)
    write_json(output/'DEMO_PROVENANCE.json',receipt)
    return dict(status='PASS',actual_pairs=len(checked),output=str(output.resolve()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--stage',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--protocol',type=Path);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('rebase-fetch');p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('render');p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets-root',type=Path)
    p.add_argument('--receipt',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,default=DEFAULT_FONT)
    p.add_argument('--check-only',action='store_true');args=parser.parse_args()
    try:
        if args.command=='plan':
            plan=prepare_plan(args.stage,args.selection,args.protocol or args.stage/'EVALUATION_PROTOCOL.json',args.output)
            result=dict(status='PASS',selected_pairs=plan['selected_pair_count'],real_artifacts=len(plan['artifacts']),videos_created=0,output=str(args.output.resolve()))
        elif args.command=='rebase-fetch':
            rows=export_fetch_plan(args.plan,args.assets_root,args.output);result=dict(status='PASS',files=len(rows),videos_created=0)
        else:
            result=render_plan(args.plan,args.assets_root or args.plan.parent,
                args.receipt or args.plan.with_name('DOWNLOAD_FETCH_PLAN.receipt.json'),args.output,args.font,args.check_only)
        print(json.dumps(result,ensure_ascii=False))
    except (fair.DataError,KeyError,TypeError,ValueError,OSError) as error:
        parser.exit(2,f'NOT_READY: {error}\nOnly completed blind results and downloaded real recordings are accepted; no placeholder demo is created.\n')


if __name__=='__main__':main()
