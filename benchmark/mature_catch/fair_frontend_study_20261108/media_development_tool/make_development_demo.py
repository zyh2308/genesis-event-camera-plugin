"""All16 selected-frontend development episodes: real recordings, not blind results."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'media_tool'))
import make_blind_demo as shared
fair=shared.fair

SCHEMA='fair_development_all16_actual_media_v1'
TITLE='前端对照｜开发样例（用于选型）'
RATES=(25.,12.5)
EXPECTED=dict(ours='w10_fit0_gap100',v2e='w0_fit1_gap100')


def prepare_plan(stage_dir,selection_path,output):
    stage=fair.load_stage(stage_dir,'validation')
    validation=fair.validate_crossover(stage,fair.DEFAULT_FRONTENDS,4)
    selection=fair.read_json(selection_path)
    fair.require(selection.get('status')=='LOCKED' and selection.get('validation_results_sha256')==fair.sha(Path(stage_dir)/'FAIR_STAGE_RESULTS.json'),
        'Development media must use the completed48-actor frontend selection')
    selected={m:fair.selected_frontend(selection['frontend_by_method'][m]) for m in fair.METHODS}
    fair.require(selected==EXPECTED,'Use the already locked development frontends without another selection')
    protocol_path=Path(stage_dir)/'EVALUATION_PROTOCOL.json';protocol=fair.read_json(protocol_path)
    indexed={(fair.block(r),r['endpoint_hz'],r['method'],r['frontend_id']):r for r in stage['rows']}
    blocks=[];artifacts=[]
    for index,physical in enumerate(sorted(validation['blocks'],key=lambda b:(b[1],b[0])),start=1):
        item=dict(block_id=f'dev_s{index:02d}',block_index=index,physical_case_id=physical[0],seed=physical[1],panels=[])
        for rate in RATES:
            for method in fair.METHODS:
                row=indexed[physical,rate,method,selected[method]]
                rate_id=f'hz{rate:g}'.replace('.','p');folder=f"assets/{item['block_id']}/{rate_id}/{method}"
                definitions=[('trace','episode_000/trace.json','trace.json'),('result','episode_000/result.json','result.json'),
                    ('actor_protocol','protocol.json','actor_protocol.json')]
                definitions.append(('video','episode_000/genesis.mp4','genesis.mp4') if row['initialization_valid'] else
                    ('initial_image','migration_audit/sensor_episode_000/initial_rgb.png','initial_rgb.png'))
                refs={}
                for kind,remote_relative,name in definitions:
                    identifier=f"{item['block_id']}:{rate_id}:{method}:{kind}"
                    artifacts.append(dict(artifact_id=identifier,kind=kind,remote=shared.remote_child(row['output'],remote_relative),
                        local_relative=f'{folder}/{name}',required=True))
                    refs[kind]=identifier
                item['panels'].append(dict(method=method,frontend_id=row['frontend_id'],endpoint_hz=rate,
                    verified_catch=row['verified_catch'],initialization_valid=row['initialization_valid'],sim_time_s=row['sim_time_s'],
                    hand_contact_steps=row['hand_contact_steps'],terminated=row['terminated'],raw_success=row['raw_success'],seed=physical[1],
                    checkpoint_sha256=protocol['checkpoint_sha256'],actor_output=row['output'],artifacts=refs))
        blocks.append(item)
    fair.require(len(blocks)==4 and sum(len(b['panels']) for b in blocks)==16,'Development media must retain all16 selected-frontend episodes')
    output=Path(output);fair.require(not (output/'DEVELOPMENT_DEMO_PLAN.json').exists(),'Use a fresh development media directory')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(schema=SCHEMA,title=TITLE,stage='validation',scope='development examples used for frontend selection; not blind or generalization results',
        actual_episodes=16,all_development_blocks_shown=True,outcome_selection=False,locked_frontends=selected,
        panel_layout='rows25/12.5Hz; columnsOurs/V2E',blocks=blocks,artifacts=artifacts,
        counts={m:dict(successes=sum(p['verified_catch'] for b in blocks for p in b['panels'] if p['method']==m),n=8) for m in fair.METHODS},
        inputs=[dict(path=str(Path(p).resolve()),sha256=fair.sha(p)) for p in stage['inputs']+[Path(selection_path),protocol_path]],
        shared_storage='g1/g3 home NFS; fetch actual files through existing g1 relay')
    path=output/'DEVELOPMENT_DEMO_PLAN.json';shared.write_json(path,plan)
    shared.write_json(output/'PLAN_SHA256.json',dict(schema=SCHEMA,plan_sha256=fair.sha(path)))
    shared.write_json(output/'DOWNLOAD_FETCH_PLAN.json',[dict(remote=a['remote'],local=str(shared.local_child(output,a['local_relative']))) for a in artifacts])
    return plan


def read_plan(path):
    plan=fair.read_json(path);seal=fair.read_json(Path(path).with_name('PLAN_SHA256.json'))
    fair.require(plan.get('schema')==SCHEMA and seal.get('plan_sha256')==fair.sha(path),'Development media plan seal mismatch')
    fair.require(plan.get('actual_episodes')==16 and len(plan.get('blocks',[]))==4 and plan.get('outcome_selection') is False,
        'Development demo must contain all16 selected-frontend episodes')
    fair.require(plan.get('locked_frontends')==EXPECTED and len({b['seed'] for b in plan['blocks']})==4,
        'Development source frontend/seed set was changed')
    for block in plan['blocks']:
        fair.require([(p['endpoint_hz'],p['method']) for p in block['panels']]==[(r,m) for r in RATES for m in fair.METHODS]
            and all(p['frontend_id']==EXPECTED[p['method']] for p in block['panels']),
            'Each development block must show the complete fixed2x2 source/rate layout')
    return plan


def canvas(block,loaded,index,counts,font_path):
    from PIL import Image,ImageDraw,ImageOps
    image=Image.new('RGB',(1280,1280),'white');draw=ImageDraw.Draw(image)
    large,medium,small=shared.fonts(str(font_path));dark='#203c50';gray='#526374';green='#157a6e';red='#a64246'
    draw.text((18,7),TITLE,font=large,fill=dark)
    draw.text((18,47),f"来球 {block['block_index']}/4 · seed {block['seed']} · 两列来源 / 两行输入帧率",font=medium,fill=dark)
    for position,(panel,data) in enumerate(zip(block['panels'],loaded)):
        row,col=divmod(position,2);x=640*col;heading=92+560*row;top=154+560*row
        color=green if panel['verified_catch'] else red;verdict='严格成功' if panel['verified_catch'] else '严格失败'
        source='Ours' if panel['method']=='ours' else 'V2E'
        draw.text((x+14,heading),f"{panel['endpoint_hz']:g} Hz · {source} · {verdict}",font=medium,fill=color)
        ended=index>=len(data['frames']);time=data['elapsed'][min(index,len(data['elapsed'])-1)]
        status='初始化失败：0步真实截图' if data['initialization_failed'] else f'模拟 {time:.2f} s'+('；已结束，定格' if ended else '')
        draw.text((x+14,heading+31),f"{panel['frontend_id']}  |  {status}",font=small,fill=gray)
        frame=data['frames'][min(index,len(data['frames'])-1)];sized=ImageOps.contain(frame,(640,480))
        image.paste(sized,(x+(640-sized.width)//2,top+(480-sized.height)//2))
    draw.line((640,86,640,1199),fill='#dae1e8',width=2)
    draw.text((18,1210),'全部16回合完整实录；短回合结束后定格，没有新增动作。固定7°、原权重、冻结仿真器。',font=small,fill=dark)
    draw.text((18,1241),f"开发选型样例：Ours {counts['ours']['successes']}/8，V2E {counts['v2e']['successes']}/8；不能作为独立盲测或正式泛化结果。",font=small,fill=gray)
    return image


def render(plan_path,receipt_path,assets_root,output,font_path,check_only=False):
    plan=read_plan(plan_path);inputs=shared.validate_receipts(plan,assets_root,receipt_path)
    fair.require(Path(font_path).is_file(),'Actual readable Chinese font required')
    checks=[]
    for block in plan['blocks']:
        loaded=[shared.validate_panel(panel,inputs,decode=True) for panel in block['panels']]
        checks.append(dict(block_id=block['block_id'],real_source_frames=[len(d['frames']) for d in loaded]))
        del loaded
    if check_only:return dict(status='PASS',actual_episodes=16,blocks=checks,videos_written=0)
    output=Path(output)
    fair.require(not (output/'DEVELOPMENT_DEMO_PROVENANCE.json').exists(),'Use a fresh rendered development demo directory')
    output.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    import numpy as np
    movie=output/'前端对照_开发选型_全部16回合.mp4';gifs=[];durations=[];palette=None;records=[];total_output=0
    with imageio.get_writer(movie,fps=25,codec='libx264',quality=8,macro_block_size=None,output_params=['-hide_banner']) as writer:
        for block in plan['blocks']:
            loaded=[shared.validate_panel(panel,inputs,decode=True) for panel in block['panels']]
            count=max(len(d['frames']) for d in loaded)
            if all(d['initialization_failed'] for d in loaded):count=max(count,50)
            gif_indices=[]
            for index in range(count):
                frame=canvas(block,loaded,index,plan['counts'],font_path);writer.append_data(np.asarray(frame));total_output+=1
                if index%2==0 or index==count-1:
                    thumbnail=frame.resize((960,960))
                    if palette is None:palette=thumbnail.quantize(colors=128)
                    gifs.append(thumbnail.quantize(palette=palette));gif_indices.append(index)
                if index==int((count-1)*.75):frame.save(output/(block['block_id']+'_preview.png'))
            durations.extend([(b-a)*40 for a,b in zip(gif_indices,gif_indices[1:])]+[40])
            records.append(dict(block_id=block['block_id'],seed=block['seed'],output_frames=count,
                panels=[dict(panel=p,recorded_steps=d['recorded_steps'],actual_source_frames=len(d['frames']),
                    final_physical_elapsed_s=d['elapsed'][-1],initialization_failed=d['initialization_failed'],
                    source={k:dict(v,path=str(v['path'].resolve())) for k,v in d['refs'].items()}) for p,d in zip(block['panels'],loaded)]))
    gif=movie.with_suffix('.gif');gifs[0].save(gif,save_all=True,append_images=gifs[1:],duration=durations,loop=0,disposal=2)
    artifacts=[movie,gif]+[output/(block['block_id']+'_preview.png') for block in plan['blocks']]
    provenance=dict(schema=SCHEMA,title=TITLE,stage='validation',actual_episodes=16,all_four_development_seeds_shown=True,
        no_outcome_selection=True,not_blind_or_formal_generalization=True,counts=plan['counts'],
        layout=plan['panel_layout'],no_episode_trimming=True,short_episode_hold_explicitly_labeled=True,
        original_frames_retained_in_mp4=True,output_fps=25,output_frames=total_output,gif_duration_ms=sum(durations),
        code_sha256=fair.sha(__file__),shared_media_code_sha256=fair.sha(shared.__file__),
        shared_validation_code_sha256=fair.sha(fair.__file__),plan_sha256=fair.sha(plan_path),
        download_receipt_sha256=fair.sha(receipt_path),font_sha256=fair.sha(font_path),records=records,
        outputs=[dict(path=str(p.resolve()),sha256=fair.sha(p)) for p in artifacts])
    shared.write_json(output/'DEVELOPMENT_DEMO_PROVENANCE.json',provenance)
    return dict(status='PASS',actual_episodes=16,mp4=str(movie.resolve()),gif=str(gif.resolve()),output_frames=total_output)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('plan');p.add_argument('--stage',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=commands.add_parser('render');p.add_argument('--plan',type=Path,required=True);p.add_argument('--assets-root',type=Path)
    p.add_argument('--receipt',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--font',type=Path,default=shared.DEFAULT_FONT)
    p.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    try:
        if args.command=='plan':
            plan=prepare_plan(args.stage,args.selection,args.output)
            result=dict(status='PASS',episodes=plan['actual_episodes'],download_files=len(plan['artifacts']),counts=plan['counts'],output=str(args.output.resolve()))
        else:
            result=render(args.plan,args.receipt or args.plan.with_name('DOWNLOAD_FETCH_PLAN.receipt.json'),
                args.assets_root or args.plan.parent,args.output,args.font,args.check_only)
        print(json.dumps(result,ensure_ascii=False))
    except (fair.DataError,KeyError,ValueError,TypeError,OSError) as error:
        parser.exit(2,f'NOT_READY: {error}\nOnly actual completed development episodes are accepted.\n')


if __name__=='__main__':main()
