"""Actual Full Ours versus endpoint-only recordings; plan first, render later.

Only complete blind320+ablation128 are accepted. No physics, SSH, scene
generation, new policy, or outcome-directed trimming runs in this program.
"""
from pathlib import Path,PurePosixPath
import argparse,csv,json,sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'media_tool'))
import plot_endpoint_ablation as paired
import make_blind_demo as base

SCHEMA='full_vs_endpoint_actual_media_v1'
CATEGORIES=('full_only','endpoint_only','both_success','both_fail')
NAMES=dict(full_only='仅 Full 成功',endpoint_only='仅端点成功',both_success='双方成功',both_fail='双方失败')
TITLE='无中间重建对照｜Genesis 事件闭环'
LABELS={'full':'Full 重建','endpoint_only':'仅端点'}
FAILURES=dict(none='成功',initialization_failure='初始化失败',no_contact='未接触',
    contact_then_termination='接触后终止',contact_not_retained_at_task_end='接触未保持到结束')


def write(path,value):base.write_json(path,value)
def require(condition,message):base.fair.require(condition,message)
def sha(path):return base.fair.sha(path)


def category(full,endpoint):
    a,b=bool(full['verified_catch']),bool(endpoint['verified_catch'])
    return 'both_success' if a and b else 'full_only' if a else 'endpoint_only' if b else 'both_fail'


def choose_pairs(join):
    indices={}
    for case in join['main']['spec']['cases']:
        key=(str(case['physical_case_id']),int(case['seed']));value=int(case['block_index'])
        require(key not in indices or indices[key]==value,'Registered block index is inconsistent')
        indices[key]=value
    blocks=sorted(join['blind']['blocks'],key=lambda b:(indices[b],b[0],b[1]))
    require(len(blocks)==32 and len({indices[b] for b in blocks})==32,'Exactly32 registered blocks required')
    pairs=[];decisions=[]
    for rate in paired.collector.RATES:
        for label in CATEGORIES:
            eligible=[b for b in blocks if category(join['rows'][(b[0],b[1],rate),'ours'],
                join['rows'][(b[0],b[1],rate),'endpoint_only'])==label]
            chosen=eligible[0] if eligible else None
            decisions.append(dict(endpoint_hz=rate,category=label,eligible_blocks=len(eligible),
                selected_block_index=indices[chosen] if chosen else None))
            if chosen is not None:
                pairs.append(dict(pair_id=('hz%g_b%03d'%(rate,indices[chosen])).replace('.','p'),
                    physical_case_id=chosen[0],seed=chosen[1],endpoint_hz=rate,block_index=indices[chosen],
                    category=label,panels=[]))
    require(len(pairs)<=16,'At most16 representative pairs')
    return pairs,decisions


def manifest_entries(folder):
    entries={}
    for line in (Path(folder)/'SHA256SUMS').read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        digest,name=line.split(None,1);name=name.lstrip('*').strip()
        require(name not in entries,'Duplicate raw-manifest artifact')
        entries[name]=digest
    return entries


def prepare(main_root,endpoint_root,selection,output):
    join=paired.joined_data(main_root,endpoint_root,selection)
    # Also require the unchanged full portable source record. This validates
    # original reset/trace availability without using it to synthesize footage.
    paired.collector.main_reference(main_root,portable=True)
    pairs,decisions=choose_pairs(join);artifacts=[]
    manifests={'full':manifest_entries(main_root),'endpoint_only':manifest_entries(endpoint_root)}
    for pair in pairs:
        logical=(pair['physical_case_id'],pair['seed'],pair['endpoint_hz'])
        for variant,display_method in (('full','ours'),('endpoint_only','endpoint_only')):
            row=join['rows'][logical,display_method]
            spec=join['main']['spec'] if variant=='full' else join['endpoint_spec']
            require(row['method']=='ours' and row['frontend_id']==paired.collector.FRONTEND,'Both panels must use identical locked Ours B frontend')
            definitions=[('trace','episode_000/trace.json','trace.json'),('result','episode_000/result.json','result.json'),
                ('actor_protocol','protocol.json','actor_protocol.json')]
            definitions.append(('video','episode_000/genesis.mp4','genesis.mp4') if row['initialization_valid'] else
                ('initial_image','migration_audit/sensor_episode_000/initial_rgb.png','initial_rgb.png'))
            refs={}
            for kind,remote_relative,name in definitions:
                remote=base.remote_child(row['output'],remote_relative)
                try:raw_name=str(PurePosixPath(remote).relative_to(PurePosixPath(spec['root'])))
                except ValueError:raw_name=remote
                require(raw_name in manifests[variant],'Actual source file missing from sealed raw manifest: '+remote)
                identifier=pair['pair_id']+':'+variant+':'+kind;refs[kind]=identifier
                artifacts.append(dict(artifact_id=identifier,pair_id=pair['pair_id'],method='ours',variant=variant,kind=kind,
                    remote=remote,local_relative='assets/'+pair['pair_id']+'/'+variant+'/'+name,
                    expected_sha256=manifests[variant][raw_name],required=True))
            pair['panels'].append(dict(method='ours',variant=variant,display_label=LABELS[variant],
                frontend_id=row['frontend_id'],endpoint_hz=row['endpoint_hz'],verified_catch=row['verified_catch'],
                initialization_valid=row['initialization_valid'],sim_time_s=row['sim_time_s'],
                hand_contact_steps=row['hand_contact_steps'],terminated=row['terminated'],raw_success=row['raw_success'],
                failure_class=row['failure_class'],seed=pair['seed'],checkpoint_sha256=spec['checkpoint_sha256'],
                actor_output=row['output'],artifacts=refs))
    output=Path(output);require(not (output/'DEMO_DOWNLOAD_PLAN.json').exists(),'Fresh plan output required')
    output.mkdir(parents=True,exist_ok=True)
    plan=dict(schema=SCHEMA,title=TITLE,main_actors=320,additional_endpoint_actors=128,physical_blocks=32,
        selected_pair_count=len(pairs),pairs=pairs,decisions=decisions,artifacts=artifacts,
        selection_rule='At every rate/category select the minimum registered block_index; four exhaustive outcome categories; at most16 pairs; not a success-rate estimate',
        category_labels=NAMES,main_blind_reference=join['main']['hashes'],same_frontend=paired.collector.FRONTEND,
        media_rules=dict(actual25fps_full_recordings=True,all_original_frames_retained=True,
            early_end_final_frame_hold_explicitly_labeled=True,no_synthetic_motion=True,all_failures_remain_in_full_tables=True),
        inputs=[dict(path=str(Path(p).resolve()),sha256=sha(p)) for p in [
            Path(main_root)/'FAIR_COLLECTION_COMPLETE.json',Path(main_root)/'FAIR_STAGE_RESULTS.json',
            Path(main_root)/'PAIRED_PORTABLE.json',Path(main_root)/'EVALUATION_PROTOCOL.json',Path(main_root)/'SHA256SUMS',
            Path(endpoint_root)/'FAIR_COLLECTION_COMPLETE.json',Path(endpoint_root)/'FAIR_STAGE_RESULTS.json',
            Path(endpoint_root)/'EVALUATION_PROTOCOL.json',Path(endpoint_root)/'ENDPOINT_VARIANT_PROOFS.json',
            Path(endpoint_root)/'ENDPOINT_SOURCE_DELTA_PROOF.json',Path(endpoint_root)/'SHA256SUMS',Path(selection)]])
    path=output/'DEMO_DOWNLOAD_PLAN.json';write(path,plan);write(output/'PLAN_SHA256.json',dict(plan_sha256=sha(path)))
    all_rows=[]
    indices={(str(c['physical_case_id']),int(c['seed'])):int(c['block_index']) for c in join['main']['spec']['cases']}
    for logical in sorted(join['main']['rows'],key=lambda x:(x[2],indices[x[:2]])):
        a,b=join['rows'][logical,'ours'],join['rows'][logical,'endpoint_only']
        all_rows.append(dict(physical_case_id=logical[0],seed=logical[1],block_index=indices[logical[:2]],endpoint_hz=logical[2],
            full_verified_catch=a['verified_catch'],endpoint_verified_catch=b['verified_catch'],category=category(a,b),
            full_failure_class=a['failure_class'],endpoint_failure_class=b['failure_class'],
            full_initialization_valid=a['initialization_valid'],endpoint_initialization_valid=b['initialization_valid'],
            full_sim_time_s=a['sim_time_s'],endpoint_sim_time_s=b['sim_time_s'],full_actor_output=a['output'],endpoint_actor_output=b['output']))
    require(len(all_rows)==128,'Full paired outcome table must retain all128 cells')
    with (output/'ALL128_PAIRED_OUTCOMES.csv').open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    write(output/'DOWNLOAD_FETCH_PLAN.json',[dict(remote=a['remote'],local=str(base.local_child(output,a['local_relative'])),
        expected_sha256=a['expected_sha256']) for a in artifacts])
    return plan


def read_plan(path):
    path=Path(path);value=base.fair.read_json(path)
    require(value.get('schema')==SCHEMA and base.fair.read_json(path.with_name('PLAN_SHA256.json'))['plan_sha256']==sha(path),'Ablation media plan seal differs')
    require(value['selected_pair_count']==len(value['pairs'])<=16,'Invalid paired-demo count')
    return value


def validate_panel(panel,inputs):
    data=base.validate_panel(panel,inputs,decode=True)
    protocol=base.fair.read_json(data['refs']['actor_protocol']['path'])
    mode=protocol.get('event_reconstruction_mode')
    require(mode=='endpoint_only' if panel['variant']=='endpoint_only' else mode!='endpoint_only',
        'Actual source reconstruction mode differs from displayed panel')
    return data


def canvas(pair,loaded,index,font):
    from PIL import Image,ImageDraw,ImageOps
    image=Image.new('RGB',(1280,720),'white');d=ImageDraw.Draw(image);large,medium,small=base.fonts(str(font))
    dark='#203c50';gray='#526374';green='#157a6e';red='#a64246'
    d.text((18,8),TITLE,font=large,fill=dark)
    d.text((18,48),'Block %02d · seed %s · %g Hz · %s'%(pair['block_index'],pair['seed'],pair['endpoint_hz'],NAMES[pair['category']]),font=medium,fill=dark)
    for col,(panel,data) in enumerate(zip(pair['panels'],loaded)):
        x=640*col;color=green if panel['verified_catch'] else red
        verdict='严格成功' if panel['verified_catch'] else '严格失败'
        d.text((x+14,89),panel['display_label']+' · '+verdict,font=medium,fill=color)
        d.text((x+14,120),'同一前端 B：'+panel['frontend_id']+' | '+FAILURES.get(panel['failure_class'],panel['failure_class']),font=small,fill=gray)
        frame=data['frames'][min(index,len(data['frames'])-1)];frame=ImageOps.contain(frame,(640,480))
        image.paste(frame,(x+(640-frame.width)//2,154+(480-frame.height)//2))
        ended=index>=len(data['frames']);elapsed=data['elapsed'][min(index,len(data['elapsed'])-1)]
        label='初始化失败：0 物理步，真实初始化截图' if data['initialization_failed'] else '模拟时间 %.2f s'%elapsed
        d.rectangle((x+7,162,x+632,195),fill='white');d.text((x+15,165),label,font=small,fill=dark)
        if ended and not data['initialization_failed']:
            d.rectangle((x+7,590,x+632,625),fill='white');d.text((x+15,596),'回合已结束；之后画面定格，没有新动作',font=small,fill=color)
    d.line((640,84,640,637),fill='#dae1e8',width=2)
    d.text((18,650),'两列均为 Ours：只改变中间重建；同来球、同前端 B、原权重与冻结像素响应。',font=small,fill=dark)
    d.text((18,681),'完整 25 fps 实录；每率每类取最小 block。全部成功与失败见逐集表，代表录像不代替统计。',font=small,fill=gray)
    return image


def render(plan_path,output,font,check_only=False):
    plan=read_plan(plan_path);assets=Path(plan_path).parent
    receipt=assets/'DOWNLOAD_FETCH_PLAN.receipt.json'
    inputs=base.validate_receipts(plan,assets,receipt)
    require(Path(font).is_file(),'Readable Chinese font required')
    for a in plan['artifacts']:
        require(inputs[a['artifact_id']]['sha256']==a['expected_sha256'],'Source changed since sealed raw manifest')
    for pair in plan['pairs']:
        loaded=[validate_panel(p,inputs) for p in pair['panels']];del loaded
    if check_only:return dict(status='PASS',actual_pairs=len(plan['pairs']),videos_written=0)
    output=Path(output);require(not output.exists(),'Fresh rendered output required');output.mkdir(parents=True)
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image
    records=[]
    for pair in plan['pairs']:
        loaded=[validate_panel(p,inputs) for p in pair['panels']]
        count=max(len(x['frames']) for x in loaded)
        if all(x['initialization_failed'] for x in loaded):count=max(count,50)
        stem=pair['pair_id']+'_'+pair['category'];movie=output/(stem+'.mp4');gifs=[];indices=[]
        with imageio.get_writer(movie,fps=25,codec='libx264',quality=8,macro_block_size=None,output_params=['-hide_banner']) as writer:
            for index in range(count):
                frame=canvas(pair,loaded,index,font);writer.append_data(np.asarray(frame))
                if index%2==0 or index==count-1:gifs.append(frame.resize((960,540)));indices.append(index)
        palette=gifs[len(gifs)//2].quantize(colors=128);indexed=[x.quantize(palette=palette) for x in gifs]
        durations=[(b-a)*40 for a,b in zip(indices,indices[1:])]+[40]
        indexed[0].save(movie.with_suffix('.gif'),save_all=True,append_images=indexed[1:],duration=durations,loop=0,disposal=2)
        canvas(pair,loaded,count//2,font).save(movie.with_suffix('.png'))
        reader=imageio.get_reader(movie)
        try:require(abs(reader.get_meta_data()['fps']-25)<1e-6 and reader.count_frames()==count,'Output MP4 frame budget/fps differs')
        finally:reader.close()
        with Image.open(movie.with_suffix('.gif')) as animation:
            gif_frames=animation.n_frames
            require(gif_frames>1 or all(x['initialization_failed'] for x in loaded),'Moving episode GIF is unexpectedly static')
        records.append(dict(pair=pair,output_frames=count,output_fps=25,gif_frames=gif_frames,
            full_source_frames_retained=True,ended_episode_hold_labeled=True,
            sources=[dict(variant=p['variant'],recorded_steps=x['recorded_steps'],source_frames=len(x['frames']),
                final_simulated_elapsed_s=x['elapsed'][-1],initialization_failed=x['initialization_failed']) for p,x in zip(pair['panels'],loaded)],
            outputs=[dict(path=str(p.resolve()),sha256=sha(p)) for p in (movie,movie.with_suffix('.gif'),movie.with_suffix('.png'))]))
    write(output/'DEMO_PROVENANCE.json',dict(schema=SCHEMA,status='PASS',plan_sha256=sha(plan_path),
        transfer_receipt_sha256=sha(receipt),code_sha256=sha(__file__),helper_sha256=sha(base.__file__),font_sha256=sha(font),
        decisions=plan['decisions'],records=records,actual_pairs=len(records),
        no_synthetic_physical_frames=True,no_outcome_based_trimming=True,all_failures_retained_in_tables=True))
    with (output/'SHA256SUMS').open('x',encoding='utf-8') as f:
        for p in sorted(output.iterdir()):
            if p.is_file() and p.name!='SHA256SUMS':f.write(sha(p)+'  '+p.name+'\n')
    return dict(status='PASS',actual_pairs=len(records),output=str(output.resolve()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('plan');p.add_argument('--main-blind-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--selection',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('render');p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path,default=base.DEFAULT_FONT);p.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if args.command=='plan':
        result=prepare(args.main_blind_root,args.root,args.selection or args.main_blind_root/'FRONTEND_SELECTION.json',args.output)
        print(json.dumps(dict(status='PASS',selected_pairs=result['selected_pair_count'],artifacts=len(result['artifacts']),videos_created=0)))
    else:print(json.dumps(render(args.plan,args.output,args.font,args.check_only),ensure_ascii=False))


if __name__=='__main__':main()
