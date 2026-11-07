"""Render archived Genesis states, with a close view; never advances physics."""
import argparse,json,os,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--episode',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--preview-only',action='store_true');args=p.parse_args()
    import numpy as np,genesis as gs,imageio.v2 as imageio
    protocol=json.loads((args.run/'protocol.json').read_text());folder=args.run/f'episode_{args.episode:03d}';rows=json.loads((folder/'trace.json').read_text())
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    gs.init(backend=gs.cpu,precision='64',logging_level='warning')
    vis=dict(segmentation_level='link')
    if protocol['lighting']=='backlit':vis.update(ambient_light=(.01,.01,.01),lights=[dict(type='point',pos=(0.,3.8,1.8),color=(1.,1.,1.),intensity=100.)])
    scene=gs.Scene(sim_options=gs.options.SimOptions(dt=.002),vis_options=gs.options.VisOptions(**vis),show_viewer=False)
    robot=scene.add_entity(gs.morphs.MJCF(file=str(args.run/'migration_audit/genesis_task.xml'),default_armature=0.,convexify=True))
    if protocol['background_wall']:scene.add_entity(gs.morphs.Box(pos=(0.,4.,1.6),size=(4.,.02,3.2),fixed=True,collision=False),surface=gs.surfaces.Default(color=(.95,.95,.95,1.)))
    whole=scene.add_camera(res=(640,480),pos=(2.6,2.4,1.7),lookat=(0.,.6,.85),fov=48,GUI=False)
    close=scene.add_camera(res=(640,480),pos=(1.,1.,1.5),lookat=(0.,.8,1.),fov=50,GUI=False)
    scene.build();hand=robot.get_link('link6');ball=robot.get_link('object')
    def array(x):return x.detach().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)
    def render(row):
        robot.set_qpos(np.asarray(row['qpos']),zero_velocity=True)
        positions=array(robot.get_links_pos(relative=False));hp=positions[hand.idx_local];bp=positions[ball.idx_local]
        assert np.linalg.norm(hp-np.asarray(row['hand_position']))<1e-5
        assert np.linalg.norm(bp-np.asarray(row['object_position']))<1e-5
        close.set_pose(pos=tuple(hp+np.array([.55,.55,.25])),lookat=tuple(hp+np.array([0.,.03,0.])))
        a=array(whole.render(rgb=True,force_render=True)[0]);b=array(close.render(rgb=True,force_render=True)[0])
        return a,b
    checks=[]
    for wanted in [.8,1.35,1.45,2.4]:
        row=min(rows,key=lambda q:abs(q['t']-wanted));a,b=render(row);im=np.concatenate((a,b),axis=1);imageio.imwrite(args.output/f'preview_t{row["t"]:.3f}.jpg',im);checks.append(float(row['t']))
    if not args.preview_only:
        with imageio.get_writer(args.output/'whole_and_hand.mp4',fps=25,codec='libx264',quality=7,macro_block_size=None) as writer:
            for row in rows:
                a,b=render(row);writer.append_data(np.concatenate((a,b),axis=1))
    result=dict(source_run=str(args.run),episode=args.episode,physics='archived actual Genesis trajectory; renderer-only replay, no new physics rollout',
                GT_used_for_camera=False,camera_tracks='robot hand proprioception only',maximum_allowed_replay_position_error_m=1e-5,frames=len(rows),checks=checks,
                source_event_control=protocol['event_control'],time_start_s=rows[0]['t'],time_end_s=rows[-1]['t'],wall_s=time.monotonic()-start)
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
