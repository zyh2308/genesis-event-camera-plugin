"""Export Stage9 audited data and optional standard matplotlib plots. No remote calls."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,re,statistics
from collections import defaultdict
from pathlib import Path

METHODS=('eventgenesis','v2e','vid2e','film_esim','rife_esim')
LABELS={'eventgenesis':'EventGenesis','v2e':'V2E','vid2e':'Vid2E','film_esim':'FILM + ESIM','rife_esim':'RIFE + ESIM'}
TASKS=('reconstruction','flow','pose')
BUDGETS={'reconstruction':12000,'flow':8910,'pose':30000}
INPUT_SIZES={'training_matrix.json':129098,'synthetic_curve_points.json':720577}

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def csv_new(path,rows,fields):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def export_data(source,out):
    for name,size in INPUT_SIZES.items():
        if (source/name).stat().st_size!=size:raise RuntimeError('Incomplete or different input: '+name)
    matrix=read(source/'training_matrix.json');raw=read(source/'synthetic_curve_points.json')
    runs=matrix['runs'];keys={(r['task'],r['method'],r['seed']) for r in runs}
    assert len(runs)==45 and keys=={(t,m,s) for t in TASKS for m in METHODS for s in range(3)}
    normalized=[];counts={};source_index={}
    for r in runs:
        task=r['task'];budget=BUDGETS[task];assert r['budget']==budget
        last=r.get('latest_metric') or {}
        metric_update=last.get('optimizer_update',last.get('optimizer_updates'))
        candidates=[x for x in (r.get('latest_update'),metric_update) if isinstance(x,int)]
        update=max(candidates) if candidates else None
        complete=(r.get('completion_verified') is True and r.get('selected_hash_verified') is True and update==budget)
        state='COMPLETE_VERIFIED' if complete else 'INCOMPLETE_OBSERVED' if update is not None else 'NOT_OBSERVED'
        row=dict(task=task,method=r['method'],seed=r['seed'],budget=budget,
            reported_update=r.get('latest_update'),latest_metric_update=metric_update,
            reconciled_update=update,training_state=state,
            completion_receipt_verified=r.get('completion_verified') is True,
            selected_checkpoint_hash_verified=r.get('selected_hash_verified') is True,
            live_file_state=r.get('live_status') or '',process_state='SEPARATE_PROCESS_RECEIPT_REQUIRED',
            real_evaluation_status='NOT_EVALUATED',observed_utc=matrix['observed_utc'])
        normalized.append(row);source_index[(task,r['method'],r['seed'])]=row
    for task in TASKS:
        rows=[r for r in normalized if r['task']==task]
        counts[task]={state:sum(r['training_state']==state for r in rows) for state in ('COMPLETE_VERIFIED','INCOMPLETE_OBSERVED','NOT_OBSERVED')}
        assert counts[task]['COMPLETE_VERIFIED']==matrix['completion_counts'][task]
    csv_new(out/'training_completion_matrix.csv',normalized,list(normalized[0]))
    summaries=[dict(task=t,planned_runs=15,budget_per_run=BUDGETS[t],verified_complete=counts[t]['COMPLETE_VERIFIED'],
        incomplete_observed=counts[t]['INCOMPLETE_OBSERVED'],not_observed=counts[t]['NOT_OBSERVED'],
        real_evaluation_status='NOT_EVALUATED',observed_utc=matrix['observed_utc']) for t in TASKS]
    csv_new(out/'training_task_summary.csv',summaries,list(summaries[0]))

    # Use validation measurements only. Optimizer losses are never relabeled as validation.
    points={}
    def add(task,method,seed,update,metric,value,unit,source_name,recorded='',epoch=None):
        if not isinstance(update,int) or not isinstance(value,(int,float)) or not math.isfinite(value):
            raise RuntimeError('Invalid validation observation')
        key=(task,method,seed,update,metric)
        if key in points:
            if points[key]['value']!=value:raise RuntimeError('Conflicting validation duplicate: '+str(key))
            return
        points[key]=dict(task=task,method=method,seed=seed,optimizer_update=update,epoch=epoch,
            metric=metric,value=value,unit=unit,source=source_name,recorded_utc=recorded,
            training_state=source_index[(task,method,seed)]['training_state'],real_evaluation_status='NOT_EVALUATED')
    for x in raw:
        task=x['task'];method=x['method'];seed=x['seed'];source_name=x.get('source','')
        if task=='pose' and 'mean_keypoint_epe_px' in x:
            found=re.search(r'synthetic_validation_(\d+)\.json$',source_name)
            if not found:raise RuntimeError('Pose validation has no explicit update: '+source_name)
            for metric,unit in [('mean_keypoint_epe_px','px'),('centroid_epe_px','px'),('center_cell_accuracy','fraction')]:
                add(task,method,seed,int(found[1]),metric,x[metric],unit,source_name)
        elif task=='flow' and x.get('event')=='synthetic_validation':
            update=x.get('optimizer_updates',x.get('optimizer_update'))
            for metric in ('model_mean_epe_px_per_30ms','zero_flow_mean_epe_px_per_30ms'):
                add(task,method,seed,update,metric,x[metric],'px/30ms',source_name,x.get('updated_utc',''),x.get('epoch'))
        elif task=='reconstruction' and x.get('event')=='synthetic_validation':
            add(task,method,seed,x['optimizer_update'],'lpips_vgg_equal_rollout_mean',x['lpips_vgg_equal_rollout_mean'],'LPIPS-VGG',source_name,x.get('recorded_utc',''),x.get('epoch'))
    for r in runs:
        x=r.get('latest_metric') or {}
        if r['task']=='reconstruction' and x.get('event')=='synthetic_validation':
            add('reconstruction',r['method'],r['seed'],x['optimizer_update'],'lpips_vgg_equal_rollout_mean',x['lpips_vgg_equal_rollout_mean'],'LPIPS-VGG','training_matrix.latest_metric',x.get('recorded_utc',''),x.get('epoch'))
    ordered=sorted(points.values(),key=lambda x:(TASKS.index(x['task']),METHODS.index(x['method']),x['seed'],x['optimizer_update'],x['metric']))
    csv_new(out/'synthetic_validation_points.csv',ordered,list(ordered[0]))
    latest={}
    for x in ordered:
        key=(x['task'],x['method'],x['seed'],x['metric'])
        if key not in latest or latest[key]['optimizer_update']<x['optimizer_update']:latest[key]=x
    csv_new(out/'latest_synthetic_validation.csv',list(latest.values()),list(ordered[0]))
    # SD is across three independently seeded runs at exactly the same update.
    grouped=defaultdict(list)
    for x in ordered:
        if x['task']=='pose':grouped[(x['method'],x['optimizer_update'],x['metric'])].append(x)
    pose_summary=[]
    for (method,update,metric),xs in sorted(grouped.items()):
        if {x['seed'] for x in xs}!={0,1,2}:raise RuntimeError('Incomplete pose seed aggregation')
        vals=[x['value'] for x in xs]
        pose_summary.append(dict(method=method,optimizer_update=update,metric=metric,n_seeds=3,
            mean=statistics.mean(vals),sample_sd=statistics.stdev(vals),unit=xs[0]['unit']))
    csv_new(out/'pose_validation_3seed_summary.csv',pose_summary,list(pose_summary[0]))
    real=[]
    for t in TASKS:
        metrics=('lpips_alex','ssim','psnr_db','mse') if t=='reconstruction' else ('',)
        for m in METHODS:
            for s in range(3):
                for metric in metrics:real.append(dict(task=t,dataset={'reconstruction':'HQF14','flow':'MVSEC','pose':'E-POSE'}[t],method=m,seed=s,metric=metric,value='',status='NOT_EVALUATED',metric_binding='REGISTERED_HQF_PROTOCOL' if t=='reconstruction' else 'FILL_FROM_FROZEN_EVALUATION_PROTOCOL'))
    csv_new(out/'real_results_NOT_EVALUATED.csv',real,list(real[0]))
    provenance=dict(snapshot_utc=matrix['observed_utc'],campaign=matrix['campaign'],frozen_core=matrix['frozen_core'],
        recipe_sha256=matrix['recipe_sha256'],inputs={name:dict(size_bytes=(source/name).stat().st_size,sha256=digest(source/name)) for name in INPUT_SIZES},
        complete_counts={t:counts[t]['COMPLETE_VERIFIED'] for t in TASKS},
        validation_observation_counts={t:sum(x['task']==t for x in ordered) for t in TASKS},
        curves_scope={'pose':'Full 1000-update interval records for all15 runs; three-seed mean and sample SD.',
            'reconstruction':'Latest validation observations only; full validation history not downloaded.',
            'flow':'Eight tail validation observations only; full validation history not downloaded.'},
        real_results='NOT_EVALUATED; no historical Stage4/Stage5 scores imported',
        process_confirmation='Live/status files alone do not prove running; use separate parent process receipt.',
        reconciled_update='max(reported_update, latest_metric optimizer_update) when present; blanks stay missing, never zero.')
    with (out/'chart_provenance.json').open('x',encoding='utf-8') as f:json.dump(provenance,f,indent=2);f.write('\n')
    print(json.dumps(provenance,indent=2))

def make_plots(out,validation_csv=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
        'axes.titlesize':11,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':250})
    matrix=read_csv(out/'training_completion_matrix.csv');points=read_csv(out/'synthetic_validation_points.csv')
    summary=read_csv(out/'pose_validation_3seed_summary.csv');provenance=read(out/'chart_provenance.json')
    full_points=read_csv(validation_csv) if validation_csv else []
    snapshot=provenance['snapshot_utc'].split('.')[0].replace('T',' ')+' UTC'
    colors=dict(zip(METHODS,('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00')))
    def save(fig,name):
        for ext in ('png','pdf'):
            path=out/(name+'.'+ext)
            if path.exists():raise FileExistsError(path)
            fig.savefig(path,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.8),layout='constrained')
    for ax,task in zip(axes,TASKS):
        rows={(x['method'],int(x['seed'])):x for x in matrix if x['task']==task}
        value={'NOT_OBSERVED':0,'INCOMPLETE_OBSERVED':1,'COMPLETE_VERIFIED':2}
        grid=[[value[rows[(m,s)]['training_state']] for s in range(3)] for m in METHODS]
        ax.imshow(grid,cmap=ListedColormap(('#E6E6E6','#F3D58A','#B3D9CD')),vmin=0,vmax=2,aspect='auto')
        for i,m in enumerate(METHODS):
            for s in range(3):
                r=rows[(m,s)];u=r['reconciled_update']
                text=f'{int(u):,}\n/ {BUDGETS[task]:,}' if u else 'Not\nobserved'
                ax.text(s,i,text,ha='center',va='center',fontsize=8,color='#222222')
        ax.set(xticks=range(3),xticklabels=['Seed0','Seed1','Seed2'],yticks=range(5),yticklabels=[LABELS[m] for m in METHODS])
        ax.tick_params(length=0,pad=6)
        ax.set_title(f'{task.capitalize()} | {provenance["complete_counts"][task]}/15 complete\nBudget: {BUDGETS[task]:,} updates/run')
    fig.suptitle('Stage9 audited training completion',fontsize=14)
    fig.supxlabel('Green: verified complete; amber: partial training observed; gray: no training artifact observed.\nReal evaluation: NOT_EVALUATED | '+snapshot,fontsize=8)
    save(fig,'training_completion_matrix')

    fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,metric,title,ylabel in zip(axes,('mean_keypoint_epe_px','centroid_epe_px','center_cell_accuracy'),
        ('Keypoint error','Centroid error','Center-cell accuracy'),('Error (px), lower is better','Error (px), lower is better','Accuracy, higher is better')):
        for method in METHODS:
            xs=sorted([x for x in summary if x['method']==method and x['metric']==metric],key=lambda x:int(x['optimizer_update']))
            x=[int(a['optimizer_update'])/1000 for a in xs];y=[float(a['mean']) for a in xs];sd=[float(a['sample_sd']) for a in xs]
            ax.plot(x,y,color=colors[method],label=LABELS[method],lw=1.6)
            ax.fill_between(x,[v-d for v,d in zip(y,sd)],[v+d for v,d in zip(y,sd)],color=colors[method],alpha=.11,linewidth=0)
        ax.set(title=title,xlabel='Optimizer updates (thousands)',ylabel=ylabel);ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
        if metric=='center_cell_accuracy':ax.set_ylim(0,1)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Pose: synthetic validation only | 15/15 training complete',fontsize=13)
    fig.supxlabel('Three-seed mean ± sample SD; full 30,000-update budget. These are synthetic localization metrics, not real E-POSE scores.',fontsize=8)
    save(fig,'pose_synthetic_validation_curves')

    for task,metric,ylabel in [('reconstruction','lpips_vgg_equal_rollout_mean','LPIPS-VGG, lower is better'),('flow','model_mean_epe_px_per_30ms','EPE (px / 30 ms), lower is better')]:
        fig,ax=plt.subplots(figsize=(7.2,4),layout='constrained')
        for method in METHODS:
            for seed,marker in enumerate(('o','s','^')):
                xs=[x for x in points if x['task']==task and x['method']==method and int(x['seed'])==seed and x['metric']==metric]
                horizontal=[METHODS.index(method)+(seed-1)*.17 for x in xs] if task=='reconstruction' else [int(x['optimizer_update'])/1000 for x in xs]
                ax.scatter(horizontal,[float(x['value']) for x in xs],
                    color=colors[method],marker=marker,s=42,label=f'{LABELS[method]} / seed{seed}' if xs else None)
        if task=='flow':
            zero={float(x['value']) for x in points if x['task']=='flow' and x['metric']=='zero_flow_mean_epe_px_per_30ms'}
            if len(zero)==1:ax.axhline(next(iter(zero)),color='#555555',ls='--',lw=1,label='Zero-flow baseline')
        ax.set(xlabel='Optimizer updates (thousands)',ylabel=ylabel,title=f'{task.capitalize()}: available synthetic validation observations')
        if task=='reconstruction':
            from matplotlib.lines import Line2D
            ax.set(xticks=range(5),xticklabels=[LABELS[m] for m in METHODS],xlabel='Generator')
            ax.legend(handles=[Line2D([],[],color='#555555',marker=marker,ls='',label=f'Seed{seed}') for seed,marker in enumerate(('o','s','^'))],frameon=False)
        else:ax.legend(frameon=False,fontsize=7,ncol=2)
        ax.grid(alpha=.2)
        fig.supxlabel('Partial validation history; points are observations, not a complete learning curve. Real evaluation: NOT_EVALUATED.',fontsize=8)
        save(fig,task+'_synthetic_validation_observations')
    print('Created four PNG/PDF pairs. No real scores plotted.')
    if full_points:
        for task,metric,ylabel in [('reconstruction','lpips_vgg_equal_rollout_mean','LPIPS-VGG, lower is better'),('flow','model_mean_epe_px_per_30ms','EPE (px / 30 ms), lower is better')]:
            fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
            for seed,ax in enumerate(axes):
                for method in METHODS:
                    xs=sorted([x for x in full_points if x['task']==task and x['method']==method and int(x['seed'])==seed and x['metric']==metric],key=lambda x:int(x['optimizer_update']))
                    if not xs:continue
                    ax.plot([int(x['optimizer_update'])/1000 for x in xs],[float(x['value']) for x in xs],color=colors[method],lw=1.5,label=LABELS[method])
                if task=='flow':
                    zero=sorted({(int(x['optimizer_update']),float(x['value'])) for x in full_points if x['task']==task and int(x['seed'])==seed and x['metric']=='zero_flow_mean_epe_px_per_30ms'})
                    if zero:ax.plot([x[0]/1000 for x in zero],[x[1] for x in zero],color='#555555',ls='--',lw=1,label='Zero-flow baseline')
                ax.set(title=f'Seed{seed}',xlabel='Optimizer updates (thousands)',ylabel=ylabel);ax.grid(alpha=.2)
                ax.set_xlim(0,BUDGETS[task]/1000);ax.legend(frameon=False,fontsize=7)
            fig.suptitle(task.capitalize()+': synthetic validation history',fontsize=13)
            fig.supxlabel('Exact streamed log observations; partial runs end at their last observed validation. Real evaluation: NOT_EVALUATED.',fontsize=8)
            save(fig,task+'_synthetic_validation_curves')
        print('Created two additional streamed validation PNG/PDF pairs.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,default=Path(__file__).parent.parent/'training_snapshot_v2')
    p.add_argument('--out',type=Path,default=Path(__file__).parent);p.add_argument('--mode',choices=('csv','plot','all'),default='all')
    p.add_argument('--validation-csv',type=Path,help='Optional full history from stream_validation_curves.py');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.mode in ('csv','all'):export_data(a.source_dir,a.out)
    if a.mode in ('plot','all'):make_plots(a.out,a.validation_csv)

if __name__=='__main__':main()
