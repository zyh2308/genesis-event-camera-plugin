from pathlib import Path

root=Path(__file__).parent
s=(root/'pose_physical_light_adaptation.py').read_text()
start=s.index('def fit(');end=s.index('\n\ndef main():',start)
fit='''def matched_affine(a,normal,target,rng):
    import copy
    paired=np.random.default_rng()
    paired.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
    first,y=shared.affine(a,target,rng)
    second,yy=shared.affine(normal,target,paired)
    assert torch.equal(y,yy)
    return first,second,y


def fit(scale,state,normal,target,aux,aux_target,seed,smoke,aux_clean,pair_weight):
    model=scale.CenterScale().to(normal.device)
    model.load_state_dict(state,strict=True)
    teacher=scale.CenterScale().to(normal.device).eval()
    teacher.load_state_dict(state,strict=True);teacher.requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    rng=np.random.default_rng(71070+seed)
    history=[];updates=0
    for epoch in range(1 if smoke else 6):
        order=rng.permutation(len(normal))
        if smoke:order=order[:32]
        totals=[]
        for start in range(0,len(order),32):
            ids=torch.as_tensor(order[start:start+32],device=normal.device)
            aux_ids=torch.as_tensor(rng.integers(len(aux),size=len(ids)),device=normal.device)
            x,y=shared.affine(normal[ids],target[ids],rng)
            a,paired,b=matched_affine(aux[aux_ids],aux_clean[aux_ids],aux_target[aux_ids],rng)
            clean,pred=model(x),model(a)
            with torch.no_grad():reference=teacher(x);paired_reference=teacher(paired)
            loss=.5*(shared.supervised(model,clean,y)+shared.supervised(model,pred,b))
            loss=loss+2*(4*shared.smooth(clean[1],reference[1])+shared.smooth(clean[2],reference[2])
                +2*shared.smooth(clean[3]-clean[2],reference[3]-reference[2]))
            loss=loss+pair_weight*(shared.smooth(pred[2],paired_reference[2])
                +2*shared.smooth(pred[3]-pred[2],paired_reference[3]-paired_reference[2]))
            opt.zero_grad(set_to_none=True);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            assert torch.isfinite(loss) and norm>0 and all(p.grad is None for p in teacher.parameters())
            opt.step();updates+=1;totals.append(float(loss.detach()))
        history.append(float(np.mean(totals)))
        print('DEPTH_EPOCH',seed,pair_weight,epoch+1,updates,history[-1],flush=True)
    assert updates==(1 if smoke else 288)
    assert all(torch.equal(v.cpu(),state[k]) for k,v in teacher.state_dict().items())
    return model.eval(),history
'''
s=s[:start]+fit+s[end:]
s=s.replace("arms=['clean_control','physical_augmented']","arms=['clean_control','physical_supervised','paired_consistency']")
s=s.replace("auxiliary={'clean_control':shared.tensors(controls,device),'physical_augmented':shared.tensors(relit,device)}", "control_tensor=shared.tensors(controls,device);relit_tensor=shared.tensors(relit,device)\n        auxiliary={'clean_control':control_tensor,'physical_supervised':relit_tensor,'paired_consistency':relit_tensor}")
needle="            result=dict(method=method,seed=seed,arms={},train_loss={},parents=parent)"
replace="""            control=R.parent/'20261107-pose-physical-light-train-v2/physical_adaptation'/f'{tag}_clean_control.pt'
            state=torch.load(control,map_location='cpu',weights_only=True)
            localizer=scale.CenterScale().to(device).eval();localizer.load_state_dict(state,strict=True)
            localizer.requires_grad_(False)
            fixed_centers={c:scale.estimate(localizer,frames,[r['K'] for r in conditions[c]])[0] for c,frames in views.items()}
            result=dict(method=method,seed=seed,arms={},train_loss={},parents=parent,
                localization_path=str(control),localization_sha256=shared.sha(control),final_center_source='frozen localizer',
                rotation_source='same original frozen scale/rotation reference in every arm')"""
assert needle in s;s=s.replace(needle,replace)
s=s.replace("fit(scale,state,x,y,auxiliary[arm],aux_y,seed,args.smoke)","fit(scale,state,x,y,auxiliary[arm],aux_y,seed,args.smoke,control_tensor,2 if arm=='paired_consistency' else 0)")
s=s.replace("                    uv,z,size=scale.estimate(model,frames,K)","                    _,z,size=scale.estimate(model,frames,K)\n                    uv=fixed_centers[c]\n                    assert np.array_equal(uv,scale.estimate(localizer,frames,K)[0])")
s=s.replace("print('PHYSICAL_ADAPTATION'","print('INDEPENDENT_DEPTH'")
s=s.replace("run['arms']['physical_augmented']","run['arms']['paired_consistency']")
needle="                    passed=b[c]['summary']['depth_mae_m']<a[c]['summary']['depth_mae_m']))"
assert needle in s
s=s.replace(needle,needle+"\n                checks.append(dict(method=run['method'],seed=run['seed'],condition=c+'_pose',\n                    passed=b[c]['summary']['success']>=a[c]['summary']['success']))")
s=s.replace("each seed improves depth at half and eighth physical radiance; both methods", "each seed improves depth at half and eighth physical radiance and never lowers relit pose success; both methods")
s=s.replace("teacher_clean_weight=2,", "teacher_clean_weight=2,paired_teacher_weight=2,final_center='frozen clean-control localizer',")
compile(s,'pose_independent_depth_consistency.py','exec')
(root/'pose_independent_depth_consistency.py').write_text(s)

r=(root/'run_pose_light_study.py').read_text()
needle="        specs=[('radiance_train','prepare_pose_train_radiance.py',['--campaign',str(campaign),\n                   '--out',str(root/'radiance_train')])]"
assert needle in r
r=r.replace(needle,"        assert (root/'radiance_train/completion.json').is_file()\n        specs=[]")
r=r.replace("'pose_physical_light_adaptation.py'","'pose_independent_depth_consistency.py'")
(root/'run_pose_independent_depth_study.py').write_text(r)
compile(r,'run_pose_independent_depth_study.py','exec')
print('built independent depth trainer and bounded resource runner')
