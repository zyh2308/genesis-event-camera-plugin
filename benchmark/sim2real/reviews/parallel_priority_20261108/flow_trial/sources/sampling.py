"""Pure train-index replay; no event values, GT flow vectors or real data."""
import hashlib,json
import numpy as np
STRATA=('camera_motion_static_scene','independent_rigid_motion_static_camera','camera_and_independent_rigid_motion','combined_motion_with_occlusion')

def need(ok,msg):
    if not ok:raise ValueError(msg)

def schedules(rollout_strata,updates=200):
    need(updates==200,'Only registered200-update experiment')
    ids=sorted(rollout_strata);need(len(ids)==24,'Need24 original train rollouts')
    samples=[(rid,i) for rid in ids for i in range(66)]
    tags=np.array([STRATA.index(rollout_strata[rid]) for rid,_ in samples])
    need(np.array_equal(np.bincount(tags,minlength=4),[396]*4),'Original equal4strata bank changed')
    control=[];seeds=[]
    for epoch in (45,46):
        perm=np.random.default_rng(1000003*epoch).permutation(1584)
        for batch in range(198):
            if len(control)>=updates*8:break
            control.extend(perm[8*batch:8*batch+8].tolist())
            seeds.extend([[0,epoch,batch,micro] for micro in range(8)])
    need(len(control)==1600 and len(seeds)==1600,'Wrong fixed replay size')
    candidate=control.copy();positions=np.flatnonzero(tags[control]==1)
    rng=np.random.default_rng(20261108);replace=rng.permutation(positions)[160:]
    current=np.bincount(tags[control],minlength=4);deficits=np.array([480,160,480,480])-current
    need(deficits[1]<=0 and all(deficits[i]>=0 for i in (0,2,3)),'Unexpected original class counts')
    replacements=np.repeat([0,2,3],[int(deficits[i]) for i in (0,2,3)])
    rng.shuffle(replacements);need(len(replace)==len(replacements),'Class replacements do not balance')
    pools={i:np.flatnonzero(tags==i) for i in (0,2,3)}
    for pos,group in zip(replace,replacements):candidate[int(pos)]=int(rng.choice(pools[int(group)]))
    need(np.array_equal(np.bincount(tags[candidate],minlength=4),[480,160,480,480]),'Target mixture differs')
    need(all(a==b for j,(a,b) in enumerate(zip(control,candidate)) if tags[a]!=1),'Changed a non-low-motion position')
    result=dict(schema='flow_shared_sampler_replay_v1',seed=0,base_epoch=45,updates=200,batch=8,samples=[list(x) for x in samples],
                augmentation_seed_sequences=seeds,arms={'control':control,'motion':candidate},
                counts={'control':np.bincount(tags[control],minlength=4).tolist(),'motion':[480,160,480,480]},
                changed_positions=len(replace),preserved_fraction=1-len(replace)/1600.,stratum_order=list(STRATA))
    result['replay_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest()
    return result
