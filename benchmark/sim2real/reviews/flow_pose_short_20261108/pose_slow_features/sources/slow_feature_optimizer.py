"""One fixed learning-rate factor; no feature freezing or distillation."""
import hashlib,json
import torch

FEATURES_LR=1.5e-5
HEADS_LR=1.5e-4
WEIGHT_DECAY=.01

def need(ok,message):
    if not ok:raise ValueError(message)

def expected_groups(model,candidate):
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    need(named,'No trainable parameters')
    if not candidate:return [('all_parameters',HEADS_LR,named)]
    features=[(n,p) for n,p in named if n.startswith('features.')]
    heads=[(n,p) for n,p in named if not n.startswith('features.')]
    need(features and heads,'Candidate requires both trainable features and heads')
    ids=[id(p) for _,_,group in [('features',FEATURES_LR,features),('heads',HEADS_LR,heads)] for _,p in group]
    need(len(ids)==len(set(ids))==len(named) and set(ids)=={id(p) for _,p in named},'Groups overlap or omit trainable parameters')
    return [('features',FEATURES_LR,features),('heads',HEADS_LR,heads)]

def validate_optimizer(model,optimizer,candidate):
    expected=expected_groups(model,candidate)
    need(len(optimizer.param_groups)==len(expected),'Wrong number of AdamW parameter groups')
    result=[]
    for group,(name,lr,named) in zip(optimizer.param_groups,expected):
        need(group['lr']==lr and group['weight_decay']==WEIGHT_DECAY and group.get('group_name')==name,'Changed fixed LR/weight-decay/group identity')
        need([id(p) for p in group['params']]==[id(p) for _,p in named],'Optimizer parameter identity/order differs')
        result.append(dict(name=name,lr=lr,weight_decay=WEIGHT_DECAY,parameter_names=[n for n,_ in named],
                           shapes=[list(p.shape) for _,p in named],parameter_tensors=len(named)))
    return dict(groups=result,sha256=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),
                candidate=candidate,feature_lr_ratio=.1 if candidate else 1.,feature_KD_weight=0.,half_teacher_weight=0.)

def make_optimizer(model,candidate):
    groups=[dict(params=[p for _,p in named],lr=lr,weight_decay=WEIGHT_DECAY,group_name=name)
            for name,lr,named in expected_groups(model,candidate)]
    optimizer=torch.optim.AdamW(groups,lr=HEADS_LR,weight_decay=WEIGHT_DECAY)
    return optimizer,validate_optimizer(model,optimizer,candidate)
