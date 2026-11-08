"""Local-only AST and spy audit. No Genesis, physics, remote access, or weights."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UP = HERE / 'upstream_reference'
ENV = UP / 'DcmmVecEnv.py'
PPO = UP / 'ppo_dcmm_catch_two_stage.py'


def methods(path, cls):
    node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, ast.ClassDef) and n.name == cls)
    return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}


def compile_method(node):
    scope = dict(np=np, torch=torch)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
                 '<verbatim-upstream-method>', 'exec'), scope)
    return scope[node.name]


env_methods = methods(ENV, 'DcmmVecEnv')
ppo_methods = methods(PPO, 'PPO_Catch_TwoStage')
stage_sites = []
for name, node in env_methods.items():
    for sub in ast.walk(node):
        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == 'self' and sub.attr == 'stage':
            stage_sites.append(dict(method=name, line=sub.lineno, mode=type(sub.ctx).__name__))
assert {r['method'] for r in stage_sites} == {'__init__', 'update_stage', 'reset', 'compute_reward', 'step'}
for name in ('obs2tensor', 'model_act', 'action2dict'):
    assert not any(isinstance(n, ast.Attribute) and n.attr in ('stage', 'info', 'terminated', 'reward_touch', 'step_touch')
                   for n in ast.walk(ppo_methods[name]))


class TaskOnlyEnv:
    def __init__(self, stage):
        self.stage = stage
        self.accesses = []

    def call(self, name):
        self.accesses.append(name)
        assert name == 'task', name
        return ['Catching']


class SpyActor:
    def act_inference(self, inp):
        self.last = {k: v.detach().clone() for k, v in inp.items()}
        # This is a structural spy, not original checkpoint inference.
        return inp['obs'].sum(dim=1, keepdim=True).repeat(1, 18) / 100


rows = []
reference_tensor = reference_action = reference_ctrl = None
for stage, truth in [('tracking', [8., 9., 10.]), ('grasping', [-8., -9., -10.])]:
    e = NS(
        _get_relative_ee_pos3d=lambda: np.array([.1, .2, .8]),
        _get_relative_object_pos3d=lambda: np.asarray(truth),
        _get_base_vel=lambda: np.array([.01, .02]),
        _get_relative_ee_quat=lambda: np.array([1., 0., 0., 0.]),
        _get_hand_obs=lambda: np.arange(12) / 100,
        Dcmm=NS(data=NS(qpos=np.arange(43) / 100, time=.8)),
        init_pos=False, prev_ee_pos3d=np.array([.09, .19, .79]),
        prev_obj_pos3d=np.asarray(truth)-.1, fps=25,
        k_obs_base=.002, k_obs_arm=.002, k_obs_hand=.002, k_obs_object=.002,
        print_obs=False, stage=stage,
    )
    np.random.seed(81231)
    obs = compile_method(env_methods['_get_obs'])(e)
    native_object = {k: v.tolist() for k, v in obs['object'].items()}
    obs['object'] = dict(pos3d=np.array([.2, 1.1, .9]), v_lin_3d=np.array([.1, -2., -.3]))
    batched = {k: ({s: np.asarray(v)[None] for s, v in value.items()} if isinstance(value, dict)
                   else np.asarray(value)[None]) for k, value in obs.items()}
    proxy = TaskOnlyEnv(stage)
    actor = SpyActor()
    a = NS(env=proxy, device='cpu', model=actor, running_mean_std_track=lambda x: x,
           running_mean_std_hand=lambda x: x,
           action_track_denorm=[1.5, .025, .01], action_catch_denorm=[1.5, .025, .15])
    tensor = compile_method(ppo_methods['obs2tensor'])(a, batched)
    action = compile_method(ppo_methods['model_act'])(a, {'obs': tensor}, inference=True)['actions']
    action_dict = compile_method(ppo_methods['action2dict'])(a, action)
    e.Dcmm.move_base_vel = lambda target: (np.arange(4)/100, np.arange(4)/200)
    e.Dcmm.arm_pid = NS(update=lambda *args: np.arange(6)/100)
    e.Dcmm.hand_pid = NS(update=lambda *args: np.arange(16)/100)
    e.action_buffer = dict(base=[action_dict['base'][0]], arm=[np.zeros(6)], hand=[np.zeros(16)])
    e.k_act = .01
    e.print_ctrl = False
    np.random.seed(991)
    ctrl = compile_method(env_methods['_get_ctrl'])(e)
    if reference_tensor is None:
        reference_tensor, reference_action, reference_ctrl = tensor.clone(), action.clone(), ctrl.copy()
    else:
        assert torch.equal(reference_tensor, tensor)
        assert torch.equal(reference_action, action)
        assert np.array_equal(reference_ctrl, ctrl)
    rows.append(dict(stage=stage, deliberately_different_native_GT_object=native_object,
                     actor_tensor=tensor.tolist(), actor_env_accesses=proxy.accesses,
                     controller_output=ctrl.tolist()))

summary = dict(
    scope='Static upstream dataflow plus AST-executed spy probes; no original NN weights or physics run',
    sources=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (ENV, PPO)],
    stage_attribute_sites=sorted(stage_sites, key=lambda r: r['line']),
    checks=dict(stage_read_absent_from_actor_bridge=True,
                only_task_queried_from_actor_env=True,
                different_GT_object_values_do_not_survive_complete_object_overwrite=True,
                stage_and_GT_object_perturbation_leave_actor_tensor_equal=True,
                stage_perturbation_leaves_fixed_noise_low_level_controller_equal=True),
    caveat='GT stage still changes reward and episode termination at NativeEnv.step849-853; it does not command grasping.',
    probes=rows,
)
(HERE / 'gt_stage_dataflow_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(summary['checks'], indent=2))
