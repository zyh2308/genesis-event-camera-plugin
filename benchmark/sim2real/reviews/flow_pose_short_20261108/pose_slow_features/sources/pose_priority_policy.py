"""Synthetic-only half-view retention; no simulator or inference modifications."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HALF_PROBABILITY = .75


def state_sha(state):
    value = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value.update(name.encode()); value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return value.hexdigest()


def draw_auxiliary(rng, half_count, batch_size):
    if half_count <= 0 or batch_size <= 0:
        raise ValueError('Auxiliary conditions must be nonempty')
    half = rng.random(batch_size) < HALF_PROBABILITY
    physical = rng.integers(half_count, size=batch_size)
    return physical + (~half) * half_count, half


def per_sample_smooth(predicted, target):
    value = F.smooth_l1_loss(predicted, target, beta=.1, reduction='none')
    return value.reshape(len(value), -1).mean(1)


def center_supervision(model, prediction, target):
    dist = (model.grid[None] - target[:, None, :2]).square().sum(-1)
    heat = (-.5 * dist * 16**2).softmax(1)
    return -(heat * prediction[0].log_softmax(1)).sum(1).mean() + 4 * per_sample_smooth(prediction[1], target[:, :2]).mean()


def center_retention(prediction, reference):
    return (4 * per_sample_smooth(prediction[1], reference[1])
            + F.kl_div(prediction[0].log_softmax(1), reference[0].softmax(1), reduction='none').sum(1))


def depth_retention(prediction, reference, include_center=False):
    result = (per_sample_smooth(prediction[2], reference[2])
              + 2 * per_sample_smooth(prediction[3] - prediction[2], reference[3] - reference[2]))
    if include_center:
        result = result + 4 * per_sample_smooth(prediction[1], reference[1])
    return result


def masked_mean(values, mask):
    """Average over the original batch; no amplification when half samples are few."""
    if values.ndim != 1 or mask.shape != values.shape:
        raise ValueError('One retention value and mask per physical sample required')
    return (values * mask.to(device=values.device, dtype=values.dtype)).mean()


def gradient_evidence(supervised, consistency, parameters):
    first = torch.autograd.grad(supervised, parameters, retain_graph=True, allow_unused=True)
    second = torch.autograd.grad(consistency, parameters, retain_graph=True, allow_unused=True)
    pairs = [(torch.zeros_like(p) if a is None else a,
              torch.zeros_like(p) if b is None else b)
             for p, a, b in zip(parameters, first, second)]
    a2 = sum(a.double().square().sum() for a, _ in pairs)
    b2 = sum(b.double().square().sum() for _, b in pairs)
    dot = sum((a.double() * b.double()).sum() for a, b in pairs)
    return dict(supervised_gradient_l2=float(a2.sqrt()), consistency_gradient_l2=float(b2.sqrt()),
                cosine=float(dot / (a2 * b2).sqrt()) if a2 > 0 and b2 > 0 else None)


class BoundedFitter:
    """Replace only the old two fit functions; retain its data/evaluation/gates."""
    def __init__(self, native, out, smoke):
        self.native, self.shared, self.out, self.smoke = native, native.shared, Path(out), smoke
        self.order = [(method, seed, arm, branch) for method in self.shared.METHODS
                      for seed in ([0] if smoke else range(3))
                      for arm in ('clean_control', 'physical_supervised', 'paired_consistency')
                      for branch in ('depth', 'center')]
        self.records, self.parents, self.probes = [], {}, {}

    def fit(self, *args):
        return self._fit('depth', *args)

    def fit_center(self, *args):
        return self._fit('center', *args)

    def _fit(self, branch, scale, state, normal, target, aux, aux_target, seed, smoke, aux_clean, pair_weight):
        method, expected_seed, arm, expected_branch = self.order[len(self.records)]
        assert seed == expected_seed and smoke == self.smoke and branch == expected_branch
        actual_arm = 'clean_control' if aux is aux_clean else 'paired_consistency' if pair_weight else 'physical_supervised'
        assert arm == actual_arm and pair_weight == (2 if arm == 'paired_consistency' else 0)
        assert len(aux) == len(aux_clean) == len(aux_target) and len(aux) % 2 == 0
        half_count = len(aux) // 2
        assert torch.equal(aux_target[:half_count], aux_target[half_count:])
        assert torch.equal(aux_clean[:half_count], aux_clean[half_count:])
        source_hash = state_sha(state)
        key = (method, seed)
        if key in self.parents: assert self.parents[key] == source_hash
        self.parents[key] = source_hash
        torch.manual_seed(90710 + seed)
        model = scale.CenterScale().to(normal.device); model.load_state_dict(state, strict=True)
        teacher = scale.CenterScale().to(normal.device).eval(); teacher.load_state_dict(state, strict=True)
        teacher.requires_grad_(False)
        if branch == 'center': model.size.requires_grad_(False)
        parameters = tuple(p for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(parameters, lr=.00015, weight_decay=.01)
        rng = np.random.default_rng(71070 + seed)
        directory = self.out / 'training_evidence'; directory.mkdir(exist_ok=True)
        tag = f'{method}_seed{seed}_{arm}_{branch}'
        supervision = (lambda p, y: self.shared.supervised(model, p, y)) if branch == 'depth' else lambda p, y: center_supervision(model, p, y)
        retention = depth_retention if branch == 'depth' else center_retention
        probe_key = (method, seed, branch)
        if probe_key not in self.probes and arm != 'clean_control':
            probes = {}
            n = min(8, half_count)
            with torch.no_grad(): reference = teacher(aux_clean[:n])
            for condition, data, y in [('half', aux[:n], aux_target[:n]),
                                       ('eighth', aux[half_count:half_count+n], aux_target[half_count:half_count+n])]:
                predicted = model(data)
                supervised = supervision(predicted, y)
                legacy = retention(predicted, reference).mean()
                probes[condition] = gradient_evidence(supervised, legacy, parameters)
            self.shared.write(directory / (f'{method}_seed{seed}_{branch}_legacy_gradient_probe.json'), probes)
            self.probes[probe_key] = probes
        history, updates = [], 0
        exposure = dict(normal=0, half=0, eighth=0)
        labels = hashlib.sha256()
        with (directory / (tag + '.jsonl')).open('x') as stream:
            for epoch in range(1 if smoke else 6):
                order = rng.permutation(len(normal)); totals = []
                if smoke: order = order[:32]
                for offset in range(0, len(order), 32):
                    indices = order[offset:offset+32]
                    ai, half = draw_auxiliary(rng, half_count, len(indices))
                    idx = torch.as_tensor(indices, device=normal.device)
                    aid = torch.as_tensor(ai, device=normal.device)
                    x, y = self.shared.affine(normal[idx], target[idx], rng)
                    a, paired, b = self.native.matched_affine(aux[aid], aux_clean[aid], aux_target[aid], rng)
                    clean, predicted = model(x), model(a)
                    with torch.no_grad():
                        reference = teacher(x)
                        same_view_reference = teacher(a) if pair_weight else None
                    supervised = .5 * (supervision(clean, y) + supervision(predicted, b))
                    clean_term = (depth_retention(clean, reference, include_center=True).mean()
                                  if branch == 'depth' else center_retention(clean, reference).mean())
                    half_term = masked_mean(retention(predicted, same_view_reference), torch.as_tensor(half, device=normal.device)) if pair_weight else supervised * 0
                    loss = supervised + 2 * clean_term + pair_weight * half_term
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                    assert torch.isfinite(loss) and norm > 0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                    assert all(p.grad is None for p in teacher.parameters())
                    optimizer.step(); updates += 1
                    steps = sorted({int(v['step']) for v in optimizer.state.values()}); assert steps == [updates]
                    labels.update(y.detach().cpu().numpy().tobytes()); labels.update(b.detach().cpu().numpy().tobytes())
                    exposure['normal'] += len(indices) * (2 if arm == 'clean_control' else 1)
                    if arm != 'clean_control':
                        exposure['half'] += int(half.sum()); exposure['eighth'] += int((~half).sum())
                    row = dict(update=updates, seed=seed, arm=arm, branch=branch,
                               normal_indices=indices.tolist(), auxiliary_indices=ai.tolist(), half_mask=half.tolist(),
                               supervised=float(supervised.detach()), clean_retention=float(clean_term.detach()),
                               half_same_view_retention=float(half_term.detach()), loss=float(loss.detach()),
                               gradient_l2=float(norm), finite_gradient_tensors=len(parameters))
                    stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                    totals.append(row['loss'])
                history.append(float(np.mean(totals)))
                print('POSE_PRIORITY_EPOCH', method, seed, arm, branch, epoch+1, updates, history[-1], flush=True)
        assert updates == (1 if smoke else 288)
        assert state_sha(teacher.state_dict()) == source_hash
        if branch == 'center': assert all(torch.equal(v.cpu(), state[k]) for k, v in model.state_dict().items() if k.startswith('size.'))
        checkpoint = directory / (tag + '_training_state.pt')
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                        torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state_all() if normal.is_cuda else [],
                        update=updates, source_state_sha256=source_hash, real_data_used=False), checkpoint)
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state_sha(restored['model']) == state_sha(model.state_dict())
        assert sorted({int(v['step']) for v in restored['optimizer']['state'].values()}) == [updates]
        cpu_rng = torch.get_rng_state()
        try:
            rebuilt = scale.CenterScale(); rebuilt.load_state_dict(restored['model'], strict=True)
            if branch == 'center': rebuilt.size.requires_grad_(False)
            rebuilt_optimizer = torch.optim.AdamW([p for p in rebuilt.parameters() if p.requires_grad], lr=.00015, weight_decay=.01)
            rebuilt_optimizer.load_state_dict(restored['optimizer'])
            assert state_sha(rebuilt.state_dict()) == state_sha(model.state_dict())
            for before, after in zip(restored['optimizer']['state'].values(), rebuilt_optimizer.state.values()):
                assert before.keys() == after.keys()
                assert all(torch.equal(before[k].cpu(), after[k].cpu()) for k in before)
        finally:
            torch.set_rng_state(cpu_rng)
        record = dict(method=method, seed=seed, arm=arm, branch=branch, updates=updates,
                      source_state_sha256=source_hash, final_state_sha256=state_sha(model.state_dict()),
                      teacher_unchanged=True, labels_sha256=labels.hexdigest(), exposure=exposure,
                      independent_model_optimizer_rebuild=True, optimizer_state_equal=True,
                      checkpoint_path=str(checkpoint), checkpoint_sha256=self.shared.sha(checkpoint))
        self.shared.write(directory / (tag + '_receipt.json'), record); self.records.append(record)
        return model.eval(), history
