"""Train the native final affine in scale and focal-normalized depth coordinates."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from pose_priority_policy import depth_retention, draw_auxiliary, state_sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


class DepthCoordinateLinear(nn.Linear):
    """Store (s, q), expose the original native raw (s, h=s+q) output."""

    def __init__(self, native_layer):
        nn.Module.__init__(self)
        require(type(native_layer) is nn.Linear and native_layer.out_features == 2
                and native_layer.bias is not None, 'Expected the original two-output affine')
        self.in_features = native_layer.in_features
        self.out_features = native_layer.out_features
        for name in ('weight', 'bias'):
            parameter = getattr(native_layer, name)
            old = parameter.detach()
            coordinates = torch.stack((old[0], old[1] - old[0]))
            setattr(self, name, nn.Parameter(coordinates, requires_grad=parameter.requires_grad))

    def forward(self, features):
        scale, normalized_depth = super().forward(features).unbind(-1)
        return torch.stack((scale, scale + normalized_depth), dim=-1)


def install_depth_coordinates(model):
    require(isinstance(model.size, nn.Sequential), 'Unexpected native depth head')
    model.size[-1] = DepthCoordinateLinear(model.size[-1])
    return model


def native_state(model):
    require(isinstance(model.size[-1], DepthCoordinateLinear), 'Missing training coordinate head')
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    prefix = f'size.{len(model.size) - 1}.'
    for name in ('weight', 'bias'):
        tensor = state[prefix + name]
        tensor[1] = tensor[0] + tensor[1]
    return state


def outputs(model, inputs, focal_log):
    logits, uv, scale, shape = model(inputs)
    return dict(logits=logits, uv=uv, log_size=scale, log_shape=shape,
                depth_m=(shape + focal_log - scale).exp(), size=scale.exp())


@torch.inference_mode()
def verify_prediction_parity(first, second, inputs, focal_log):
    before = outputs(first, inputs, focal_log)
    after = outputs(second, inputs, focal_log)
    maxima = {}
    for name, value in before.items():
        difference = (value - after[name]).abs().max().item()
        require(np.isfinite(difference) and difference <= 2e-6, f'Coordinate parity failed for {name}')
        maxima[name] = difference
    return maxima


def build_native(scale, state, device):
    with torch.random.fork_rng(devices=[]):
        model = scale.CenterScale()
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def verify_adam_readback(scale, checkpoint, expected_state_sha, updates):
    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require(restored['update'] == updates, 'Checkpoint cursor differs')
    with torch.random.fork_rng(devices=[]):
        model = install_depth_coordinates(scale.CenterScale())
        model.load_state_dict(restored['model'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.00015, weight_decay=.01)
        optimizer.load_state_dict(restored['optimizer'])
    require(state_sha(model.state_dict()) == expected_state_sha, 'Coordinate model readback differs')
    require(len(optimizer.state) == len(tuple(model.parameters())), 'Adam parameter inventory differs')
    require({int(value['step']) for value in optimizer.state.values()} == {updates}, 'Adam counters differ')
    for before, after in zip(restored['optimizer']['state'].values(), optimizer.state.values(), strict=True):
        require(before.keys() == after.keys(), 'Adam slot keys differ')
        require(all(torch.equal(before[key], after[key].cpu()) for key in before), 'Adam slots differ')


def fit_depth(native, scale, source, data, method, seed, smoke, out, original_log):
    """Run the C1 objective and tape with a single trainable coordinate change."""
    shared = native.shared
    normal, target, auxiliary, auxiliary_target, clean_auxiliary = data
    half_count = len(auxiliary) // 2
    require(len(auxiliary) == len(auxiliary_target) == len(clean_auxiliary) == 2 * half_count,
            'Original auxiliary layout differs')
    require(torch.equal(auxiliary_target[:half_count], auxiliary_target[half_count:]),
            'Original auxiliary labels differ')
    torch.manual_seed(90710 + seed)
    model = scale.CenterScale().to(normal.device)
    model.load_state_dict(source, strict=True)
    teacher = scale.CenterScale().to(normal.device).eval()
    teacher.load_state_dict(source, strict=True)
    teacher.requires_grad_(False)
    source_hash = state_sha(source)
    parameter_names = tuple(name for name, _ in model.named_parameters())
    install_depth_coordinates(model)
    require(tuple(name for name, _ in model.named_parameters()) == parameter_names,
            'Parameter inventory changed')
    initial_basis_hash = state_sha(model.state_dict())
    probe = normal[:32]
    probe_focal = target[:32, 3]
    initial_parity = verify_prediction_parity(teacher, model, probe, probe_focal)
    initial_round_trip = verify_prediction_parity(
        teacher, build_native(scale, native_state(model), normal.device), probe, probe_focal)

    parameters = tuple(model.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=.00015, weight_decay=.01)
    rng = np.random.default_rng(71070 + seed)
    labels = hashlib.sha256()
    exposure = dict(normal=0, half=0, eighth=0)
    updates = 0
    history = []
    first_oracle = None
    evidence = Path(out) / 'training_evidence'
    evidence.mkdir(exist_ok=True)
    tag = f'{method}_seed{seed}_focal_normalized_depth_coordinate'
    log = evidence / (tag + '.jsonl')
    with log.open('x') as stream:
        for epoch in range(1 if smoke else 6):
            order = rng.permutation(len(normal))
            if smoke:
                order = order[:32]
            losses = []
            for offset in range(0, len(order), 32):
                indices = order[offset:offset + 32]
                aux_indices, half = draw_auxiliary(rng, half_count, len(indices))
                normal_index = torch.as_tensor(indices, device=normal.device)
                aux_index = torch.as_tensor(aux_indices, device=normal.device)
                x, y = shared.affine(normal[normal_index], target[normal_index], rng)
                a, _, b = native.matched_affine(auxiliary[aux_index], clean_auxiliary[aux_index],
                                                auxiliary_target[aux_index], rng)
                clean, relit = model(x), model(a)
                with torch.no_grad():
                    reference = teacher(x)
                supervised = .5 * (shared.supervised(model, clean, y) + shared.supervised(model, relit, b))
                retention = depth_retention(clean, reference, include_center=True).mean()
                loss = supervised + 2 * retention
                row = dict(update=updates + 1, normal_indices=indices.tolist(),
                           auxiliary_indices=aux_indices.tolist(), half_mask=half.tolist(),
                           supervised=supervised.detach().item(), clean_retention=retention.detach().item(),
                           loss=loss.detach().item())
                old = original_log[updates]
                require(all(row[key] == old[key] for key in ('update', 'normal_indices', 'auxiliary_indices', 'half_mask')),
                        'Actual candidate draw differs from original C1')
                if updates == 0:
                    differences = {key: abs(row[key] - old[key]) for key in ('supervised', 'clean_retention', 'loss')}
                    limit = 2e-6 + 1e-6 * abs(old['loss'])
                    require(all(np.isfinite(value) and value <= limit for value in differences.values()),
                            'First original C1 scalar oracle differs')
                    first_oracle = dict(passed=True, absolute_errors=differences, tolerance=limit)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters),
                        'Missing/nonfinite native depth gradients')
                require(all(p.grad is None for p in teacher.parameters()), 'Frozen teacher received gradients')
                norm = nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
                require(torch.isfinite(loss) and norm > 0, 'Invalid depth objective or gradient')
                optimizer.step()
                updates += 1
                require({int(v['step']) for v in optimizer.state.values()} == {updates}, 'Adam counters differ')
                labels.update(y.detach().cpu().numpy().tobytes())
                labels.update(b.detach().cpu().numpy().tobytes())
                exposure['normal'] += len(indices)
                exposure['half'] += int(half.sum())
                exposure['eighth'] += int((~half).sum())
                row.update(gradient_l2=norm.item(), finite_gradient_tensors=len(parameters))
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                losses.append(row['loss'])
            history.append(float(np.mean(losses)))
            print('DEPTH_COORDINATE_EPOCH', method, seed, epoch + 1, updates, history[-1], flush=True)

    require(updates == (1 if smoke else 288), 'Incorrect registered budget')
    require(state_sha(teacher.state_dict()) == source_hash, 'Frozen teacher state changed')
    model.eval()
    output_state = native_state(model)
    output_model = build_native(scale, output_state, normal.device)
    final_parity = verify_prediction_parity(model, output_model, probe, probe_focal)
    basis_hash = state_sha(model.state_dict())
    checkpoint = evidence / (tag + '_coordinate_training_state.pt')
    torch.save(dict(schema='pose_depth_coordinate_optimizer_v1', model=model.state_dict(), optimizer=optimizer.state_dict(),
                    update=updates, source_state_sha256=source_hash, numpy_rng=rng.bit_generator.state,
                    torch_cpu_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state_all() if normal.is_cuda else [],
                    native_output_state_sha256=state_sha(output_state)), checkpoint)
    verify_adam_readback(scale, checkpoint, basis_hash, updates)
    final_model = Path(out) / (tag + '.pt')
    torch.save(output_state, final_model)
    require(state_sha(torch.load(final_model, map_location='cpu', weights_only=True)) == state_sha(output_state),
            'Native final checkpoint readback differs')
    receipt = dict(method=method, seed=seed, updates=updates, source_state_sha256=source_hash,
                   initial_coordinate_state_sha256=initial_basis_hash, final_coordinate_state_sha256=basis_hash,
                   final_native_state_sha256=state_sha(output_state), labels_sha256=labels.hexdigest(), exposure=exposure,
                   teacher_unchanged=True, parameter_names_unchanged=True, first_native_loss_oracle=first_oracle,
                   initial_source_to_coordinate_parity=initial_parity, initial_coordinate_to_native_parity=initial_round_trip,
                   final_coordinate_to_native_parity=final_parity, independent_model_adam_readback=True,
                   resume_supported=False, training_state_path=str(checkpoint), final_model_path=str(final_model),
                   training_log_path=str(log), train_loss=history)
    return output_model, receipt
