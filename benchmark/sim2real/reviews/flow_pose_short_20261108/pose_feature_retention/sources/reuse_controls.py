"""Hash-locked reuse of immutable C0/C1, with independent input-trajectory replay."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pose_feature_retention_policy import state_sha, draw_auxiliary

STUDY_SHA = '6540c8e62b3606376641590fa267bf90111e889e279213ab4f6a8537c2cc2ac0'
RECEIPT_SHA = 'e74399d0d9e54b39afe33e087c41a5349b2df5fb162e8f16214477da16497f51'
DIAGNOSTIC_SHA = 'f89f02cfa4f66aa095692263ac1eb31a4d41d11828c8ea3aa060064e84dfa25e'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()


def record(path):
    p = Path(path)
    return dict(path=str(p), size_bytes=p.stat().st_size, sha256=sha(p))


def read_log(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    assert [r['update'] for r in rows] == list(range(1, len(rows)+1))
    assert all(np.isfinite(r[k]) for r in rows for k in ('loss','gradient_l2','supervised','clean_retention'))
    assert all(r['gradient_l2'] > 0 and r['finite_gradient_tensors'] > 0 for r in rows)
    return rows


def trajectory(rows):
    return [{k:r[k] for k in ('update','normal_indices','auxiliary_indices','half_mask')} for r in rows]


def replay(fitter, normal, target, aux, aux_target, aux_clean, seed, branch, arm):
    rng = np.random.default_rng(71070+seed); labels = hashlib.sha256(); rows = []
    half_count = len(aux)//2
    for _ in range(1 if fitter.smoke else 6):
        order = rng.permutation(len(normal))
        if fitter.smoke: order = order[:32]
        for offset in range(0,len(order),32):
            indices = order[offset:offset+32]
            ai, half = draw_auxiliary(rng, half_count, len(indices))
            idx = torch.as_tensor(indices,device=normal.device)
            aid = torch.as_tensor(ai,device=normal.device)
            _, y = fitter.shared.affine(normal[idx],target[idx],rng)
            _, _, b = fitter.native.matched_affine(aux[aid],aux_clean[aid],aux_target[aid],rng)
            labels.update(y.detach().cpu().numpy().tobytes()); labels.update(b.detach().cpu().numpy().tobytes())
            rows.append(dict(update=len(rows)+1,normal_indices=indices.tolist(),
                             auxiliary_indices=ai.tolist(),half_mask=half.tolist()))
    return labels.hexdigest(), rows


class PriorControls:
    def __init__(self, root, diagnostic, strict=True):
        self.root = Path(root); self.diagnostic = Path(diagnostic)
        completion = self.root/'completion.json'; execution = self.root/'priority_execution_receipt.json'
        if strict:
            assert sha(completion) == STUDY_SHA and sha(execution) == RECEIPT_SHA
            assert sha(self.diagnostic) == DIAGNOSTIC_SHA
        self.completed = json.loads(completion.read_text())
        self.execution = json.loads(execution.read_text())
        assert self.completed['status'] == 'complete' and self.completed['real_data_used'] is False
        assert self.execution['status'] == 'complete_synthetic_only' and not self.execution['real_scores_computed']
        self.runs = {(r['method'],r['seed']):r for r in self.completed['results']}
        self.records = {(r['method'],r['seed'],r['arm'],r['branch']):r for r in self.execution['records']}
        if strict:
            assert len(self.runs) == 6 and len(self.records) == 36
            diag = json.loads(self.diagnostic.read_text())
            assert diag['status'] == 'complete' and diag['optimizer_updates'] == 0 and not diag['real_data_used']
            assert diag['source_completion_sha256'] == STUDY_SHA
            self.weights = {str(Path(r['path'])):r for r in diag['source_weights']}
        else: self.weights = {}
        self.strict = strict; self.audit = []

    def reuse(self, fitter, method, seed, arm, branch, scale, state, normal, target, aux, aux_target, aux_clean):
        expected = self.records[(method,seed,arm,branch)]
        assert expected['updates'] == (1 if fitter.smoke else 288)
        assert expected['teacher_unchanged'] and expected['independent_model_optimizer_rebuild'] and expected['optimizer_state_equal']
        assert expected['source_state_sha256'] == state_sha(state)
        tag = f'{method}_seed{seed}_{arm}_{branch}'
        evidence = self.root/'training_evidence'
        cp = Path(expected['checkpoint_path']); assert cp.parent == evidence
        assert sha(cp) == expected['checkpoint_sha256']
        receipt_path = evidence/(tag+'_receipt.json')
        assert json.loads(receipt_path.read_text()) == expected
        log_path = evidence/(tag+'.jsonl'); rows = read_log(log_path)
        assert len(rows) == expected['updates']
        labels, draws = replay(fitter,normal,target,aux,aux_target,aux_clean,seed,branch,arm)
        assert labels == expected['labels_sha256'] and draws == trajectory(rows)
        restored = torch.load(cp,map_location='cpu',weights_only=False)
        assert restored['update'] == expected['updates'] and restored['source_state_sha256'] == state_sha(state)
        assert state_sha(restored['model']) == expected['final_state_sha256']
        assert sorted({int(v['step']) for v in restored['optimizer']['state'].values()}) == [expected['updates']]
        model_path = self.root/(f'{method}_seed{seed}_{arm}'+('_localizer' if branch=='center' else '')+'.pt')
        output_state = torch.load(model_path,map_location='cpu',weights_only=True)
        assert state_sha(output_state) == expected['final_state_sha256']
        if self.strict:
            binding = self.weights[str(model_path)]
            assert sha(model_path) == binding['sha256'] and state_sha(output_state) == binding['state_sha256']
        # Independently rebuild the entire optimizer state, without an update.
        saved_rng = torch.get_rng_state()
        model = scale.CenterScale(); model.load_state_dict(output_state,strict=True)
        if branch == 'center': model.size.requires_grad_(False)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.00015,weight_decay=.01)
        optimizer.load_state_dict(restored['optimizer'])
        for before, after in zip(restored['optimizer']['state'].values(),optimizer.state.values()):
            assert before.keys() == after.keys() and all(torch.equal(before[k].cpu(),after[k].cpu()) for k in before)
        torch.set_rng_state(saved_rng)
        run = self.runs[(method,seed)]
        history = run['localization_train_loss' if branch=='center' else 'train_loss'][arm]
        chunk = len(rows)//len(history)
        assert len(rows)%len(history) == 0
        means = [float(np.mean([r['loss'] for r in rows[i:i+chunk]])) for i in range(0,len(rows),chunk)]
        assert means == history
        item = dict(expected,reused_immutable_control=True,new_optimizer_updates=0,
                    canonical_arm=arm, original_model=record(model_path),original_log=record(log_path),
                    original_receipt=record(receipt_path), label_and_augmentation_replay_exact=True)
        directory = fitter.out/'training_evidence'; directory.mkdir(exist_ok=True)
        fitter.shared.write(directory/(tag+'_reused_receipt.json'),item)
        fitter.records.append(item); self.audit.append(item)
        return model.to(normal.device).eval(),history

    def check_candidate(self, candidate, log_path):
        reference = self.records[(candidate['method'],candidate['seed'],'physical_supervised',candidate['branch'])]
        assert candidate['labels_sha256'] == reference['labels_sha256']
        assert candidate['source_state_sha256'] == reference['source_state_sha256']
        assert candidate['updates'] == reference['updates']
        ref_path = self.root/'training_evidence'/f"{candidate['method']}_seed{candidate['seed']}_physical_supervised_{candidate['branch']}.jsonl"
        assert trajectory(read_log(log_path)) == trajectory(read_log(ref_path))
        assert candidate['features_frozen'] is False and candidate['feature_coefficient'] > 0
        assert candidate['dark_feature_retention'] is False and candidate['retention_views']==['normal','half']
        assert candidate['actual_half_teacher_output_weight']==0 and candidate['normal_teacher_weight']==2
        candidate['matched_C1_trajectory_sha256'] = sha(ref_path)
        candidate['labels_and_draws_identical_to_C1'] = True

    def compare_control_summaries(self, result):
        old = self.runs[(result['method'],result['seed'])]
        for category in ('arms','depth_only_arms','translation_only_arms'):
            for arm in ('clean_control','physical_supervised'):
                for condition, value in result[category][arm].items():
                    a,b = value['summary'],old[category][arm][condition]['summary']
                    assert a.keys() == b.keys()
                    for key in a:
                        if isinstance(a[key],(float,int)) and not isinstance(a[key],bool):
                            assert np.isclose(a[key],b[key],rtol=1e-6,atol=1e-8), (category,arm,condition,key,a[key],b[key])
                        else: assert a[key] == b[key]
