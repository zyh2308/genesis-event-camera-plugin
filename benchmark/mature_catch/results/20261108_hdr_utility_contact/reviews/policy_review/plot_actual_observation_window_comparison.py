"""Plot completed paired physical outcomes only; fail closed on absent/partial data.

Default inputs are the shared-flight-fit whole-packet run and the future shared
flight-fit 10 ms run. No synthetic data or pending-result placeholders are made.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

METHODS = ('ours', 'v2e')
WINDOW_KEYS = ('observation_window_ms', 'event_observation_window_ms', 'window_ms')
HERE = Path(__file__).resolve().parent
DELIVERY = HERE.parent / 'delivery_local'
PREFIX = 'contact_observation_window_comparison'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_completed(path):
    if not path.is_file():
        raise ValueError(f'Actual completed result is not available: {path}')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('status') != 'PASS' or data.get('all_planned_successes_and_failures_retained') is not True:
        raise ValueError(f'Completed audited result retaining all outcomes required: {path}')
    rows = data.get('rows')
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError(f'Expected exactly12 actual actor episodes (2methods x6cases): {path}')
    indexed = {method: {} for method in METHODS}
    for row in rows:
        method = row.get('method')
        case = row.get('case_id')
        if method not in METHODS or not isinstance(case, str) or not case:
            raise ValueError('Only explicit Ours/V2E actor episodes with case IDs are accepted')
        if case in indexed[method]:
            raise ValueError(f'Duplicate actor episode: {method}/{case}')
        if type(row.get('seed')) is not int:
            raise ValueError('Every actual episode requires an integer seed')
        for field in ('verified_catch', 'raw_success', 'terminated', 'final_contact'):
            if type(row.get(field)) is not bool:
                raise ValueError(f'Actual boolean {field} required for {method}/{case}')
        touches = row.get('hand_contact_steps')
        if type(touches) is not int or touches < 0:
            raise ValueError('Invalid measured contact-step count')
        strict = row['raw_success'] and not row['terminated'] and touches > 0 and row['final_contact']
        if strict != row['verified_catch']:
            raise ValueError(f'Inconsistent strict catch result: {method}/{case}')
        indexed[method][case] = row
    for method in METHODS:
        actual = indexed[method]
        if len(actual) != 6 or len({r['seed'] for r in actual.values()}) != 6:
            raise ValueError('Each group requires six distinct actual case/seed pairs')
        counts = dict(n=6, verified_catches=sum(r['verified_catch'] for r in actual.values()),
                      contact_episodes=sum(r['hand_contact_steps'] > 0 for r in actual.values()))
        declared = data.get('scores', {}).get(method, {})
        if any(type(declared.get(k)) is not int or declared[k] != v for k, v in counts.items()):
            raise ValueError(f'Declared scores disagree with all six actual rows: {method}')
    return data, indexed


def validate_pair(base_path, window_path):
    base, b = read_completed(base_path)
    window, w = read_completed(window_path)
    normalized = []
    for data, expected, label in ((base, 0., 'whole-packet'), (window, 10., '10ms')):
        params = dict(data.get('fixed_parameters', {}))
        declared = [params.pop(k) for k in WINDOW_KEYS if k in params]
        if not declared and expected != 0:
            raise ValueError('New result must explicitly declare fixed_parameters.observation_window_ms=10')
        values = [float(v) for v in declared] or [0.]
        if any(not math.isfinite(v) or v != expected for v in values):
            raise ValueError(f'Conflicting observation-window declaration for {label}: {declared}')
        if params.get('causal_flight_fit_enabled') is not True:
            raise ValueError('Shared causal flight fit must be enabled in both comparisons')
        if params.get('endpoint_hz') != 25:
            raise ValueError('Both actual runs must use25Hz source endpoints')
        for field in ('camera_pitch_degrees', 'policy', 'contact', 'event_core', 'init'):
            if field not in params:
                raise ValueError(f'Missing common-configuration declaration: {field}')
        normalized.append(params)
    if normalized[0] != normalized[1]:
        raise ValueError('Declared configuration differs beyond the shared observation-window setting')
    cases = sorted(b['ours'])
    seeds = {case: b['ours'][case]['seed'] for case in cases}
    for indexed in (b, w):
        for method in METHODS:
            if set(indexed[method]) != set(cases):
                raise ValueError('The four groups do not share exactly the same six cases')
            if {case: r['seed'] for case, r in indexed[method].items()} != seeds:
                raise ValueError('The four groups do not share the exact case-to-seed mapping')
    # Strengthen fixed-core/weight evidence if both source packages include their
    # actual experiment receipts. Missing optional receipts are not invented.
    receipts = []
    for path in (base_path, window_path):
        p = path.parent / 'EVALUATION_PROTOCOL.json'
        receipts.append((p, json.loads(p.read_text(encoding='utf-8-sig'))) if p.is_file() else (p, None))
    receipt_audit = dict(available=[item[1] is not None for item in receipts], checked_fields=[])
    if all(item[1] is not None for item in receipts):
        for field in ('checkpoint_sha256', 'core_files', 'profile_sha256', 'v2e_checkpoint_sha256'):
            left, right = receipts[0][1].get(field), receipts[1][1].get(field)
            if left is None or right is None or left != right:
                raise ValueError(f'Actual frozen-checkpoint/core receipt mismatch or absence: {field}')
            receipt_audit['checked_fields'].append(field)
    receipt_audit['sources'] = [dict(path=str(p.resolve()), sha256=sha(p)) for p, data in receipts if data is not None]
    return base, b, window, w, cases, seeds, normalized[0], receipt_audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, default=DELIVERY/'contact_paired25_flightfit/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--windows', type=Path, default=DELIVERY/'contact_paired25_windows10/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--output', type=Path, default=DELIVERY/'figures')
    p.add_argument('--check-only', action='store_true', help='Validate real completed sources, produce no artifacts')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    try:
        base, b, window, w, cases, seeds, fixed, receipt_audit = validate_pair(a.base, a.windows)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        p.exit(2, f'NOT_READY: {error}\nNo figure or CSV has been generated.\n')
    sequences = [('whole_packet', 'ours', b['ours']), ('whole_packet', 'v2e', b['v2e']),
                 ('window10ms', 'ours', w['ours']), ('window10ms', 'v2e', w['v2e'])]
    counts = [sum(r['verified_catch'] for r in indexed.values()) for _, _, indexed in sequences]
    if a.check_only:
        print(json.dumps(dict(status='READY', actual_episodes=24, paired_cases=cases, counts=counts,
                              source_receipts=receipt_audit, artifacts_generated=False), ensure_ascii=False, indent=2))
        return
    targets = [a.output/(PREFIX+suffix) for suffix in ('.png', '.pdf', '_episodes.csv', '_SOURCE_SHA256.json', '_SHA256SUMS.txt')]
    if not a.overwrite and any(path.exists() for path in targets):
        p.exit(2, 'Output exists. Explicit --overwrite is required to regenerate these artifacts.\n')

    # Matplotlib is imported only after all four groups and fixed parameters pass.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.font_manager import FontProperties
    font = next((path for path in (Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')) if path.exists()), None)
    if font is None:
        p.exit(2, 'Chinese plotting font is unavailable; no figure generated.\n')
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(),
        'axes.unicode_minus': False, 'font.size': 11, 'pdf.fonttype': 42,
        'axes.spines.top': False, 'axes.spines.right': False})
    a.output.mkdir(parents=True, exist_ok=True)
    png, pdf, csv_path, manifest_path, sums_path = targets
    matrix = np.array([[int(rows[case]['verified_catch']) for case in cases] for _, _, rows in sequences])
    fig = plt.figure(figsize=(13, 5.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.15], wspace=.13)
    ax, bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['#e5b8a6', '#75b9ad']), norm=BoundaryNorm([-.5, .5, 1.5], 2))
    labels = ['整包观察 · Ours', '整包观察 · V2E', '10 ms短窗 · Ours', '10 ms短窗 · V2E']
    ax.set_yticks(range(4), labels)
    short = [case.split('_')[0] for case in cases]
    ax.set_xticks(range(6), short if len(set(short)) == 6 else cases)
    for i, (_, _, indexed) in enumerate(sequences):
        for j, case in enumerate(cases):
            row = indexed[case]
            outcome = '成功' if row['verified_catch'] else ('接触后失败' if row['hand_contact_steps'] else '未接触失败')
            ax.text(j, i, outcome, ha='center', va='center', fontsize=10.5, color='#263544')
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.axhline(1.5, color='#35424f', linewidth=1.6)
    ax.set_title('同一批六条开发来球｜全部成功与失败保留', pad=13)
    bar.barh(range(4), counts, color=['#218c87', '#c17c48', '#218c87', '#c17c48'], height=.57)
    bar.set_ylim(3.5, -.5)
    bar.set_yticks([])
    bar.set_xlim(0, 7.2)
    bar.set_xticks([0, 2, 4, 6])
    bar.axhline(1.5, color='#35424f', linewidth=1.6)
    bar.grid(axis='x', alpha=.15)
    bar.set_axisbelow(True)
    for i, count in enumerate(counts):
        bar.text(count+.12, i, f'{count}/6', va='center', fontsize=13)
    bar.set_title('严格成功 / 实测回合', pad=13)
    bar.set_xlabel('严格接取成功回合数')
    fig.suptitle('Genesis事件闭环｜整包观察与10 ms短窗', fontsize=18, y=.985)
    fig.text(.5, .91, '两种事件源共同启用因果轨迹拟合；25 Hz源图像与原控制频率不变', ha='center', fontsize=11, color='#52606c')
    note = ('原策略权重、冻结事件核心、7°机位、共同接触物理与成功判据固定；10 ms仅为已到达包内观察窗。\n'
            f'六条已查看的开发来球，非独立测试；Ours {counts[0]}/6 → {counts[2]}/6，V2E {counts[1]}/6 → {counts[3]}/6。')
    fig.text(.025, .025, note, fontsize=10, color='#52606c')
    fig.subplots_adjust(left=.19, right=.98, bottom=.20, top=.81)
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['observer', 'method', 'case_id', 'seed', 'verified_catch', 'raw_success',
                         'terminated', 'final_contact', 'hand_contact_steps', 'first_touch_s',
                         'raw_events', 'result_directory', 'source_summary_sha256'])
        for stage, method, indexed in sequences:
            source_sha = sha(a.base if stage == 'whole_packet' else a.windows)
            for case in cases:
                row = indexed[case]
                writer.writerow([stage, method, case, row['seed'], row['verified_catch'], row['raw_success'],
                                 row['terminated'], row['final_contact'], row['hand_contact_steps'],
                                 row.get('first_touch_s'), row.get('raw_events'), row.get('output'), source_sha])
    manifest = dict(
        scope='24 actual actor rollouts: six matched, already-observed development cases x2event sources x2common observers; not independent testing',
        sources={label: dict(path=str(path.resolve()), sha256=sha(path)) for label, path in [('whole_packet', a.base), ('window10ms', a.windows)]},
        actual_actor_episodes=24, cases=cases, seeds=seeds, fixed_parameters=fixed,
        sole_intended_common_frontend_change='observation_window_ms:0(default whole-packet detector) ->10',
        shared_flight_fit_enabled=True, no_GT_or_initialization_only_episodes_pooled=True,
        all_successes_and_failures_retained=True, declared_physics_and_score_fixed=True,
        source_receipt_check=receipt_audit,
        outcomes=[dict(observer=stage, method=method, episodes=6, verified_catches=count)
                  for (stage, method, _), count in zip(sequences, counts)],
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        outputs={path.name: sha(path) for path in (png, pdf, csv_path)},
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sums_path.write_text(''.join(f'{sha(path)}  {path.name}\n' for path in (png, pdf, csv_path, manifest_path)), encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETE_REAL_RESULTS', counts=counts,
                          artifacts=[str(path) for path in targets]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
