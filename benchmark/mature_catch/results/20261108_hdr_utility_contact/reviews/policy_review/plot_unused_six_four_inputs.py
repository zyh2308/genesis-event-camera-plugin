"""Plot the completed, fixed-configuration six-case confirmation bank only."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
METHODS = ('ours', 'v2e', 'init_only', 'state_reference')
PREFIX = 'contact_windows10_unseen25_four_inputs'


def read(path):
    if not path.is_file():
        raise ValueError(f'Completed actual artifact not available: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(result_path, bank_path, development_path):
    data = read(result_path)
    bank = read(bank_path)
    protocol_path = result_path.parent/'EVALUATION_PROTOCOL.json'
    protocol = read(protocol_path)
    development = read(development_path)
    development_protocol_path = development_path.parent/'EVALUATION_PROTOCOL.json'
    development_protocol = read(development_protocol_path)
    if (data.get('status') != 'PASS' or data.get('independent_confirmation') is not True or
        data.get('all_planned_successes_and_failures_retained') is not True or
        data.get('all_reference_outcomes_retained') is not True):
        raise ValueError('Completed independent confirmation retaining all24 actor outcomes required')
    if (data.get('primary_comparison_methods') != ['ours', 'v2e'] or
        set(data.get('reference_methods', [])) != {'init_only', 'state_reference'}):
        raise ValueError('Primary comparison and input/feasibility-reference roles must be explicit')
    expected = {row['case_id']: row['seed'] for row in bank['cases']}
    if len(expected) != 6 or len(set(expected.values())) != 6:
        raise ValueError('Registered bank must have six unique cases and seeds')
    if protocol.get('independent_bank_file_sha256') != sha(bank_path):
        raise ValueError('Actual evaluation protocol does not match the registered bank SHA256')
    if protocol.get('expected_episodes') != 24:
        raise ValueError('Four inputs x same six registered cases must be planned')
    planned = protocol.get('cases', [])
    if ({row['case_id']: row['seed'] for row in planned} != expected or len(planned) != 6 or
        any(len(row['method_order']) != 4 or set(row['method_order']) != set(METHODS) for row in planned)):
        raise ValueError('Protocol case/seed/method plan differs from the registered six-case bank')
    old_seeds = {row['seed'] for row in development['rows']}
    if old_seeds & set(expected.values()):
        raise ValueError('Confirmation seeds overlap the viewed development bank')
    for field in ('camera_pitch_degrees', 'endpoint_hz', 'policy', 'contact', 'event_core',
                  'causal_flight_fit_enabled', 'observation_window_ms'):
        new_value = data['fixed_parameters'].get(field)
        old_value = development['fixed_parameters'].get(field)
        if new_value is None or new_value != old_value:
            raise ValueError(f'Configuration changed after development: {field}')
    if (data['fixed_parameters']['observation_window_ms'] != 10 or
        data['fixed_parameters']['causal_flight_fit_enabled'] is not True):
        raise ValueError('Locked shared10ms causal-flight-fit configuration required')
    checked = ('checkpoint_sha256', 'core_files', 'profile_sha256', 'v2e_checkpoint_sha256')
    for field in checked:
        if protocol.get(field) is None or protocol[field] != development_protocol.get(field):
            raise ValueError(f'Frozen policy/core/profile receipt differs: {field}')
    rows = data.get('rows', [])
    if len(rows) != 24:
        raise ValueError('All24 actual actor rows must be present before drawing')
    indexed = {method: {} for method in METHODS}
    for row in rows:
        method, case = row['method'], row['case_id']
        if method not in indexed or case not in expected or case in indexed[method]:
            raise ValueError('Unknown or duplicate actor case/method')
        if type(row.get('seed')) is not int or row['seed'] != expected[case]:
            raise ValueError('The four input groups are not matched by registered seed')
        for field in ('verified_catch', 'raw_success', 'terminated', 'final_contact'):
            if type(row.get(field)) is not bool:
                raise ValueError('Actual boolean catch/termination/contact fields required')
        n = row['hand_contact_steps']
        if type(n) is not int or n < 0:
            raise ValueError('Invalid actual contact-step count')
        if row['verified_catch'] != (row['raw_success'] and not row['terminated'] and n > 0 and row['final_contact']):
            raise ValueError('Conflicting strict catch score')
        indexed[method][case] = row
    for method, group in indexed.items():
        if set(group) != set(expected):
            raise ValueError('Incomplete group: '+method)
        scores = dict(n=6, verified_catches=sum(r['verified_catch'] for r in group.values()),
                      contact_episodes=sum(r['hand_contact_steps'] > 0 for r in group.values()))
        if any(type(data['scores'][method].get(k)) is not int or data['scores'][method][k] != v for k, v in scores.items()):
            raise ValueError('Declared counts disagree with actual episodes: '+method)
        role = data['method_roles'][method]
        event_active = method in ('ours', 'v2e')
        budget = 0 if method == 'state_reference' else 1
        if (role.get('event_frontend_active') is not event_active or
            role.get('RGB_frame_budget') != budget or role.get('control_depth_frame_budget') != budget):
            raise ValueError('Input/reference role contract differs: '+method)
    sources = {name: dict(path=str(path.resolve()), sha256=sha(path)) for name, path in
        [('confirmation', result_path), ('registered_bank', bank_path), ('actual_protocol', protocol_path),
         ('development', development_path), ('development_protocol', development_protocol_path)]}
    return data, indexed, sorted(expected), expected, sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=LOCAL/'delivery_local/contact_windows10_unseen25/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--bank', type=Path, default=LOCAL/'FIT_UNSEEN_BANK.json')
    p.add_argument('--development', type=Path, default=LOCAL/'delivery_local/contact_paired25_windows10/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--output', type=Path, default=LOCAL/'delivery_local/figures')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    try:
        data, indexed, cases, seeds, sources = validate(args.input, args.bank, args.development)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        p.exit(2, f'NOT_READY: {error}\nNo placeholder figure or CSV has been produced.\n')
    counts = {method: sum(row['verified_catch'] for row in indexed[method].values()) for method in METHODS}
    if args.check_only:
        print(json.dumps(dict(status='READY', actual_episodes=24, cases=cases, counts=counts,
                              primary_methods=['ours', 'v2e'], references=['init_only', 'state_reference'],
                              artifacts_generated=False), ensure_ascii=False, indent=2))
        return
    png, pdf, table, manifest_path, sums_path = [args.output/(PREFIX+suffix) for suffix in
        ('.png', '.pdf', '_episodes.csv', '_SOURCE_SHA256.json', '_SHA256SUMS.txt')]
    if not args.overwrite and any(path.exists() for path in (png, pdf, table, manifest_path, sums_path)):
        p.exit(2, 'Output exists; use --overwrite to regenerate only these confirmation artifacts.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.font_manager import FontProperties
    font = next((f for f in (Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')) if f.exists()), None)
    if font is None:
        p.exit(2, 'Chinese plotting font unavailable; no confirmation figure generated.\n')
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(), 'font.size': 11,
                         'axes.unicode_minus': False, 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False})
    matrix = np.array([[int(indexed[m][c]['verified_catch']) for m in METHODS] for c in cases])
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[2., 1.], hspace=.30)
    ax, bar = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['#e5b8a6', '#75b9ad']), norm=BoundaryNorm([-.5, .5, 1.5], 2))
    labels = ['Ours\n事件闭环', 'V2E\n事件闭环', '仅初始化\n无后续事件', '含噪GT\n可行性参考']
    ax.set_xticks(range(4), labels)
    ax.tick_params(axis='x', top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_yticks(range(6), [case.split('_')[0] for case in cases])
    for i, case in enumerate(cases):
        for j, method in enumerate(METHODS):
            r = indexed[method][case]
            text = '成功' if r['verified_catch'] else ('接触后失败' if r['hand_contact_steps'] else '未接触失败')
            ax.text(j, i, text, ha='center', va='center', color='#263544', fontsize=10.5)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.axvline(1.5, color='#35424f', linewidth=2.)
    values = [counts[m] for m in METHODS]
    bar.bar(range(4), values, color=['#218c87', '#c17c48', '#929baa', '#637a88'], width=.58)
    bar.set_xticks(range(4), ['Ours', 'V2E', '仅初始化', '含噪GT参考'])
    bar.set_xlim(-.5, 3.5)
    bar.set_ylim(0, 7.1)
    bar.set_yticks([0, 2, 4, 6])
    bar.set_ylabel('严格接取成功回合数')
    bar.grid(axis='y', alpha=.15)
    bar.set_axisbelow(True)
    bar.axvline(1.5, color='#35424f', linewidth=1.5)
    for i, n in enumerate(values):
        bar.text(i, n+.16, f'{n}/6', ha='center', fontsize=13)
    fig.suptitle('Genesis闭环｜锁定配置后的六条新来球', fontsize=18, y=.985)
    fig.text(.5, .93, '共同10 ms观察窗与因果轨迹拟合；原权重、冻结核心、接触物理保持固定', ha='center', fontsize=11, color='#52606c')
    note = ('每列6条同seed配对回合；Ours/V2E为主比较，另外两列分别为输入消融和含噪GT可行性参考。\n'
            '仅初始化列使用首帧RGB-D后停止事件更新；含噪GT列使用原任务状态输入。\n'
            '该批新来球与六条开发来球分开，全部成功和失败保留；各列结果不混入其他列分母。')
    fig.text(.04, .025, note, color='#52606c', fontsize=10)
    fig.subplots_adjust(left=.12, right=.96, bottom=.20, top=.80)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    fields = ['case_id', 'seed', 'method', 'role', 'verified_catch', 'raw_success', 'terminated',
              'final_contact', 'hand_contact_steps', 'first_touch_s', 'raw_events', 'output']
    with table.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for method in METHODS:
                row = {key: indexed[method][case].get(key) for key in fields}
                row['role'] = data['method_roles'][method]['role']
                writer.writerow(row)
    manifest = dict(scope='Locked-configuration six-new-case confirmation, four separately reported inputs',
        sources=sources, cases=cases, seeds=seeds, actual_actor_episodes=24, counts=counts,
        primary_methods=['ours', 'v2e'], input_ablation='init_only', feasibility_reference='state_reference',
        denominator_per_column=6, reference_trials_not_pooled=True,
        fixed_parameters=data['fixed_parameters'], all_outcomes_retained=True,
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        outputs={f.name: sha(f) for f in (png, pdf, table)})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sums_path.write_text(''.join(f'{sha(f)}  {f.name}\n' for f in (png, pdf, table, manifest_path)), encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETE_REAL_RESULTS', counts=counts,
        outputs=[str(f) for f in (png, pdf, table, manifest_path, sums_path)]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
