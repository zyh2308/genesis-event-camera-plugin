"""Paired source-cadence comparison on the same six existing seed blocks.

Only the original Ours/V2E actor rows enter this figure. Reused initialization-only
and noisy-GT references are never counted as new episodes or pooled denominators.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True

from plot_actual_observation_window_comparison import read_completed
from plot_unused_six_four_inputs import validate as validate_confirmed25

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent
METHODS = ('ours', 'v2e')
PREFIX = 'contact_cadence25_vs12p5'


def read(path):
    if not path.is_file():
        raise ValueError(f'Actual completed artifact is not available: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_plan(protocol, expected_rate):
    result = {}
    cases = protocol.get('cases', [])
    if len(cases) != 6:
        raise ValueError('The actual cadence protocol must contain exactly six cases')
    for case in cases:
        name = case['case_id']
        if name in result or type(case['seed']) is not int or case['endpoint_hz'] != expected_rate:
            raise ValueError('Protocol duplicate case/seed or wrong actual source cadence')
        launch = case['launch']
        if set(launch) != {'position', 'velocity'}:
            raise ValueError('Explicit complete launch position/velocity contract required')
        if any(len(launch[k]) != 3 or any(not math.isfinite(float(x)) for x in launch[k]) for k in launch):
            raise ValueError('Invalid actual launch vector')
        if set(case['method_order']) & set(METHODS) != set(METHODS):
            raise ValueError('Both primary methods must be planned for each case')
        if expected_rate == 12.5 and (len(case['method_order']) != 2 or set(case['method_order']) != set(METHODS)):
            raise ValueError('New12.5Hz plan must contain only the12 primary actor episodes')
        result[name] = dict(seed=case['seed'], launch=launch)
    if len({x['seed'] for x in result.values()}) != 6:
        raise ValueError('Six distinct seed blocks required')
    return result


def validate_pair(args):
    # Also validates all24 already-completed25Hz actors and their reference roles,
    # without pooling those references into this cadence comparison.
    old, old_indexed, cases, seeds, old_sources = validate_confirmed25(
        args.hz25, args.bank, args.development)
    new, new_indexed = read_completed(args.hz12p5)
    p25_path = args.hz25.parent/'EVALUATION_PROTOCOL.json'
    p12_path = args.hz12p5.parent/'EVALUATION_PROTOCOL.json'
    p25, p12 = read(p25_path), read(p12_path)
    # The25Hz protocol retains its parent template's literal
    # 'case-specific100/50/25' overview. Individual actual case rates are the
    # authoritative scalar contract and are all checked below; never rewrite
    # this legacy overview in the input protocol just to make a figure.
    if p25.get('endpoint_hz') not in (25, 'case-specific100/50/25') or p12.get('endpoint_hz') != 12.5:
        raise ValueError('Actual cadence protocol overview conflicts with the case-specific rate contract')
    if p25.get('expected_episodes') != 24 or p12.get('expected_episodes') != 12:
        raise ValueError('Expected old24includingreferences and new12primary actors only')
    plans25, plans12 = paired_plan(p25, 25), paired_plan(p12, 12.5)
    if plans25 != plans12:
        raise ValueError('Source-cadence runs differ in case ID, seed or exact launch position/velocity')
    for method in METHODS:
        if (set(new_indexed[method]) != set(cases) or
            {case: r['seed'] for case, r in new_indexed[method].items()} != seeds):
            raise ValueError('The four cadence/method groups are not exactly case/seed paired')
    # Labels and values are checked, not silently corrected by the plotting code.
    for data, rate in ((old, 25), (new, 12.5)):
        fixed = data['fixed_parameters']
        if fixed.get('endpoint_hz') != rate:
            raise ValueError(f'Result summary cadence metadata is not the actual{rate}Hz protocol')
        if fixed.get('observation_window_ms') != 10 or fixed.get('causal_flight_fit_enabled') is not True:
            raise ValueError('Shared10ms/flight-fit setup must stay fixed')
    for field in ('camera_pitch_degrees', 'policy', 'contact', 'event_core',
                  'causal_flight_fit_enabled', 'observation_window_ms'):
        if old['fixed_parameters'][field] != new['fixed_parameters'].get(field):
            raise ValueError('A common task/frontend setting changed: '+field)
    fixed_protocol_fields = ('control_hz', 'physics_step_s', 'simulated_packet_delay_s',
        'episode_timeout_s', 'camera_pitch_relative_degrees', 'contact_profile',
        'shared_estimator_overrides', 'runtime_files', 'checkpoint_sha256', 'core_files',
        'profile_sha256', 'v2e_checkpoint_sha256', 'v2e_source_files')
    for field in fixed_protocol_fields:
        if p25.get(field) is None or p25[field] != p12.get(field):
            raise ValueError('Fixed source/policy/core/physics/timing contract changed: '+field)
    if p25['control_hz'] != 25 or p25['shared_estimator_overrides'].get('CATCHIT_EVENT_OBSERVATION_WINDOW_MS') != '10':
        raise ValueError('Expected25Hz control and common10ms observer')
    sources = dict(old_sources)
    sources.update(cadence12p5=dict(path=str(args.hz12p5.resolve()), sha256=sha(args.hz12p5)),
                   cadence12p5_protocol=dict(path=str(p12_path.resolve()), sha256=sha(p12_path)))
    return old, old_indexed, new, new_indexed, cases, seeds, plans25, sources, fixed_protocol_fields


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hz25', type=Path, default=LOCAL/'delivery_local/contact_windows10_unseen25/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--hz12p5', type=Path, default=LOCAL/'delivery_local/contact_windows10_cadence12p5/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--bank', type=Path, default=LOCAL/'FIT_UNSEEN_BANK.json')
    p.add_argument('--development', type=Path, default=LOCAL/'delivery_local/contact_paired25_windows10/CONTACT_PAIRED_COMPARISON.json')
    p.add_argument('--output', type=Path, default=LOCAL/'delivery_local/figures')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    try:
        old, old_i, new, new_i, cases, seeds, launches, sources, checked_fields = validate_pair(args)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        p.exit(2, f'NOT_READY: {error}\nNo cadence figure or placeholder CSV has been generated.\n')
    sequences = [(25, 'ours', old_i['ours']), (25, 'v2e', old_i['v2e']),
                 (12.5, 'ours', new_i['ours']), (12.5, 'v2e', new_i['v2e'])]
    counts = [sum(row['verified_catch'] for row in indexed.values()) for _, _, indexed in sequences]
    if args.check_only:
        print(json.dumps(dict(status='READY', paired_seed_blocks=6, plotted_primary_actor_episodes=24,
            new_actor_episodes=12, counts=counts, exact_launch_case_seed_pairing=True,
            reused_reference_episodes_not_pooled=True, artifacts_generated=False), ensure_ascii=False, indent=2))
        return
    png, pdf, csv_path, manifest_path, sums_path = [args.output/(PREFIX+s) for s in
        ('.png', '.pdf', '_episodes.csv', '_SOURCE_SHA256.json', '_SHA256SUMS.txt')]
    if not args.overwrite and any(f.exists() for f in (png, pdf, csv_path, manifest_path, sums_path)):
        p.exit(2, 'Cadence output exists; --overwrite required to regenerate these files.\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.font_manager import FontProperties
    font = next((f for f in (Path('C:/Windows/Fonts/msyh.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')) if f.exists()), None)
    if font is None:
        p.exit(2, 'Chinese plotting font missing; no cadence figure generated.\n')
    plt.rcParams.update({'font.family': FontProperties(fname=str(font)).get_name(), 'font.size': 11,
        'axes.unicode_minus': False, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(13, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.3, 1.15], wspace=.13)
    ax, bar = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    matrix = np.array([[int(by_case[c]['verified_catch']) for c in cases] for _, _, by_case in sequences])
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['#e5b8a6', '#75b9ad']), norm=BoundaryNorm([-.5, .5, 1.5], 2))
    ax.set_yticks(range(4), ['25 Hz · Ours', '25 Hz · V2E', '12.5 Hz · Ours', '12.5 Hz · V2E'])
    ax.set_xticks(range(6), [c.split('_')[0] for c in cases])
    for i, (_, _, group) in enumerate(sequences):
        for j, case in enumerate(cases):
            row = group[case]
            label = '成功' if row['verified_catch'] else ('接触后失败' if row['hand_contact_steps'] else '未接触失败')
            ax.text(j, i, label, ha='center', va='center', fontsize=10.5, color='#263544')
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=3)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.axhline(1.5, color='#35424f', linewidth=1.6)
    ax.set_title('同六条来球重复比较｜case、seed、发射配置严格配对', pad=13)
    bar.barh(range(4), counts, height=.57, color=['#218c87', '#c17c48', '#218c87', '#c17c48'])
    bar.set_ylim(3.5, -.5)
    bar.set_yticks([])
    bar.set_xlim(0, 7.2)
    bar.set_xticks([0, 2, 4, 6])
    bar.axhline(1.5, color='#35424f', linewidth=1.6)
    bar.grid(axis='x', alpha=.15)
    bar.set_axisbelow(True)
    for i, n in enumerate(counts):
        bar.text(n+.12, i, f'{n}/6', va='center', fontsize=13)
    bar.set_title('严格成功 / 实测回合', pad=13)
    bar.set_xlabel('严格接取成功回合数')
    fig.suptitle('Genesis事件闭环｜25 Hz与12.5 Hz稀疏输入', fontsize=18, y=.985)
    fig.text(.5, .91, '源图像间隔40 → 80 ms；控制25 Hz、共同10 ms观察窗与50 ms确认保持不变', ha='center', fontsize=11, color='#52606c')
    note = ('原权重、冻结事件核心、共同接触物理和来球固定；只改变源图像采集频率。\n'
            '四行各6条，使用同六个种子块重复评估；初始化/含噪GT参考复用，不重跑或计入本图分母。\n'
            '12.5 Hz组新增12条主方法回合，没有新增六个独立样本。')
    fig.text(.025, .022, note, fontsize=10, color='#52606c')
    fig.subplots_adjust(left=.19, right=.98, bottom=.23, top=.81)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    fields = ['source_hz', 'source_interval_ms', 'method', 'case_id', 'seed', 'verified_catch',
              'raw_success', 'terminated', 'final_contact', 'hand_contact_steps', 'first_touch_s',
              'raw_events', 'output', 'launch_position_json', 'launch_velocity_json']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for hz, method, group in sequences:
            for case in cases:
                row = {key: group[case].get(key) for key in fields}
                row.update(source_hz=hz, source_interval_ms=1000/hz,
                    launch_position_json=json.dumps(launches[case]['launch']['position']),
                    launch_velocity_json=json.dumps(launches[case]['launch']['velocity']))
                writer.writerow(row)
    helpers = [HERE/'plot_unused_six_four_inputs.py', HERE/'plot_actual_observation_window_comparison.py']
    manifest = dict(scope='Paired cadence stress comparison on six previously evaluated seed blocks; not six new independent cases',
        sources=sources, paired_cases=cases, paired_seeds=seeds, exact_paired_launches=launches,
        plotted_primary_actor_episodes=24, new_actor_episodes=12, paired_seed_blocks=6,
        primary_methods=list(METHODS), reused_reference_trials_not_pooled=True,
        only_changed_setting='source image cadence25Hz->12.5Hz', checked_fixed_protocol_fields=list(checked_fields),
        common_frontend='10ms within-parent event windows; two-observation50ms confirmation; shared causal flight fit',
        output_control_hz=25, all_successes_and_failures_retained=True,
        outcomes=[dict(endpoint_hz=hz, method=method, n=6, verified_catches=count) for (hz, method, _), count in zip(sequences, counts)],
        script=dict(path=str(Path(__file__).resolve()), sha256=sha(Path(__file__))),
        validation_helpers={f.name: sha(f) for f in helpers}, outputs={f.name: sha(f) for f in (png, pdf, csv_path)})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    sums_path.write_text(''.join(f'{sha(f)}  {f.name}\n' for f in (png, pdf, csv_path, manifest_path)), encoding='utf-8')
    print(json.dumps(dict(status='GENERATED_FROM_COMPLETE_ACTUAL_CADENCE_RESULTS', counts=counts,
        outputs=[str(f) for f in (png, pdf, csv_path, manifest_path, sums_path)]), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
