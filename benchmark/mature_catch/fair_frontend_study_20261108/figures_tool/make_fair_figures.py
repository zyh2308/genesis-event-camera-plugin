"""Validated standalone figures/statistics for the paired robot utility study.

Official plots require the collector's completion receipt and exact CSV/JSON
agreement. Missing, partial or incompatible data never produces placeholder
figures. Episode wall time is deliberately absent from the cost calculation.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
import argparse
import csv
import hashlib
import json
import math
import re
import sys

sys.dont_write_bytecode = True

METHODS = ('ours', 'v2e')
REFERENCES = ('init_only', 'state_reference')
VALIDATION_RATES = (12.5, 25.)
BLIND_RATES = (12.5, 25., 50., 100.)
DEFAULT_FRONTENDS = ('w0_fit1_gap100', 'w10_fit0_gap100', 'w10_fit1_gap50')
ENV_KEYS = ('CATCHIT_EVENT_OBSERVATION_WINDOW_MS', 'CATCHIT_EVENT_FLIGHT_FIT', 'CATCHIT_EVENT_CONFIRMATION_MAX_GAP_MS')
EPISODE_FIELDS = ('physical_case_id', 'seed', 'endpoint_hz', 'method', 'frontend_id',
    'verified_catch', 'raw_success', 'terminated', 'hand_contact_steps', 'final_contact',
    'initialization_valid', 'sim_time_s', 'wall_s', 'output')
COST_FIELDS = ('method', 'source_hz', 'frontend_id', 'repeat_id', 'simulated_s',
    'measured_wall_s', 'measurement_start_s', 'measurement_end_s', 'cost_scope',
    'complete_window', 'native_only', 'audit_included', 'primary_validated', 'instrumented',
    'reference_trace_sha256', 'source_snapshot_sha256', 'policy_sha256', 'host', 'gpu_uuid')
COST_SCOPE = 'native_capture+generator+frontend+actor+common_physics'
COLORS = dict(ours='#218d88', v2e='#c68049', init_only='#929eae', state_reference='#627e8b')
NAMES = dict(ours='Ours', v2e='V2E', init_only='RGB-D initialization only', state_reference='Noisy-state reference')


class DataError(ValueError):
    pass


def require(condition, message):
    if not condition: raise DataError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    require(Path(path).is_file(), f'Missing required artifact: {path}')
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def boolean(value, name, optional=False):
    if value is None or str(value).strip().lower() in ('', 'null', 'none'):
        if optional: return None
        raise DataError(f'Missing boolean {name}')
    if isinstance(value, bool): return value
    text = str(value).strip().lower()
    require(text in ('true', 'false', '1', '0'), f'Invalid boolean {name}: {value!r}')
    return text in ('true', '1')


def number(value, name, optional=False):
    if value is None or str(value).strip().lower() in ('', 'null', 'none'):
        if optional: return None
        raise DataError(f'Missing numeric value {name}')
    try: result = float(value)
    except (ValueError, TypeError) as error: raise DataError(f'Invalid number {name}: {value!r}') from error
    require(math.isfinite(result), f'Non-finite number {name}')
    return result


def integer(value, name):
    result = number(value, name)
    require(result.is_integer(), f'Non-integral {name}: {value!r}')
    return int(result)


def csv_rows(path, fields):
    require(Path(path).is_file(), f'Missing CSV: {path}')
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and set(fields) <= set(reader.fieldnames),
                f'CSV schema mismatch: {path}; required fields {fields}')
        rows = list(reader)
    require(bool(rows), f'CSV contains no completed observations: {path}')
    return rows


def episode(row):
    method = str(row['method']); require(method in METHODS+REFERENCES, f'Unknown method: {method}')
    physical = str(row['physical_case_id']).strip(); require(bool(physical), 'Empty physical_case_id')
    result = dict(physical_case_id=physical, seed=integer(row['seed'], 'seed'), method=method,
        endpoint_hz=number(row['endpoint_hz'], 'endpoint_hz', optional=True),
        frontend_id=str(row.get('frontend_id') or '').strip(),
        verified_catch=boolean(row['verified_catch'], 'verified_catch'),
        raw_success=boolean(row['raw_success'], 'raw_success'),
        terminated=boolean(row['terminated'], 'terminated'),
        hand_contact_steps=integer(row['hand_contact_steps'], 'hand_contact_steps'),
        final_contact=boolean(row['final_contact'], 'final_contact'),
        initialization_valid=boolean(row.get('initialization_valid'), 'initialization_valid', optional=method=='state_reference'),
        sim_time_s=number(row['sim_time_s'], 'sim_time_s'), wall_s=number(row['wall_s'], 'wall_s'),
        output=str(row['output']))
    require(result['hand_contact_steps'] >= 0 and result['sim_time_s'] >= 0 and result['wall_s'] >= 0,
            'Negative episode count/time')
    strict = result['raw_success'] and not result['terminated'] and result['hand_contact_steps'] > 0 and result['final_contact']
    require(strict == result['verified_catch'], f'Catch score inconsistent with raw flags: {physical}/{method}')
    if method in METHODS:
        require(result['endpoint_hz'] is not None and result['endpoint_hz'] > 0, 'Event method requires its real source rate')
        require(re.fullmatch(r'w(?:0|10)_fit[01]_gap(?:50|100)', result['frontend_id']) is not None,
                f'Unexpected frontend ID: {result["frontend_id"]}')
    else:
        require(result['endpoint_hz'] is None, 'References must appear once per block with endpoint_hz=null, never repeated across rates')
    if method != 'state_reference' and not result['initialization_valid']:
        require(not result['verified_catch'], 'Initialization failure cannot become a successful catch')
    return result


def key(row):
    return (row['physical_case_id'], row['seed'], row['endpoint_hz'], row['method'], row['frontend_id'])


def block(row):
    return row['physical_case_id'], row['seed']


def load_stage(folder, expected_stage):
    folder = Path(folder); marker = read_json(folder/'FAIR_COLLECTION_COMPLETE.json')
    require(marker.get('status') == 'PASS' and marker.get('all_actual_jobs_normal_exit') is True
            and marker.get('all_task_failures_retained') is True, 'Collector stage is incomplete or contains unresolved infrastructure failures')
    data_path = folder/'FAIR_STAGE_RESULTS.json'; data = read_json(data_path)
    require(marker.get('results_sha256') == sha(data_path), 'Collector results hash does not match completion receipt')
    require(data.get('metadata', {}).get('stage') == expected_stage, f'Expected {expected_stage} collector stage')
    rows = [episode(row) for row in csv_rows(folder/'FAIR_EPISODES.csv', EPISODE_FIELDS)]
    original = [episode(row) for row in data['rows']]
    require(len(rows) == len(original) == integer(marker['actual_episodes'], 'actual_episodes'), 'CSV/JSON/receipt episode counts differ')
    require(len(rows) == integer(data['metadata']['actual_planned_actors'], 'actual_planned_actors'), 'Actual planned actors are missing')
    left, right = {key(r): r for r in rows}, {key(r): r for r in original}
    require(len(left) == len(rows) and len(right) == len(original), 'Duplicate physical block/rate/frontend/method actor')
    require(left == right, 'Collector CSV and JSON disagree')
    return dict(folder=folder, rows=rows, marker=marker, metadata=data['metadata'],
        inputs=[folder/'FAIR_COLLECTION_COMPLETE.json', folder/'FAIR_EPISODES.csv', data_path])


def counts(rows):
    n = len(rows); require(n > 0, 'Cannot calculate a success interval without completed episodes')
    k = sum(r['verified_catch'] for r in rows); lo, hi = wilson(k, n)
    return dict(successes=k, n=n, success_rate=k/n, wilson95_low=lo, wilson95_high=hi)


def wilson(successes, n, confidence=.95):
    require(n > 0 and 0 <= successes <= n, 'Invalid binomial count for Wilson interval')
    z = NormalDist().inv_cdf(.5+confidence/2); p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0., center-half), min(1., center+half)


def exact_mcnemar(ours_only, v2e_only):
    require(ours_only >= 0 and v2e_only >= 0, 'Negative discordant-pair count')
    discordant = ours_only+v2e_only
    if discordant == 0: return 1.
    return min(1., 2*sum(math.comb(discordant, j) for j in range(min(ours_only, v2e_only)+1))/2**discordant)


def holm(values):
    order = sorted(range(len(values)), key=lambda i: values[i]); result = [None]*len(values); previous = 0.
    for position, index in enumerate(order):
        previous = min(1., max(previous, (len(values)-position)*values[index])); result[index] = previous
    return result


def validate_crossover(stage, frontends, expected_blocks):
    rows = stage['rows']; require(all(r['method'] in METHODS for r in rows), 'Validation crossover unexpectedly contains reference actors')
    blocks = sorted({block(r) for r in rows})
    require(len(blocks) == expected_blocks and len({seed for _, seed in blocks}) == expected_blocks,
            f'Expected {expected_blocks} distinct development seed blocks')
    expected = {(b, rate, method, frontend) for b in blocks for rate in VALIDATION_RATES for method in METHODS for frontend in frontends}
    actual = {(block(r), r['endpoint_hz'], r['method'], r['frontend_id']) for r in rows}
    require(actual == expected and len(rows) == len(expected), 'Frontend crossover is not the complete symmetric block/rate/source/frontend Cartesian product')
    summaries = [dict(endpoint_hz=rate, method=method, frontend_id=frontend,
        **counts([r for r in rows if r['endpoint_hz']==rate and r['method']==method and r['frontend_id']==frontend]))
        for rate in VALIDATION_RATES for frontend in frontends for method in METHODS]
    return dict(blocks=blocks, frontends=frontends, summaries=summaries)


def selected_frontend(value):
    if isinstance(value, str): result = value
    else:
        require(isinstance(value, dict), 'Invalid locked frontend declaration')
        declared = value.get('frontend_id', value.get('candidate_id'))
        env = value.get('env', value)
        if all(k in env for k in ENV_KEYS):
            result = f'w{integer(env[ENV_KEYS[0]], "window")}_fit{integer(env[ENV_KEYS[1]], "fit")}_gap{integer(env[ENV_KEYS[2]], "gap")}'
            require(declared is None or declared == result, 'Frontend ID and environment disagree')
        else:
            require(declared is not None, 'Locked selection has neither frontend ID nor complete settings'); result = declared
    require(re.fullmatch(r'w(?:0|10)_fit[01]_gap(?:50|100)', result) is not None, 'Unregistered locked frontend')
    return result


def validate_blind(stage, selection_path, expected_blocks, rates):
    selection = read_json(selection_path)
    selected = {method: selected_frontend(selection['frontend_by_method'][method]) for method in METHODS}
    rows = stage['rows']; blocks = sorted({block(r) for r in rows})
    require(len(blocks) == expected_blocks and len({seed for _, seed in blocks}) == expected_blocks,
            f'Expected {expected_blocks} distinct paired blind seed blocks, not pooled source-rate rows')
    indexed = {}
    for row in rows:
        logical = block(row), row['endpoint_hz'], row['method']
        require(logical not in indexed, 'Multiple selected frontends or duplicate actors inside one blind cell')
        indexed[logical] = row
        if row['method'] in METHODS:
            require(row['frontend_id'] == selected[row['method']], 'Blind frontend changes across conditions or differs from locked development selection')
    expected = {(b, rate, method) for b in blocks for rate in rates for method in METHODS}
    expected |= {(b, None, method) for b in blocks for method in REFERENCES}
    require(set(indexed) == expected, 'Blind data is incomplete or duplicates references across rates')
    summaries = [dict(endpoint_hz=rate, method=method, frontend_id=selected[method],
        **counts([indexed[b, rate, method] for b in blocks])) for rate in rates for method in METHODS]
    references = [dict(endpoint_hz=None, method=method, frontend_id='reference',
        **counts([indexed[b, None, method] for b in blocks])) for method in REFERENCES]
    paired = []
    for rate in rates:
        cells = dict(both_success=0, ours_only=0, v2e_only=0, both_fail=0)
        for b in blocks:
            a, v = indexed[b, rate, 'ours']['verified_catch'], indexed[b, rate, 'v2e']['verified_catch']
            cells['both_success' if a and v else 'ours_only' if a else 'v2e_only' if v else 'both_fail'] += 1
        paired.append(dict(endpoint_hz=rate, paired_blocks=len(blocks), **cells,
            ours_catches=cells['both_success']+cells['ours_only'], v2e_catches=cells['both_success']+cells['v2e_only'],
            success_rate_difference=(cells['ours_only']-cells['v2e_only'])/len(blocks),
            mcnemar_exact_two_sided=exact_mcnemar(cells['ours_only'], cells['v2e_only'])))
    for row, adjusted in zip(paired, holm([r['mcnemar_exact_two_sided'] for r in paired])):
        row['mcnemar_holm_across_rates'] = adjusted
    return dict(blocks=blocks, rates=rates, indexed=indexed, selected=selected,
        summaries=summaries, references=references, paired=paired, selection_path=Path(selection_path))


def block_bootstrap(blind, repeats, seed):
    import numpy as np
    require(repeats >= 1000, 'Use at least 1000 block bootstrap draws or disable it')
    matrix = np.asarray([[int(blind['indexed'][b, rate, 'ours']['verified_catch'])-
                          int(blind['indexed'][b, rate, 'v2e']['verified_catch'])
                          for rate in blind['rates']] for b in blind['blocks']], dtype=float)
    generator = np.random.default_rng(seed); sampled = generator.integers(0, len(matrix), size=(repeats, len(matrix)))
    means = matrix[sampled].mean(axis=1); overall = means.mean(axis=1)
    return dict(replicates=repeats, rng_seed=seed, resampling_unit='physical seed block; all source-rate outcomes kept together',
        independent_blocks=len(matrix), never_pool_rates_as_independent=True,
        per_rate=[dict(endpoint_hz=rate, mean_difference=float(matrix[:, i].mean()),
            percentile95_low=float(np.quantile(means[:, i], .025)), percentile95_high=float(np.quantile(means[:, i], .975)))
            for i, rate in enumerate(blind['rates'])],
        equally_weighted_rate_mean_difference=dict(mean=float(matrix.mean()),
            percentile95_low=float(np.quantile(overall, .025)), percentile95_high=float(np.quantile(overall, .975))))


def validate_cost(path, blind, gpu_uuid=None, minimum_repeats=3):
    require(minimum_repeats >= 3, 'The registered cost curve requires at least three repeats per method/rate cell')
    raw = csv_rows(path, COST_FIELDS); accepted = []; excluded = []
    for line, row in enumerate(raw, start=2):
        reason = None; method = row['method']; rate = number(row['source_hz'], 'cost source_hz')
        if method not in METHODS: reason = 'not_an_event_method'
        elif rate not in blind['rates']: reason = 'outside_registered_rates'
        elif row['frontend_id'] != blind['selected'][method]: reason = 'not_the_locked_frontend'
        elif gpu_uuid and row['gpu_uuid'] != gpu_uuid: reason = 'outside_explicit_hardware_subset'
        elif not boolean(row['native_only'], 'native_only') or not boolean(row['primary_validated'], 'primary_validated'):
            reason = 'not_a_validated_native_measurement'
        elif boolean(row['audit_included'], 'audit_included') or boolean(row['instrumented'], 'instrumented'):
            reason = 'audit_or_instrumentation_workload'
        if reason:
            excluded.append(dict(csv_line=line, method=method, source_hz=rate, reason=reason)); continue
        require(boolean(row['complete_window'], 'complete_window'), 'A selected native cost run ended before its full fixed window')
        simulated = number(row['simulated_s'], 'simulated_s'); wall = number(row['measured_wall_s'], 'measured_wall_s')
        start = number(row['measurement_start_s'], 'measurement_start_s'); end = number(row['measurement_end_s'], 'measurement_end_s')
        require(abs(start-.32) <= 1e-8 and abs(end-1.12) <= 1e-8 and abs(simulated-.8) <= 1e-8,
                'Cost runs do not cover the registered identical0.32..1.12s window')
        require(wall > 0 and row['cost_scope'] == COST_SCOPE, 'Invalid native end-to-end cost scope/time')
        accepted.append(dict(method=method, source_hz=rate, frontend_id=row['frontend_id'], repeat_id=row['repeat_id'],
            wall_s_per_sim_s=wall/simulated, measured_wall_s=wall, simulated_s=simulated,
            host=row['host'], gpu_uuid=row['gpu_uuid'], reference_trace_sha256=row['reference_trace_sha256'],
            source_snapshot_sha256=row['source_snapshot_sha256'], policy_sha256=row['policy_sha256']))
    require(bool(accepted), 'No complete validated native fixed-window cost measurements')
    for name in ['reference_trace_sha256', 'source_snapshot_sha256', 'policy_sha256']:
        values = {r[name] for r in accepted}
        require(len(values) == 1 and re.fullmatch(r'[a-fA-F0-9]{64}', next(iter(values))) is not None,
                f'Cost workload/source is inconsistent: {name}')
    hardware = {(r['host'], r['gpu_uuid']) for r in accepted}
    require(len(hardware) == 1, 'Do not mix hardware in the cost curve; explicitly select one common GPU subset')
    groups = defaultdict(list)
    for row in accepted: groups[row['method'], row['source_hz']].append(row)
    require(set(groups) == {(m, rate) for m in METHODS for rate in blind['rates']}, 'Cost curves require every method/rate cell')
    summaries = []
    for method in METHODS:
        for rate in blind['rates']:
            group = groups[method, rate]; ids = [r['repeat_id'] for r in group]
            require(len(ids) == len(set(ids)) and len(group) >= minimum_repeats, 'Missing/duplicate fixed-window cost repeats')
            values = [r['wall_s_per_sim_s'] for r in group]
            summaries.append(dict(method=method, source_hz=rate, frontend_id=blind['selected'][method], repeats=len(values),
                mean_wall_s_per_sim_s=sum(values)/len(values), minimum=min(values), maximum=max(values),
                cost_scope=COST_SCOPE, fixed_simulated_s=.8, episode_wall_time_used=False))
    return dict(summaries=summaries, accepted=accepted, excluded=excluded, hardware=list(hardware),
        measurement_window_s=[.32, 1.12], episode_wall_time_used=False)


def write_csv(path, rows):
    require(bool(rows), 'Refuse to write an empty official statistics table')
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader(); writer.writerows(rows)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.titleweight': 'bold', 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white'})
    return plt


def save_figure(fig, output, prefix):
    paths = [output/(prefix+'.png'), output/(prefix+'.pdf')]
    fig.savefig(paths[0], dpi=240)
    fig.savefig(paths[1], metadata={'CreationDate': None, 'ModDate': None})
    return paths


def manifest(output, prefix, inputs, artifacts, facts):
    import matplotlib
    payload = dict(schema='fair_robot_figure_provenance_v1', code_sha256=sha(__file__),
        inputs=[dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in inputs],
        outputs=[dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in artifacts],
        matplotlib_version=matplotlib.__version__, facts=facts)
    (output/(prefix+'_MANIFEST.json')).write_text(json.dumps(payload, indent=2), encoding='utf-8', newline='\n')


def figure_crossover(stage, validation, output):
    plt = pyplot(); fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharey=True)
    for ax, rate in zip(axes, VALIDATION_RATES):
        for method, offset in [('ours', -.18), ('v2e', .18)]:
            rows = [next(r for r in validation['summaries'] if r['method']==method and r['endpoint_hz']==rate and r['frontend_id']==f) for f in validation['frontends']]
            x = [i+offset for i in range(len(rows))]; values = [r['success_rate'] for r in rows]
            ax.bar(x, values, width=.34, color=COLORS[method], label=NAMES[method], zorder=3)
            ax.errorbar(x, values, yerr=[[max(0., r['success_rate']-r['wilson95_low']) for r in rows],
                [max(0., r['wilson95_high']-r['success_rate']) for r in rows]], fmt='none', ecolor='#334155', capsize=3, linewidth=1.1, zorder=4)
            for xi, row in zip(x, rows): ax.text(xi, row['success_rate']+.035, f"{row['successes']}/{row['n']}", ha='center', fontsize=10, zorder=5)
        ax.set_title(f'{rate:g} Hz source images'); ax.set_xticks(range(len(validation['frontends'])), [chr(65+i) for i in range(len(validation['frontends']))])
        ax.set_ylim(0, 1.17); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(axis='y', alpha=.18, zorder=0)
        ax.set_xlabel('Shared frontend configuration')
    axes[0].set_ylabel('Verified catch success'); axes[1].legend(loc='upper left', frameon=False)
    fig.suptitle('Genesis robot task: frontend crossover on development throws', y=.98, fontsize=15, fontweight='bold')
    descriptions = []
    for index, identifier in enumerate(validation['frontends']):
        match = re.fullmatch(r'w(0|10)_fit([01])_gap(50|100)', identifier)
        descriptions.append(f"{chr(65+index)}: {'packet-end detector' if match[1]=='0' else '10 ms windows'}, extra flight fit {'on' if match[2]=='1' else 'off'}, gap {match[3]} ms")
    fig.subplots_adjust(left=.08, right=.98, top=.85, bottom=.30, wspace=.14)
    fig.text(.035, .14, '\n'.join(descriptions), fontsize=9, color='#475569')
    fig.text(.035, .035, f"Same {len(validation['blocks'])} paired development blocks in every cell; all outcomes retained. Error bars: Wilson 95% intervals.", fontsize=9, color='#475569')
    artifacts = save_figure(fig, output, 'frontend_crossover'); plt.close(fig)
    table = output/'frontend_crossover_groups.csv'; write_csv(table, validation['summaries']); artifacts.append(table)
    manifest(output, 'frontend_crossover', stage['inputs'], artifacts,
        dict(independent_seed_blocks=len(validation['blocks']), shared_candidate_ids=validation['frontends'], rates=VALIDATION_RATES,
             no_generator_ranking_from_cache_score=True))


def figure_blind(stage, blind, output, bootstrap):
    plt = pyplot(); fig, ax = plt.subplots(figsize=(9.2, 5.8))
    for method, marker in [('ours', 'o'), ('v2e', 's')]:
        rows = [r for r in blind['summaries'] if r['method'] == method]
        x = [r['endpoint_hz'] for r in rows]; y = [r['success_rate'] for r in rows]
        ax.plot(x, y, marker=marker, color=COLORS[method], linewidth=2, markersize=6, label=NAMES[method])
        ax.fill_between(x, [r['wilson95_low'] for r in rows], [r['wilson95_high'] for r in rows], color=COLORS[method], alpha=.13)
        for rate, row in zip(x, rows): ax.annotate(f"{row['successes']}/{row['n']}", (rate, row['success_rate']), xytext=(0, 9 if method=='ours' else -16), textcoords='offset points', ha='center', fontsize=9, color=COLORS[method])
    for row, style in zip(blind['references'], ['--', ':']):
        method = row['method']; ax.axhline(row['success_rate'], color=COLORS[method], linestyle=style, linewidth=1.6,
            label=f"{NAMES[method]} ({row['successes']}/{row['n']}; reused)")
    ax.set_xscale('log', base=2); ax.set_xticks(blind['rates'], [f'{r:g}' for r in blind['rates']])
    ax.set_ylim(-.04, 1.08); ax.set_yticks([0, .25, .5, .75, 1]); ax.grid(alpha=.18)
    ax.set_xlabel('Source image rate (Hz)'); ax.set_ylabel('Verified catch success')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False, fontsize=9)
    fig.suptitle('Genesis robot task: locked-frontends blind evaluation', y=.98, fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.31)
    fig.text(.045, .035, f"Same {len(blind['blocks'])} seed blocks at all rates; each source uses one locked frontend. Shading: pointwise Wilson 95% intervals.\nReference actors appear once per block and are not added to source/rate denominators.", fontsize=8.8, color='#475569')
    artifacts = save_figure(fig, output, 'blind_success_vs_rate'); plt.close(fig)
    table = output/'blind_success_by_rate.csv'; write_csv(table, blind['summaries']+blind['references']); artifacts.append(table)
    paired = output/'blind_paired_by_rate.csv'; write_csv(paired, blind['paired']); artifacts.append(paired)
    if bootstrap is not None:
        path = output/'blind_block_bootstrap.json'; path.write_text(json.dumps(bootstrap, indent=2), encoding='utf-8', newline='\n'); artifacts.append(path)
    manifest(output, 'blind_success_vs_rate', stage['inputs']+[blind['selection_path']], artifacts,
        dict(independent_seed_blocks=len(blind['blocks']), source_frontends=blind['selected'], rates=blind['rates'],
             mcnemar_unit='paired seed block separately at each rate; rates are never pooled as independent observations',
             multiple_rate_tests='Holm-adjusted values reported alongside raw exact two-sided McNemar',
             bootstrap_unit=None if bootstrap is None else 'entire paired multi-rate seed block'))


def figure_cost(stage, blind, costs, cost_path, output):
    plt = pyplot(); fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for method, marker in [('ours', 'o'), ('v2e', 's')]:
        points = [r for r in costs['summaries'] if r['method'] == method]
        points.sort(key=lambda r: r['source_hz'])
        utility = [next(r for r in blind['summaries'] if r['method']==method and r['endpoint_hz']==p['source_hz']) for p in points]
        x = [p['mean_wall_s_per_sim_s'] for p in points]; y = [r['success_rate'] for r in utility]
        ax.plot(x, y, color=COLORS[method], linewidth=1.6, marker=marker, markersize=6, label=NAMES[method])
        ax.errorbar(x, y, xerr=[[max(0., p['mean_wall_s_per_sim_s']-p['minimum']) for p in points],
            [max(0., p['maximum']-p['mean_wall_s_per_sim_s']) for p in points]],
            yerr=[[max(0., r['success_rate']-r['wilson95_low']) for r in utility],
                  [max(0., r['wilson95_high']-r['success_rate']) for r in utility]],
            fmt='none', ecolor=COLORS[method], alpha=.65, linewidth=1, capsize=3)
        for point, util in zip(points, utility):
            ax.annotate(f"{point['source_hz']:g} Hz", (point['mean_wall_s_per_sim_s'], util['success_rate']),
                xytext=(5, 8 if method=='ours' else -15), textcoords='offset points', fontsize=9, color=COLORS[method])
    ax.set_xscale('log'); ax.set_ylim(-.04, 1.08); ax.grid(alpha=.18)
    ax.set_xlabel('Measured wall seconds per simulated second (lower is better)')
    ax.set_ylabel('Blind verified catch success'); ax.legend(frameon=False)
    fig.suptitle('Task utility versus measured native pipeline cost', y=.98, fontsize=15, fontweight='bold')
    fig.subplots_adjust(left=.10, right=.97, top=.88, bottom=.24)
    fig.text(.045, .045, 'Cost: matched hardware, fixed 0.32–1.12 s shadow workload; warm setup excluded. X: repeat mean and range.\nUtility: independent completed blind actors; Y: Wilson 95% intervals. Episode duration/wall time is never used for cost.', fontsize=8.8, color='#475569')
    artifacts = save_figure(fig, output, 'blind_success_vs_cost'); plt.close(fig)
    table = output/'validated_cost_by_rate.csv'; write_csv(table, costs['summaries']); artifacts.append(table)
    exclusions = output/'cost_row_admission.json'; exclusions.write_text(json.dumps(dict(excluded=costs['excluded'], hardware=costs['hardware']), indent=2), encoding='utf-8', newline='\n'); artifacts.append(exclusions)
    manifest(output, 'blind_success_vs_cost', stage['inputs']+[blind['selection_path'], Path(cost_path)], artifacts,
        dict(hardware=costs['hardware'], cost_scope=COST_SCOPE, fixed_shadow_window=costs['measurement_window_s'],
             episode_wall_time_used=False, reference_utility_not_assigned_unmeasured_cost=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path)
    parser.add_argument('--blind', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--cost', type=Path)
    parser.add_argument('--cost-gpu-uuid')
    parser.add_argument('--cost-min-repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('figures'))
    parser.add_argument('--frontends', default=','.join(DEFAULT_FRONTENDS))
    parser.add_argument('--validation-blocks', type=int, default=4)
    parser.add_argument('--blind-blocks', type=int, default=32)
    parser.add_argument('--bootstrap-repetitions', type=int, default=0)
    parser.add_argument('--bootstrap-seed', type=int, default=20261109)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--print-schema', choices=['episodes', 'cost'])
    args = parser.parse_args()
    if args.print_schema:
        print(','.join(EPISODE_FIELDS if args.print_schema=='episodes' else COST_FIELDS)); return
    try:
        require(args.validation is not None or args.blind is not None, 'Supply a completed validation and/or blind stage')
        require(args.cost is None or args.blind is not None, 'Utility–cost figure requires completed blind utility data')
        frontends = tuple(args.frontends.split(',')); require(len(frontends)==3 and len(set(frontends))==3, 'Expected the three registered shared crossover frontends')
        validation_stage = load_stage(args.validation, 'validation') if args.validation else None
        validation = validate_crossover(validation_stage, frontends, args.validation_blocks) if validation_stage else None
        blind_stage = load_stage(args.blind, 'blind') if args.blind else None
        selection = args.selection
        if blind_stage and selection is None:
            selection = next((p for p in [args.blind/'FRONTEND_SELECTION.json', args.blind.parent/'FRONTEND_SELECTION.json'] if p.is_file()), None)
            require(selection is not None, 'Supply the locked FRONTEND_SELECTION.json for the blind stage')
        blind = validate_blind(blind_stage, selection, args.blind_blocks, BLIND_RATES) if blind_stage else None
        costs = validate_cost(args.cost, blind, args.cost_gpu_uuid, args.cost_min_repeats) if args.cost else None
        bootstrap = block_bootstrap(blind, args.bootstrap_repetitions, args.bootstrap_seed) if blind and args.bootstrap_repetitions else None
        planned = []
        if validation: planned.append('frontend_crossover')
        if blind: planned.append('blind_success_vs_rate')
        if costs: planned.append('blind_success_vs_cost')
        if not args.overwrite:
            require(not any((args.output/(p+extension)).exists() for p in planned for extension in ['.png', '.pdf', '_MANIFEST.json']),
                    'Official outputs already exist; use a fresh output directory or explicitly --overwrite')
        if args.check_only:
            print(json.dumps(dict(status='PASS', check_only=True, plots_written=0, planned_figures=planned,
                validation_actors=None if validation is None else len(validation_stage['rows']),
                blind_actors=None if blind is None else len(blind_stage['rows']),
                independent_blind_blocks=None if blind is None else len(blind['blocks']),
                validated_native_cost_rows=None if costs is None else len(costs['accepted'])))); return
        args.output.mkdir(parents=True, exist_ok=True)
        if validation: figure_crossover(validation_stage, validation, args.output)
        if blind: figure_blind(blind_stage, blind, args.output, bootstrap)
        if costs: figure_cost(blind_stage, blind, costs, args.cost, args.output)
        print(json.dumps(dict(status='PASS', output=str(args.output.resolve()), figures=planned)))
    except (DataError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f'NOT_READY: {error}\nNo placeholder figures or invented measurements are produced.\n')


if __name__ == '__main__':
    main()
