"""Offline A/A variability analysis; never calls or changes the original full-matrix verdict."""
import argparse
from collections import defaultdict
import statistics
from pathlib import Path

from qual_common import load, save

ARMS = ('plain_a', 'plain_b')


def rows(path):
    return [__import__('json').loads(line) for line in Path(path).read_text().splitlines() if line.strip()] \
        if Path(path).exists() else []


def span(values):
    return {'min': min(values), 'max': max(values)} if values else {'min': None, 'max': None}


def analyze(root):
    root = Path(root)
    meta = load(root / 'metadata.json')
    records = rows(root / 'cells.jsonl')
    events = rows(root / 'events.jsonl')
    cells = [c for c in records if c['kind'] == 'cell']
    oracles = [c for c in records if c['kind'] == 'oracle']
    expected = {(r, a, p, t, w) for r in range(3) for a in ARMS for p in ('code', 'chat')
                for t in (0.0, 1.0) for w in (1, 4, 8, 16)}
    expected_oracles = {(r, a, p, t, s) for r in range(3) for a in ARMS for p in ('code', 'chat')
                        for t in (0.0, 1.0) for s in ([1234] if t == 0 else range(1234, 1250))}
    identity = lambda c: (c['rep'], c['arm'], c['prompt'], float(c['temperature']), c['streams'])
    by_id = {identity(c): c for c in cells}
    seen_oracles = {(c['rep'], c['arm'], c['prompt'], float(c['temperature']), c['seed']) for c in oracles}
    failures = []
    for record in records:
        if not record.get('ok') or record.get('failures'):
            failures.append({'what': 'original_request_or_oracle_check_failed', 'record': record})
    if len(cells) != len(by_id) or len(oracles) != len(seen_oracles):
        failures.append({'what': 'duplicate_cell_or_oracle'})
    if set(by_id) != expected or seen_oracles != expected_oracles:
        failures.append({'what': 'missing_or_extra_cells_or_oracles'})
    hashes, outputs = defaultdict(list), defaultdict(list)
    for c in cells:
        want = [1234 + i if c['temperature'] > 0 else 1234 for i in range(c['streams'])]
        if [r.get('seed') for r in c['requests']] != want:
            failures.append({'what': 'wrong_request_seed_order', 'cell': identity(c)})
        for r in c['requests']:
            key = (c['prompt'], float(c['temperature']), r.get('seed'))
            hashes[key].append((c['arm'], r.get('token_sha256')))
            outputs[key].append((c['arm'], r.get('output_sha256')))
            if r.get('tokens') != 512 or r.get('complete') is not True or r.get('error'):
                failures.append({'what': 'incomplete_request', 'cell': identity(c), 'seed': r.get('seed')})
            if r.get('cached_tokens') != 0:
                failures.append({'what': 'unexpected_cache_reuse', 'cell': identity(c), 'seed': r.get('seed')})
    for c in oracles:
        key = (c['prompt'], float(c['temperature']), c['seed'])
        hashes[key].extend((c['arm'], c.get(name)) for name in ('solo_token_sha256', 'draft_false_token_sha256'))
    equality = []
    for key in sorted(hashes):
        items, text = hashes[key], outputs[key]
        token_equal = {a for a, _ in items} == set(ARMS) and len({h for _, h in items}) == 1 and all(h for _, h in items)
        output_equal = {a for a, _ in text} == set(ARMS) and len({h for _, h in text}) == 1 and all(h for _, h in text)
        equality.append({'prompt': key[0], 'temperature': key[1], 'seed': key[2],
                         'token_equal_all_arms_reps_oracles': bool(token_equal),
                         'output_equal_all_measured_arms_reps': bool(output_equal),
                         'token_observations': len(items), 'output_observations': len(text)})
        if not token_equal or not output_equal:
            failures.append({'what': 'both_arm_equality_failed', 'key': key})
    if len(equality) != 34:
        failures.append({'what': 'missing_equality_keys', 'observed': len(equality), 'expected': 34})
    groups, paired_percentages, median_percentages = [], [], []
    for prompt in ('code', 'chat'):
        for temperature in (0.0, 1.0):
            for width in (1, 4, 8, 16):
                group = {'prompt': prompt, 'temperature': temperature, 'streams': width, 'arms': {}, 'paired': []}
                for arm in ARMS:
                    values = [by_id[(r, arm, prompt, temperature, width)]['aggregate_tps'] for r in range(3)
                              if (r, arm, prompt, temperature, width) in by_id]
                    good = [v for v in values if isinstance(v, (int, float)) and v > 0]
                    if len(good) != 3:
                        failures.append({'what': 'missing_valid_speed', 'arm': arm, 'prompt': prompt,
                                         'temperature': temperature, 'streams': width})
                    group['arms'][arm] = {'values_in_rep_order': values, 'median': statistics.median(good) if good else None,
                                          'range': span(good)}
                for rep in range(3):
                    left = by_id.get((rep, ARMS[0], prompt, temperature, width))
                    right = by_id.get((rep, ARMS[1], prompt, temperature, width))
                    if left is None or right is None:
                        continue
                    paired_equal = len(left['requests']) == len(right['requests']) == width and all(
                        l.get('seed') == r.get('seed') and l.get('token_sha256') == r.get('token_sha256')
                        and l.get('output_sha256') == r.get('output_sha256')
                        for l, r in zip(left['requests'], right['requests']))
                    a, b = left['aggregate_tps'], right['aggregate_tps']
                    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or min(a, b) <= 0:
                        continue
                    percent = 100 * (b / a - 1)
                    paired_percentages.append(percent)
                    group['paired'].append({'rep': rep, 'plain_a_tps': a, 'plain_b_tps': b,
                                            'delta_tps_b_minus_a': b - a, 'delta_percent': percent,
                                            'both_arm_equal': paired_equal})
                    if not paired_equal:
                        failures.append({'what': 'paired_equality_failed', 'rep': rep, 'prompt': prompt,
                                         'temperature': temperature, 'streams': width})
                a, b = (group['arms'][arm]['median'] for arm in ARMS)
                group['median_delta_tps_b_minus_a'] = b - a if a is not None and b is not None else None
                group['median_delta_percent'] = 100 * (b / a - 1) if a and b else None
                if group['median_delta_percent'] is not None:
                    median_percentages.append(group['median_delta_percent'])
                group['paired_delta_tps_range'] = span([r['delta_tps_b_minus_a'] for r in group['paired']])
                group['paired_delta_percent_range'] = span([r['delta_percent'] for r in group['paired']])
                groups.append(group)
    labels = [f'r{rep}-{arm}' for rep, rotation in enumerate(meta['protocol']['aa']['rotations']) for arm in rotation]
    ready = [e for e in events if e['event'] == 'ready']
    shutdown = [e for e in events if e['event'] == 'shutdown']
    if [e['label'] for e in ready] != labels or any(e.get('dspark_mode') is not None for e in ready):
        failures.append({'what': 'wrong_plain_launches'})
    if [e['label'] for e in shutdown] != labels or any(e.get('forced_kill') or e.get('returncode') != 0 for e in shutdown):
        failures.append({'what': 'incomplete_or_failed_shutdown'})
    if any(e['event'] == 'abort' for e in events):
        failures.append({'what': 'run_aborted'})
    result = {'schema': 'lfm2-dspark-plain-aa/1', 'label': meta['label'],
              'source_revision': meta['source_revision'], 'binary_sha256': meta['binary_sha256'],
              'expected_cells': 96, 'recorded_cells': len(cells),
              'expected_oracle_seeds': 204, 'recorded_oracle_seeds': len(oracles),
              'missing_cells': sorted(expected - set(by_id)), 'both_arm_equality': equality,
              'cell_medians_and_deltas': groups, 'observed_paired_delta_percent_range': span(paired_percentages),
              'observed_median_delta_percent_range': span(median_percentages),
              'failures': failures, 'complete': not failures, 'exit_code': 1 if failures else 0,
              'interpretation': 'Variability evidence only; no performance pass/fail threshold or waiver. '
                                'The original full-matrix exit code remains authoritative and unchanged.'}
    save(root / 'summary.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    raise SystemExit(analyze(p.parse_args().output)['exit_code'])
