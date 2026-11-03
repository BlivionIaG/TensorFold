import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


T_DF5_ONE_SIDED_95 = 2.015048373333


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interval(values):
    assert len(values) == 6
    mean = statistics.mean(values)
    half = T_DF5_ONE_SIDED_95 * statistics.stdev(values) / math.sqrt(6)
    return mean - half, mean + half


def analyze(root, clock_root):
    state = json.loads((root / 'observed.json').read_text())
    clocks = json.loads((clock_root / 'observed.json').read_text())
    assert state['complete'] is True and len(state['launches']) == 24
    assert clocks['controller_returncode'] == 0 and clocks['sampler_returncode'] == 0
    assert clocks['reader_stopped'] and not clocks.get('owned_sampler_forced_stop', False)
    assert not clocks['samples']['parse_errors']
    assert all(clocks['clock_time_coverage'][key] for key in (
        'began_before_controller', 'ended_within_two_samples_of_controller', 'no_gap_over_two_and_half_samples'))
    binding = state['binding']['binding']
    assert binding['source_revision'] == clocks['runtime_source_revision']
    assert binding['native_private_instrumentation'] is False
    cells = {}
    for launch in state['launches']:
        assert launch['returncode'] == 0 and len(launch['cells']) == 48
        measured = [cell for cell in launch['cells'] if cell['phase'] == 'measured']
        assert len(measured) == 16 and all(cell['original_oracles_equal'] for cell in launch['cells'])
        for cell in measured:
            key = (cell['prompt'], cell['temperature'], cell['streams'])
            role_key = (launch['rotation'], launch['role'])
            assert role_key not in cells.setdefault(key, {})
            cells[key][role_key] = cell
    assert len(cells) == 16
    aa, comparisons = [], []
    for key, observations in sorted(cells.items()):
        assert len(observations) == 24
        aa_logs = []
        candidate_logs = {'bf16': [], 'bits8': []}
        points = []
        for rotation in range(6):
            before = observations[(rotation, 'plain_before')]['aggregate_tps']
            after = observations[(rotation, 'plain_after')]['aggregate_tps']
            reference = math.sqrt(before * after)
            aa_logs.append(math.log(after / before))
            row = {'rotation': rotation, 'plain_before_tps': before, 'plain_after_tps': after,
                   'plain_reference_geometric_mean_tps': reference,
                   'plain_after_vs_before_percent': (after / before - 1) * 100}
            for arm in candidate_logs:
                candidate = observations[(rotation, arm)]
                ratio = candidate['aggregate_tps'] / reference
                candidate_logs[arm].append(math.log(ratio))
                row[arm] = {'aggregate_tps': candidate['aggregate_tps'], 'paired_delta_percent': (ratio - 1) * 100,
                            'activity': candidate['activity']}
            points.append(row)
        aa_low, aa_high = interval(aa_logs)
        aa_valid = aa_low > math.log(0.99) and aa_high < math.log(1.01)
        identity = {'prompt': key[0], 'temperature': key[1], 'streams': key[2]}
        aa.append({**identity, 'rounds': points,
                   'mean_plain_after_vs_before_percent': math.expm1(statistics.mean(aa_logs)) * 100,
                   'aa_90percent_interval_percent': [math.expm1(aa_low) * 100, math.expm1(aa_high) * 100],
                   'within_prespecified_one_percent_equivalence_margin': aa_valid})
        for arm, logs in candidate_logs.items():
            low, high = interval(logs)
            median = (statistics.median([math.exp(value) for value in logs]) - 1) * 100
            lower_percent = math.expm1(low) * 100
            comparisons.append({**identity, 'arm': arm,
                                'median_paired_delta_percent': median,
                                'mean_paired_delta_percent': math.expm1(statistics.mean(logs)) * 100,
                                'one_sided_95percent_lower_bound_percent': lower_percent,
                                'one_sided_95percent_upper_bound_percent': math.expm1(high) * 100,
                                'observed_median_inside_owner_margin': median > -1,
                                'lower_bound_inside_owner_margin': low > math.log(0.99),
                                'same_cell_aa_valid': aa_valid,
                                'six_paired_log_ratios': logs})
    valid = all(cell['within_prespecified_one_percent_equivalence_margin'] for cell in aa)
    noninferior = all(cell['observed_median_inside_owner_margin'] and cell['lower_bound_inside_owner_margin'] for cell in comparisons)
    return {
        'label': 'PRESPECIFIED_CONTROLLED_STUDY_NOT_REPLACEMENT_FROZEN_VERDICT',
        'source_revision': binding['source_revision'], 'native_binary_sha256': binding['native_binary_sha256'],
        'protocol_sha256': binding['protocol_sha256'], 'original_qualification_source': binding['oracle_source_revision'],
        'original_frozen_qualification_replaced': False,
        'clean_nonforced_server_exits': 24, 'measured_cells': 384, 'fixed_warmup_cells': 768,
        'all_original_token_oracles_equal': True,
        'global_clock_records': clocks['samples']['complete_records'],
        'aa_valid_in_every_cell': valid, 'all32_candidate_bounds_inside_owner_margin': noninferior,
        'controlled_owner_one_percent_verdict': 'PASS' if valid and noninferior else 'INCONCLUSIVE' if not valid else 'NOT_CLEARED',
        'original_maintainer_zero_tolerance_verdict_changed': False,
        'method': 'Six independent launch rounds. Candidate TPS divided by geometric mean of bracketing independent plain launches. Median of six paired ratios is reported. Student-t intervals use mean log ratio, df5, critical2.015048373333. A/A90percent equivalence and candidate95percent one-sided noninferiority are separate required checks.',
        'limits': 'Normal log-ratio model and launch-round independence are assumptions, not a universal error floor or proof of stationarity. Every observation retained; no adaptive sample count, discarded slow run, thermal override or qualification relabelling.',
        'aa_cells': aa, 'comparisons': comparisons,
        'inputs_sha256': {'study_observed': sha(root / 'observed.json'), 'clock_observed': sha(clock_root / 'observed.json')},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-root', type=Path, required=True)
    parser.add_argument('--clock-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.study_root, args.clock_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('aa_cells', 'comparisons')}, indent=2))


if __name__ == '__main__':
    main()
