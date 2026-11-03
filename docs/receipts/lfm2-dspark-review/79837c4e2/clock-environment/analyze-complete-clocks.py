import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_interval(sample):
    timestamp = datetime.fromisoformat(sample['timestamp'])
    assert timestamp.microsecond == 0, 'Revisit the explicitly one-second plist date resolution'
    date_floor = int(timestamp.timestamp()) * 1_000_000_000
    # Cover either start-date or end-date semantics plus one-second date quantization.
    # Delivery waits for the next NUL; never use its receipt time as the sampling time.
    return date_floor - sample['elapsed_ns'], date_floor + sample['elapsed_ns'] + 1_000_000_000


def summarize_samples(samples):
    elapsed = sum(sample['elapsed_ns'] for sample in samples)
    states = [state for sample in samples for state in sample['gpu']['dvfm_states']]
    busy = sum(state['used_ns'] for state in states)
    maximum = max((state['freq'] for state in states), default=None)
    return {
        'complete_interior_samples': len(samples),
        'nominal_sample_elapsed_ns': elapsed,
        'global_gpu_busy_ns': busy,
        'global_gpu_active_ratio_not_occupancy': busy / elapsed if elapsed else None,
        'busy_weighted_raw_frequency': sum(state['freq'] * state['used_ns'] for state in states) / busy if busy else None,
        'maximum_reported_raw_frequency': maximum,
        'busy_fraction_at_maximum_reported_state': sum(state['used_ns'] for state in states if state['freq'] == maximum) / busy if busy else None,
        'thermal_pressure_observations': sorted({sample['thermal_pressure'] for sample in samples}),
    }


def analyze(root):
    outer = json.loads((root / 'observed.json').read_text())
    probe = json.loads((root / 'probe/observed.json').read_text())
    samples = [json.loads(line) for line in (root / 'clock-samples.jsonl').read_text().splitlines()]
    assert outer['controller_returncode'] == 0 and outer['sampler_returncode'] == 0
    assert not outer.get('owned_sampler_forced_stop', False) and outer['reader_stopped']
    assert not outer['samples']['parse_errors']
    assert outer['samples']['complete_records'] == len(samples)
    assert all(outer['clock_time_coverage'][key] for key in ('began_before_controller', 'ended_within_two_samples_of_controller', 'no_gap_over_two_and_half_samples'))
    assert len(probe['records']) == 3 and all(record['returncode'] == 0 for record in probe['records'])
    assert probe['binding']['base_source'] == outer['runtime_source_revision']
    assert probe['binding']['published_docs_descendant'] == outer['published_docs_descendant']
    for name, digest in outer['sha256'].items():
        assert sha(root / name) == digest, (name, 'actual capture bytes changed')
    anchors = [outer[key] for key in ('start', 'controller_start', 'controller_end', 'end')]
    anchors += [sample['received'] for sample in samples]
    offsets = [anchor['wall_ns'] - anchor['perf_counter_ns'] for anchor in anchors]
    low, high = min(offsets), max(offsets)
    intervals = [source_interval(sample) for sample in samples]
    phases = []
    by_arm = {}
    for record in probe['records']:
        arm = record['arm']
        by_arm[arm] = record['phases']
        prefix = f'r0-{arm}-'
        for phase in record['phases']:
            assert phase['cell'].startswith(prefix)
            start = round(phase['sent'] * 1e9) + high
            end = round(phase['completed'] * 1e9) + low
            selected = [i for i, (earliest, latest) in enumerate(intervals) if earliest >= start and latest <= end]
            phases.append({
                'arm': arm, 'scenario': phase['cell'].removeprefix(prefix), 'phase': phase['phase'],
                'sent_perf_seconds': phase['sent'], 'completed_perf_seconds': phase['completed'],
                'rpc_wall_ms': phase['rpc_wall_ms'], 'activity': phase['activity'],
                'enrolled_contexts': sum(bool(row['tracked']) for row in phase['enrollments']),
                'private_delta': phase['private_delta'],
                'source_interval_sample_indices': selected,
                'clocks': summarize_samples([samples[i] for i in selected]),
            })
    plain = by_arm['plain']
    exact = []
    for arm, arm_phases in by_arm.items():
        if arm == 'plain':
            continue
        assert len(arm_phases) == len(plain)
        for left, right in zip(plain, arm_phases, strict=True):
            left_key = (left['cell'].removeprefix('r0-plain-'), left['phase'], left['requests'], left['max_new'])
            right_key = (right['cell'].removeprefix(f'r0-{arm}-'), right['phase'], right['requests'], right['max_new'])
            assert left_key == right_key
            assert left['runs'] == right['runs'], (arm, left_key, 'cross-arm token oracle differs')
        exact.append({'arm': arm, 'phase_pairs': len(plain), 'all_run_seeds_lengths_hashes_equal': True})
    measured = [phase for phase in phases if phase['phase'] == 'measured']
    plain_measured = {phase['scenario']: phase for phase in measured if phase['arm'] == 'plain'}
    comparisons = []
    for phase in measured:
        if phase['arm'] == 'plain':
            continue
        baseline = plain_measured[phase['scenario']]
        a = phase['private_delta']['target']['decode']
        b = baseline['private_delta']['target']['decode']
        comparisons.append({
            'arm': phase['arm'], 'scenario': phase['scenario'],
            'activity_equal': phase['activity'] == baseline['activity'],
            'all_target_physical_row_histograms_equal': phase['private_delta']['rows_by_count'] == baseline['private_delta']['rows_by_count'],
            'plain_decode_gpu_ms': b['gpu_ms'], 'arm_decode_gpu_ms': a['gpu_ms'],
            'decode_gpu_delta_percent': (a['gpu_ms'] / b['gpu_ms'] - 1) * 100,
            'plain_decode_calls': b['calls'], 'arm_decode_calls': a['calls'],
            'plain_clocks': baseline['clocks'], 'arm_clocks': phase['clocks'],
        })
    return {
        'label': 'PRIVATE_GLOBAL_CLOCK_ASSOCIATION_NOT_QUALIFICATION_OR_CAUSAL_PROOF',
        'runtime_source_revision': outer['runtime_source_revision'],
        'published_docs_descendant': outer['published_docs_descendant'],
        'source_binding': probe['binding'],
        'controller_returncode': outer['controller_returncode'],
        'sampler_returncode': outer['sampler_returncode'],
        'clean_nonforced_server_exits': len(probe['records']),
        'complete_clock_records': len(samples),
        'partial_tail_bytes': outer['samples']['incomplete_tail_bytes'],
        'observed_wall_to_perf_offset_range_ns': high - low,
        'clock_time_coverage': outer['clock_time_coverage'],
        'source_date_resolution_ns': 1_000_000_000,
        'phase_sample_rule': 'Conservative interval [date-elapsed_ns,date+elapsed_ns+1second] covers either start-date or end-date semantics and one-second quantization. Entire interval must be inside RPC window, using the observed wall/perf offset range. No delayed receipt timestamp is treated as a sampling timestamp.',
        'limitations': 'Global GPU clock/busy observations, not process attribution, SIMD occupancy or bandwidth. Raw frequency units preserved. Interior-only samples exclude boundaries and may be absent for short phases. No A/A control, confidence bound, qualification retry, timing clearance, power override or causal intervention.',
        'global_clock_summary': summarize_samples(samples),
        'cross_arm_exact_phase_pairs': exact,
        'phases': phases,
        'measured_comparisons': comparisons,
        'capture_sha256': {name: sha(root / name) for name in ('observed.json', 'probe/observed.json', 'clock-samples.jsonl', 'powermetrics.pliststream')},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.capture_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('source_binding', 'phases', 'measured_comparisons')}, indent=2))


if __name__ == '__main__':
    main()
