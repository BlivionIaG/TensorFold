import argparse, hashlib, json, math, plistlib, subprocess, sys, threading, time
from datetime import datetime, timezone
from pathlib import Path


def clocks():
    return {'wall_ns': time.time_ns(), 'perf_counter_ns': time.perf_counter_ns(), 'monotonic_ns': time.monotonic_ns()}


def split_records(pending, chunk):
    parts = (pending + chunk).split(b'\0')
    return [part for part in parts[:-1] if part.strip()], parts[-1]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--runtime-source', required=True)
    parser.add_argument('--published-descendant', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    assert command
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    observed = {
        'label': 'PRIVATE_COMPLETE_GLOBAL_GPU_CLOCK_COVERAGE_NOT_QUALIFICATION',
        'runtime_source_revision': args.runtime_source,
        'published_docs_descendant': args.published_descendant,
        'controller_command': command,
        'raw_samples_private': True,
        'limitations': 'Global GPU clocks, not process GPU attribution or occupancy. Raw freq_hz units are preserved without assuming SI Hz. No power or frequency override.',
        'start': clocks(),
    }

    def save():
        (root / 'observed.json').write_text(json.dumps(observed, indent=2) + '\n')

    power_command = ['sudo', '-n', '/usr/bin/powermetrics', '--samplers', 'gpu_power,thermal', '--format', 'plist', '--buffer-size', '0', '--sample-rate', '1000']
    observed['power_command'] = power_command
    save()
    ready = threading.Event()
    state = {'complete_records': 0, 'parse_errors': [], 'incomplete_tail_bytes': 0}
    with (root / 'powermetrics.pliststream').open('wb') as data, (root / 'powermetrics.stderr.log').open('wb') as errors, (root / 'clock-samples.jsonl').open('w') as samples, (root / 'controller.log').open('wb') as log:
        sampler = subprocess.Popen(power_command, stdout=subprocess.PIPE, stderr=errors, stdin=subprocess.DEVNULL)
        observed['owned_sampler_pid'] = sampler.pid
        save()

        def record(block):
            try:
                row = plistlib.loads(block)
                timestamp = row['timestamp']
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                gpu = row.get('gpu')
                assert isinstance(gpu, dict) and isinstance(gpu.get('freq_hz'), (int, float)), 'missing actual GPU frequency field'
                assert math.isfinite(gpu['freq_hz']) and gpu['freq_hz'] > 0, 'invalid actual GPU frequency'
                previous = state.get('last_complete_timestamp')
                if previous is not None:
                    gap = (timestamp - datetime.fromisoformat(previous)).total_seconds()
                    assert gap > 0, 'clock sample time did not advance'
                    state['max_timestamp_gap_seconds'] = max(state.get('max_timestamp_gap_seconds', 0), gap)
                thin = {'timestamp': timestamp.isoformat(), 'elapsed_ns': row.get('elapsed_ns'), 'gpu': gpu, 'thermal_pressure': row.get('thermal_pressure'), 'received': clocks()}
                samples.write(json.dumps(thin) + '\n')
                samples.flush()
                state['complete_records'] += 1
                state.setdefault('first_complete_timestamp', thin['timestamp'])
                state['last_complete_timestamp'] = thin['timestamp']
                ready.set()
            except Exception as error:
                state['parse_errors'].append({'type': type(error).__name__, 'error': str(error), 'bytes': len(block)})

        def pump():
            pending = b''
            try:
                while chunk := sampler.stdout.read1(65536):
                    data.write(chunk)
                    data.flush()
                    complete, pending = split_records(pending, chunk)
                    for block in complete:
                        record(block)
                if pending.strip():
                    try:
                        plistlib.loads(pending)
                    except Exception:
                        state['incomplete_tail_bytes'] = len(pending)
                    else:
                        record(pending)
            except Exception as error:
                state['reader_error'] = {'type': type(error).__name__, 'error': str(error)}

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            assert ready.wait(15), 'no complete actual clock record; model controller not started'
            assert sampler.poll() is None, 'clock sampler ended before model controller'
            observed['controller_start'] = clocks()
            save()
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            observed['controller_returncode'] = result.returncode
        except BaseException as error:
            observed['controller_error'] = {'type': type(error).__name__, 'error': str(error)}
            raise
        finally:
            observed['controller_end'] = clocks()
            observed['sampler_ended_before_controller_cleanup'] = sampler.poll() is not None
            save()
            if sampler.poll() is None:
                flush = subprocess.run(['sudo', '-n', '/bin/kill', '-IO', str(sampler.pid)], capture_output=True)
                observed['owned_sampler_flush_returncode'] = flush.returncode
                stop = subprocess.run(['sudo', '-n', '/bin/kill', '-INT', str(sampler.pid)], capture_output=True)
                observed['owned_sampler_stop_returncode'] = stop.returncode
            try:
                sampler.wait(timeout=20)
            except subprocess.TimeoutExpired:
                kill = subprocess.run(['sudo', '-n', '/bin/kill', '-KILL', str(sampler.pid)], capture_output=True)
                observed['owned_sampler_kill_returncode'] = kill.returncode
                observed['owned_sampler_forced_stop'] = True
                sampler.wait(timeout=20)
            observed['sampler_returncode'] = sampler.returncode
            reader.join(timeout=20)
            observed['reader_stopped'] = not reader.is_alive()
            observed['samples'] = state
            observed['end'] = clocks()
            save()
            assert not reader.is_alive(), 'sampler reader did not stop; saved metadata retained'
            sampler.stdout.close()
    observed['sha256'] = {name: sha(root / name) for name in ('powermetrics.pliststream', 'powermetrics.stderr.log', 'clock-samples.jsonl', 'controller.log')}
    if state['complete_records']:
        first_ns = int(datetime.fromisoformat(state['first_complete_timestamp']).timestamp() * 1e9)
        last_ns = int(datetime.fromisoformat(state['last_complete_timestamp']).timestamp() * 1e9)
        observed['clock_time_coverage'] = {
            'first_sample_end_wall_ns': first_ns,
            'last_sample_end_wall_ns': last_ns,
            'began_before_controller': first_ns <= observed['controller_start']['wall_ns'],
            'ended_within_two_samples_of_controller': last_ns + 2000000000 >= observed['controller_end']['wall_ns'],
            'no_gap_over_two_and_half_samples': state.get('max_timestamp_gap_seconds', 0) <= 2.5,
        }
    save()
    print(json.dumps(observed), flush=True)
    assert observed['controller_returncode'] == 0
    assert state['complete_records'] and not state['parse_errors'] and 'reader_error' not in state
    assert not observed['sampler_ended_before_controller_cleanup'], 'clock coverage ended early; do not treat as complete'
    assert observed['sampler_returncode'] == 0 and not observed.get('owned_sampler_forced_stop', False)
    assert all(observed['clock_time_coverage'][key] for key in ('began_before_controller', 'ended_within_two_samples_of_controller', 'no_gap_over_two_and_half_samples')), 'incomplete clock time coverage; per-phase association remains unqualified'


if __name__ == '__main__':
    main()
