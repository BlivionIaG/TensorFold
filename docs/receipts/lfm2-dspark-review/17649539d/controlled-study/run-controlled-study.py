import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@contextmanager
def presence(root, session, cwd):
    stop = threading.Event()
    failures = []

    def pulse():
        while not stop.wait(180):
            began = time.time_ns()
            try:
                result = subprocess.run(['atc', 'pulse', '1374', str(session), 'working', '--json'],
                                        cwd=cwd, capture_output=True, text=True, timeout=40)
                record = {'began_unix_ns': began, 'ended_unix_ns': time.time_ns(),
                          'returncode': result.returncode, 'result': json.loads(result.stdout)}
                if result.returncode != 0 or record['result'].get('publication') != 'public':
                    failures.append('ATC active-work check-in failed')
            except Exception as error:
                record = {'began_unix_ns': began, 'ended_unix_ns': time.time_ns(),
                          'error': {'type': type(error).__name__, 'message': str(error)}}
                failures.append('ATC active-work check-in failed')
            with (root / 'coordination.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')

    worker = threading.Thread(target=pulse, daemon=True)
    worker.start()
    try:
        yield failures
    finally:
        stop.set()
        worker.join(timeout=45)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f'Owned study interrupted by signal {signum}')


def main():
    parser = argparse.ArgumentParser()
    for name in ('driver', 'binary', 'target', 'drafter', 'output', 'binding', 'protocol', 'oracle-cells'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--lock-inode', type=int, required=True)
    parser.add_argument('--atc-session', type=int, required=True)
    parser.add_argument('--coordination-root', required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.driver)
    import bench_lfm2_dspark as d

    root = Path(args.output)
    root.mkdir(parents=True)
    binding = json.loads(Path(args.binding).read_text())
    protocol = json.loads(Path(args.protocol).read_text())
    assert sha(args.binary) == binding['native_binary_sha256']
    assert sha(args.protocol) == binding['protocol_sha256']
    assert sha(args.oracle_cells) == binding['oracle_cells_sha256']
    for name, digest in binding['driver_sha256'].items():
        assert sha(Path(args.driver) / name) == digest
    assert protocol['scope']['rounds'] == 6
    assert protocol['scope']['widths'] == [1, 4, 8, 16]
    assert protocol['scope']['measured_sweeps_per_launch'] == 1
    assert binding['native_private_instrumentation'] is False
    references = {}
    for line in Path(args.oracle_cells).read_text().splitlines():
        row = json.loads(line)
        if row['kind'] != 'oracle':
            continue
        assert row['ok'] and not row['failures']
        assert row['solo_token_sha256'] == row['draft_false_token_sha256']
        key = (row['prompt'], float(row['temperature']), row['seed'])
        previous = references.setdefault(key, row['solo_token_sha256'])
        assert previous == row['solo_token_sha256']
    assert len(references) == 34
    settings = {
        'label': protocol['label'], 'binding': binding, 'protocol': protocol,
        'original_frozen_qualification_replaced': False,
        'controller_sha256': sha(__file__), 'started_unix_ns': time.time_ns(),
    }
    (root / 'binding.json').write_text(json.dumps(settings, indent=2) + '\n')
    launches = []
    state = {'binding': settings, 'launches': launches, 'complete': False}

    def save():
        (root / 'observed.json').write_text(json.dumps(state, indent=2) + '\n')

    save()
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    with presence(root, args.atc_session, args.coordination_root) as coordination_errors, open(d.LOCK, 'r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.fstat(lock.fileno()).st_ino == args.lock_inode
        assert d.pressure_level() == 1
        assert not d.competing_processes(Path(args.binary).name)
        for kind in ('target', 'drafter'):
            for name, digest in binding['model_sha256'][kind].items():
                assert sha(Path(getattr(args, kind)) / name) == digest
        for rotation in range(6):
            candidates = ('bf16', 'bits8') if rotation % 2 == 0 else ('bits8', 'bf16')
            for role in ('plain_before', *candidates, 'plain_after'):
                arm = 'plain' if role.startswith('plain_') else role
                label = f'r{rotation}-{role}'
                out = root / label
                out.mkdir()
                launch = {'rotation': rotation, 'role': role, 'arm': arm, 'started_unix_ns': time.time_ns(), 'cells': []}
                launches.append(launch)
                save()
                cfg = SimpleNamespace(binary=args.binary, target=args.target, drafter=args.drafter,
                                      widths='1,4,8,16', temperatures='0,1', tokens=512, reps=1, seed=1234,
                                      port=18194, context=4096, startup_timeout=300)
                ctx = d.Context(cfg, out)
                ctx.logs.mkdir()
                child = None
                try:
                    with d.Server(ctx, label, arm) as server:
                        child = server.proc
                        for sweep in range(3):
                            phase = 'measured' if sweep == 2 else f'warmup_{sweep}'
                            for prompt_name in ('code', 'chat'):
                                item = next(p for p in d.bc.PROMPTS if p['name'] == prompt_name)
                                for temperature in (0.0, 1.0):
                                    specs = [(item, sd) for sd in d.seeds_for(ctx.seed, ctx.widths, temperature)]
                                    for width in ctx.widths:
                                        assert not coordination_errors, coordination_errors
                                        server.check()
                                        runs = d.bc.together(ctx.base, d.SERVED_NAME, specs[:width], 512, temperature)
                                        cell = f'{label}-{prompt_name}-T{temperature:g}-w{width}'
                                        ctx.raw(cell, phase, runs)
                                        for run in runs:
                                            failures = d.problems(run, 512, True)
                                            assert not failures, (cell, phase, failures)
                                            expected = references[(prompt_name, temperature, run['seed'])]
                                            assert run['token_sha256'] == expected, (cell, phase, 'original token oracle differs')
                                            assert run['cached_tokens'] == 0, (cell, phase, 'cache is not disabled')
                                        server.check()
                                        aggregate = d.bc.aggregates(runs)['aggregate_tps']
                                        assert aggregate is not None and aggregate > 0
                                        record = {
                                            'kind': 'controlled_cell', 'phase': phase, 'rotation': rotation,
                                            'role': role, 'arm': arm, 'prompt': prompt_name,
                                            'temperature': temperature, 'streams': width,
                                            'aggregate_tps': aggregate, 'activity': d.total_activity(runs),
                                            'sent': min(r['sent'] for r in runs),
                                            'completed': max(r['completed'] for r in runs),
                                            'run_hashes': [r['token_sha256'] for r in runs],
                                            'original_oracles_equal': True,
                                        }
                                        ctx.record(record)
                                        launch['cells'].append(record)
                                        save()
                    launch['returncode'] = child.returncode
                    assert child.returncode == 0
                    launch['completed_unix_ns'] = time.time_ns()
                    assert len(launch['cells']) == 48
                    launch['private_native_log_sha256'] = sha(server.log_path)
                    print(json.dumps({'launch_completed': label, 'warmup_cells': 32, 'measured_cells': 16,
                                      'returncode': child.returncode}), flush=True)
                except BaseException as error:
                    launch['error'] = {'type': type(error).__name__, 'message': str(error)}
                    if child is not None:
                        launch['observed_returncode'] = child.poll()
                    save()
                    raise
                save()
    state['complete'] = True
    state['completed_unix_ns'] = time.time_ns()
    save()
    print(json.dumps({'complete': True, 'launches': len(launches), 'measured_cells': 384}), flush=True)


if __name__ == '__main__':
    main()
