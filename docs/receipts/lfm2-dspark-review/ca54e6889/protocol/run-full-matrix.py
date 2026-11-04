import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('binding', 'guard-proof', 'binary', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    b = json.loads(Path(args.binding).read_text())
    proof = json.loads(Path(args.guard_proof).read_text())
    assert proof['source_revision'] == b['source_revision']
    assert proof['base_revision'] == b['base_revision']
    assert all(row['returncode'] == 0 for row in proof['records'])
    assert {row['bits'] for row in proof['cases']} == {0, 8}
    binary_sha = sha(args.binary)
    assert binary_sha == proof['production_binary_sha256']
    for name, digest in b['driver_sha256'].items():
        assert sha(Path(b['frozen_driver']) / name) == digest
    for kind in ('target', 'drafter'):
        for name, digest in b['model_sha256'][kind].items():
            assert sha(Path(b['remote_' + kind]) / name) == digest
    sys.path.insert(0, b['frozen_driver'])
    import bench_lfm2_dspark as gate

    env, _ = gate.clean_env()
    root = Path(args.output)
    root.mkdir(parents=True)
    records = []
    binding = dict(b, native_binary_sha256=binary_sha, guard_proof_sha256=sha(args.guard_proof),
                   controller_sha256=sha(__file__), original_driver_unmodified=True)
    state = {'binding': binding, 'records': records, 'complete': False}

    def save():
        (root / 'observed.json').write_text(json.dumps(state, indent=2) + '\n')

    def run(label, command):
        log = root / (label + '.log')
        began = time.monotonic()
        with log.open('xb') as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        row = {'label': label, 'command': command, 'returncode': result.returncode,
               'elapsed_s': time.monotonic() - began, 'log_sha256': sha(log)}
        records.append(row)
        save()
        print(json.dumps(row), flush=True)
        return result.returncode

    save()
    with Path(b['shared_lock']).open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert os.fstat(lock.fileno()).st_ino == b['shared_lock_inode_observed']
        assert gate.pressure_level() == 1
        assert not gate.competing_processes(Path(args.binary).name)
        assert run('native-help', [args.binary, '--help']) == 0
    command = [sys.executable, str(Path(b['frozen_driver']) / 'bench_lfm2_dspark.py'), 'run',
               '--binary', args.binary, '--target', b['remote_target'], '--drafter', b['remote_drafter'],
               '--source-revision', b['source_revision'], '--target-revision', b['target_revision'],
               '--drafter-revision', b['drafter_revision'], '--output-dir', str(root / '1394-full-gate')]
    code = run('1394-full-gate', command)
    assert sha(args.binary) == binary_sha
    assert code in (0, 3), 'original full gate exited outside its completed verdict'
    summary_path = root / '1394-full-gate/summary.json'
    summary = json.loads(summary_path.read_text())
    assert summary['smoke'] is False and summary['rotation_balanced'] is True
    assert summary['expected_cells'] == summary['recorded_cells'] == summary['ok_cells'] == 144
    assert summary['missing_cells'] == []
    assert summary['expected_oracle_seeds'] == summary['oracle_seeds'] == summary['oracle_ok'] == 306
    assert summary['exactness']['failures'] == summary['scenario']['failures'] == 0
    assert summary['exit_code'] == code
    state['original_summary_sha256'] = sha(summary_path)
    state['complete'] = True
    state['original_gate_returncode'] = code
    state['completed_unix_ns'] = time.time_ns()
    save()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
