"""Separate plain/plain experiment using original server, requests, warmups and scenario APIs."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace

from analyze_plain_aa import analyze
from qual_common import load, require, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('binding', 'guard-proof', 'binary', 'protocol', 'protocol-sha256', 'output',
                 'source-revision', 'source-sha256'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    b, proof = load(a.binding), load(a.guard_proof)
    require(b['source_revision'] == proof['source_revision'] == a.source_revision, 'Wrong source revision')
    require(b['source_archive_sha256'] == a.source_sha256, 'Wrong source archive')
    require(sha(a.protocol) == a.protocol_sha256, 'Predeclared protocol changed')
    protocol = load(a.protocol)
    require(protocol['aa']['rotations'] == [['plain_a', 'plain_b'], ['plain_b', 'plain_a'], ['plain_a', 'plain_b']],
            'Unexpected A/A launch declaration')
    binary_sha = sha(a.binary)
    require(binary_sha == proof['production_binary_sha256'], 'Binary differs from guard-built source')
    for name, digest in b['driver_sha256'].items():
        require(sha(Path(b['frozen_driver']) / name) == digest, 'Frozen driver changed: ' + name)
    sys.path.insert(0, b['frozen_driver'])
    import bench_lfm2_dspark as d
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = SimpleNamespace(binary=a.binary, target=b['remote_target'], drafter=b['remote_drafter'],
                          widths='1,4,8,16', temperatures='0,1', tokens=512, reps=3, seed=1234,
                          port=18194, context=4096, startup_timeout=900, smoke=False,
                          source_revision=a.source_revision, target_revision=b['target_revision'],
                          drafter_revision=b['drafter_revision'])

    class AAContext(d.Context):
        study_arm = None

        def event(self, kind, **data):
            super().event(kind, study_arm=self.study_arm, **data)

    ctx = AAContext(cfg, out)
    ctx.logs.mkdir()
    metadata = {
        'schema': 'lfm2-dspark-plain-aa/1', 'label': protocol['aa']['label'],
        'source_revision': a.source_revision, 'source_archive_sha256': a.source_sha256,
        'binary_sha256': binary_sha, 'binding': b, 'protocol': protocol,
        'protocol_sha256': sha(a.protocol), 'controller_sha256': sha(__file__),
        'started_unix_ns': time.time_ns(), 'environment': d.metadata(ctx)['environment'],
        'removed_environment': ctx.env_removed,
        'prompts': d.bc.PROMPTS, 'same_native_argv_for_both_arms': d.Server(ctx, 'declaration', 'plain').argv(),
        'original_driver_and_verdict_unmodified': True,
        'full_matrix_evidence_separate': True,
        'copy_behavior': 'Unchanged plain command; no drafter and no disabling copies',
    }
    save(out / 'metadata.json', metadata)
    state = {'complete': False, 'launches': [], 'started_unix_ns': time.time_ns()}
    save(out / 'observed.json', state)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, d.interrupt)
    try:
        with Path(b['shared_lock']).open('r+') as lock:
            require(os.fstat(lock.fileno()).st_ino == Path(b['shared_lock']).stat().st_ino == 3719605,
                    'Shared lock inode changed')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            require(d.pressure_level() == 1, 'OS pressure is not normal')
            require(not d.competing_processes(Path(a.binary).name), 'Competing model process')
            for kind in ('target', 'drafter'):
                for name, digest in b['model_sha256'][kind].items():
                    require(sha(Path(b['remote_' + kind]) / name) == digest, 'Model identity changed')
            for rep, rotation in enumerate(protocol['aa']['rotations']):
                for arm in rotation:
                    require(sha(a.binary) == binary_sha, 'Bound binary changed')
                    require(Path(b['shared_lock']).stat().st_ino == 3719605, 'Lock path changed')
                    label = f'r{rep}-{arm}'
                    ctx.study_arm = arm
                    launch = {'rep': rep, 'arm': arm, 'native_arm': 'plain', 'label': label,
                              'started_unix_ns': time.time_ns()}
                    state['launches'].append(launch)
                    save(out / 'observed.json', state)
                    # Only evidence labels differ; the original Server always receives native arm plain.
                    with d.Server(ctx, label, 'plain') as server:
                        launch['pid'] = server.proc.pid
                        launch['argv'] = server.argv()
                        save(out / 'observed.json', state)
                        for item in d.bc.PROMPTS:
                            for temperature in ctx.temps:
                                d.scenario(ctx, server, rep, arm, item, temperature)
                    launch['completed_unix_ns'] = time.time_ns()
                    launch['server_log_sha256'] = sha(server.log_path)
                    save(out / 'observed.json', state)
            require(sha(a.binary) == binary_sha, 'Bound binary changed during A/A')
    except BaseException as error:
        state['error'] = {'type': type(error).__name__, 'message': str(error)}
        ctx.event('abort', detail=ctx.scrub(f'{type(error).__name__}: {error}'))
    result = analyze(out)
    state.update(complete=result['complete'], exit_code=result['exit_code'], completed_unix_ns=time.time_ns(),
                 summary_sha256=sha(out / 'summary.json'))
    save(out / 'observed.json', state)
    return result['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
