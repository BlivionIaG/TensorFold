#!/usr/bin/env python3
"""Package validated A/B evidence without publishing workstation metadata."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import ab_validate as validator
import summarize_ab as summarizer


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def private_string(value):
    return bool(re.search(r'/Users/|/home/|/private/|/tmp/|localhost|127\.0\.0\.1|\b192\.168\.\d+\.\d+\b|\b10\.\d+\.\d+\.\d+\b|\b172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+\b|\b[\w.-]+@(?:[\w.-]+\.)[A-Za-z]{2,}\b', value, re.I))


def audit(value):
    if isinstance(value, str):
        require(not private_string(value), 'receipt contains private path/address metadata; inspect before publication')
    elif isinstance(value, dict):
        for key, item in value.items():
            audit(key)
            audit(item)
    elif isinstance(value, list):
        for item in value:
            audit(item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('summary', type=Path)
    p.add_argument('roots', nargs=2, type=Path, help='the two actual timestamped A/B receipt directories')
    p.add_argument('output', type=Path, help='new clean output directory; must not exist')
    p.add_argument('--fixtures-dir', required=True, type=Path, help='directory containing original prompt8192.json and prompt32768.json')
    p.add_argument('--clients-dir', type=Path, help='original benchmark clients in decode/ and cold/ subdirectories')
    p.add_argument('--source-bridge', type=Path, help='optional source-comparison-only bridge JSON')
    args = p.parse_args()
    report = json.loads(args.summary.read_text())
    require(report.get('schema_version') == 1 and report.get('validated') is True, 'summary is not validated schema 1')
    runs = report.get('runs', [])
    require(len(runs) == 2 and {r['rig'] for r in runs} == {'V620tp1', 'W7800tp2'}, 'expected both rigs')
    require(not args.output.exists(), 'output directory already exists')
    roots = {root.name: root for root in args.roots}
    require(len(roots) == 2 and all(root.is_dir() for root in roots.values()), 'expected two distinct receipt roots')
    source_bridge = None
    if args.source_bridge:
        source_bridge = json.loads(args.source_bridge.read_text())
        require(source_bridge.get('schema_version') == 1 and source_bridge.get('artifact_kind') == 'source_comparison_only' and source_bridge.get('runtime_claim') is False, 'invalid source bridge schema')
        require({b.get('pr') for b in source_bridge.get('bridges', [])} == {41, 46}, 'source bridge must cover PR41 and PR46')
        for bridge in source_bridge['bridges']:
            require(all(re.fullmatch(r'[0-9a-f]{40}', bridge.get(k, '')) for k in ['proof_source_commit', 'review_source_commit']), 'invalid source bridge commit')
            if bridge['pr'] == 46:
                require(bridge['review_source_commit'] == summarizer.COMMITS['d46'], 'source bridge targets a different final head')
            require(re.fullmatch(r'[0-9a-f]{64}', bridge.get('git_diff_sha256', '')), 'missing bridge diff hash')
            for changed in bridge['changed_files']:
                name = changed['path']
                require('..' not in Path(name).parts and (name == 'docs/kernel-contract.md' or name == 'zig/build/hip_contract.zig' or name.startswith('zig/tests/hip/')), 'bridge changes production source')
        audit(source_bridge)
    manifest = {'schema_version': 1, 'comparison_order': summarizer.ORDER,
                'source_commits': summarizer.COMMITS, 'runs': [], 'files': [], 'validators': {}}
    fixture_data = {}
    for n in [8192, 32768]:
        path = args.fixtures_dir / f'prompt{n}.json'
        ids = json.loads(path.read_text())
        require(isinstance(ids, list) and len(ids) == n and all(type(x) is int and 0 <= x <= 0xffffffff for x in ids), f'invalid uint32 prompt{n} array')
        fixture_data[str(n)] = {'path': path, 'ids': ids, 'file_sha256': sha(path),
                                'fixture_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
    client_sources = []
    if args.clients_dir:
        for mode, names in {'decode': ['bench_openai.py', 'bench_evidence.py'],
                            'cold': ['bench_cache.py', 'bench_concurrent.py', 'bench_evidence.py', 'bench_auth.py']}.items():
            for name in names:
                source = args.clients_dir / mode / name
                require(source.is_file(), f'missing reproduction client {mode}/{name}')
                client_sources.append((mode + '/' + name, source))
        optional = args.clients_dir / 'cold' / 'bench_openai.py'
        if optional.is_file():
            client_sources.append(('cold/bench_openai.py', optional))
    for run in runs:
        require(set(run.get('client_sha256', {})) == summarizer.CLIENTS, 'missing startup client hash bindings')
        require(set(run.get('harness_sha256', {})) == summarizer.HARNESSES, 'missing startup harness hash bindings')
    require(runs[0]['client_sha256'] == runs[1]['client_sha256'], 'rig client versions differ')
    require(runs[0]['harness_sha256'] == runs[1]['harness_sha256'], 'rig harness versions differ')
    require(args.clients_dir is not None, 'startup-bound benchmark clients require --clients-dir')
    for relative, source in client_sources:
        if relative in summarizer.CLIENTS:
            require(sha(source) == runs[0]['client_sha256'][relative], f'client differs from measured source: {relative}')
    for name in summarizer.HARNESSES:
        require(sha(Path(__file__).parent / name) == runs[0]['harness_sha256'][name], f'harness differs from measured source: {name}')
    clean_runs = []
    pending = []
    for run in runs:
        root = roots.get(Path(run['remote_output']).name)
        require(root is not None, 'summary output directory does not match supplied roots')
        bindings = run['source_binary_bindings']
        settings = {(b['arch'], b['world'], b['devs']) for b in bindings.values()}
        expected = ('gfx1030', '1', '2') if run['rig'] == 'V620tp1' else ('gfx1100', '2', '0,1')
        require(settings == {expected}, 'rig settings mismatch')
        clean = {'rig': run['rig'], 'client_sha256': run['client_sha256'], 'harness_sha256': run['harness_sha256'], 'settings': {'arch': expected[0], 'world': int(expected[1]),
                 'devices': expected[2], 'context': 65536, 'decode_parallel': 4, 'cold_parallel': 1,
                 'cold_prompt_cache_gib': 0, 'decode_tokens': 256, 'decode_repetitions': 3,
                 'cold_tokens': 16, 'cold_repeats': 1, 'cold_streams': 1},
                 'prompt_file_sha256': run['prompt_file_sha256'], 'conditions': run['conditions']}
        for n, fixture in fixture_data.items():
            require(run['prompt_file_sha256'][n] == fixture['file_sha256'], f'prompt{n} original file SHA mismatch')
            require(run['conditions']['cold_' + n]['fixture_sha256'] == fixture['fixture_sha256'], f'prompt{n} canonical fixture SHA mismatch')
        for tree, commit in summarizer.COMMITS.items():
            b = bindings[tree]
            require(b['commit'] == commit and re.fullmatch(r'[0-9a-f]{64}', b['binary_sha256']), 'invalid build binding')
        ports = []
        for condition in summarizer.CONDITIONS:
            item = clean['conditions'][condition]
            require(item['order'] == summarizer.ORDER and len(item['receipts']) == 4, 'bad quartet')
            mode = 'decode' if condition.startswith('decode') else 'cold'
            n = 0 if mode == 'decode' else int(condition.split('_')[1])
            hashes = []
            for i, (tree, name) in enumerate(zip(summarizer.ORDER, item['receipts'])):
                require(Path(name).name == name, 'unsafe receipt name')
                m = re.fullmatch(re.escape(condition) + '_' + tree + r'_(\d+)\.json', name)
                require(m is not None and int(m[1]) == item['ports'][i], 'receipt order/port mismatch')
                ports.append(int(m[1]))
                path = root / name
                require(sha(path) == item['receipt_sha256'][i], 'raw receipt changed since summary validation')
                with contextlib.redirect_stdout(io.StringIO()):
                    hashes.append(validator.read(path, mode, n))
                if mode == 'cold':
                    require(hashes[-1]['fixture'] == fixture_data[str(n)]['fixture_sha256'], 'cold receipt fixture does not match actual prompt array')
                raw = json.loads(path.read_text())
                if mode == 'cold':
                    require(raw.get('requested_reply_tokens') == 16 and raw.get('temperature') == 0 and raw.get('seed') == 1234, 'cold request parameters do not match exact packaged request')
                audit(raw)
                relative = f'{run["rig"]}/{name}'
                binding = {'source_commit': bindings[tree]['commit'], 'binary_sha256': bindings[tree]['binary_sha256'],
                           'rig': run['rig'], 'tree': tree, 'condition': condition, 'hold_index': i,
                           'raw_sha256': sha(path)}
                if mode == 'cold':
                    binding['prompt_file_sha256'] = run['prompt_file_sha256'][str(n)]
                    binding['fixture_sha256'] = hashes[-1]['fixture']
                if isinstance(raw, list):
                    envelope = {'receipt_schema': 'bench_openai_list_wrapped_v1', **binding, 'results': raw}
                else:
                    require(all(raw[k] == binding[k] for k in set(raw) & set(binding)), 'raw receipt collides with annotation fields')
                    envelope = {**raw, **binding, 'receipt_schema': 'bench_cache_annotated_v1'}
                pending.append((relative, envelope))
                manifest['files'].append({'file': relative, **binding, 'settings': clean['settings']})
            require(all(h == hashes[0] for h in hashes), 'token hashes differ across quartet')
        require(ports == list(range(ports[0], ports[0] + 16)), 'nonconsecutive run ports')
        clean_runs.append(clean)
        manifest['runs'].append({'rig': clean['rig'], 'settings': clean['settings'],
                                 'prompt_file_sha256': clean['prompt_file_sha256'],
                                 'builds': {t: {'source_commit': b['commit'], 'binary_sha256': b['binary_sha256']} for t, b in bindings.items()}})
    require(clean_runs[0]['prompt_file_sha256'] == clean_runs[1]['prompt_file_sha256'], 'different prompt files between rigs')
    # Recompute comparison statistics from trusted raw files and compare every recorded measurement.
    for run in clean_runs:
        rig = run['rig']
        source = next(r for r in runs if r['rig'] == rig)
        root = roots[Path(source['remote_output']).name]
        for condition, item in run['conditions'].items():
            docs = [json.loads((root / name).read_text()) for name in item['receipts']]
            if condition.startswith('decode'):
                for prompt, row in item['prompts'].items():
                    selected = [next(x for x in d if x['prompt'] == prompt and x['temperature'] == 0) for d in docs]
                    require(row['decode_tps'] == summarizer.compare([x['decode_tps_median'] for x in selected]), 'decode timing summary mismatch')
            else:
                import statistics
                require(item['first_ttft_s'] == summarizer.compare([d['rows'][0]['ttft_s'] for d in docs]), 'first timing summary mismatch')
                require(item['all3_median_ttft_s'] == summarizer.compare([statistics.median(x['ttft_s'] for x in d['rows']) for d in docs]), 'median timing summary mismatch')
    clean_summary = {'schema_version': 1, 'validated': True, 'runs': clean_runs,
                     'pr46_line': summarizer.pr_line(clean_runs)}
    audit(clean_summary)
    audit(manifest)
    args.output.mkdir(parents=True)
    manifest['fixtures'] = {}
    for n, fixture in fixture_data.items():
        ids_target = args.output / 'ids' / f'prompt{n}.json'
        ids_target.parent.mkdir(exist_ok=True)
        shutil.copyfile(fixture['path'], ids_target)
        # Exact json.dumps payload emitted by bench_concurrent._stream; all three phases use this body.
        body = {'model': 'm', 'max_tokens': 16, 'temperature': 0, 'stream': True,
                'stream_options': {'include_usage': True}, 'ignore_eos': True, 'seed': 1234,
                'prompt': fixture['ids']}
        request_target = args.output / 'requests' / f'cold{n}.json'
        request_target.parent.mkdir(exist_ok=True)
        request_target.write_bytes(json.dumps(body).encode())
        manifest['fixtures'][n] = {'ids_file': f'ids/prompt{n}.json', 'prompt_file_sha256': sha(ids_target),
                                   'fixture_sha256': fixture['fixture_sha256'], 'tokens': int(n),
                                   'request_file': f'requests/cold{n}.json', 'request_sha256': sha(request_target),
                                   'method': 'POST', 'endpoint': '/v1/completions',
                                   'phases': ['first', 'repeated', 'concurrent']}
    for relative, envelope in pending:
        target = args.output / relative
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(envelope, indent=2) + '\n')
        next(f for f in manifest['files'] if f['file'] == relative)['packaged_sha256'] = sha(target)
    for name in ['ab_validate.py', 'summarize_ab.py', 'package_nanfix_receipts.py', 'ab.sh']:
        source = Path(__file__).parent / name
        destination = args.output / 'validation' / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        manifest['validators'][name] = {'sha256': sha(source), 'version': 'sha256-addressed source',
                                           'role': 'measured-run harness' if name in summarizer.HARNESSES else 'post-run validation'}
    manifest['benchmark_clients'] = {}
    for relative, source in client_sources:
        target = args.output / 'tools' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest['benchmark_clients']['tools/' + relative] = {'sha256': sha(target)}
    if source_bridge is not None:
        target = args.output / 'source_bridge.json'
        shutil.copyfile(args.source_bridge, target)
        manifest['source_bridge'] = {'file': target.name, 'sha256': sha(target), 'runtime_claim': False}
    (args.output / 'summary.json').write_text(json.dumps(clean_summary, indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'summary.md').write_text('# NaN fix A/B receipts\n\n' + clean_summary['pr46_line'] + '\n\nFull hold-order values, all three cold request timings and medians, source commits, binary hashes and token hashes are in summary.json and manifest.json.\n')
    (args.output / 'README.md').write_text('''# NaN fix A/B evidence

Each condition runs old, new, new, old (c46, d46, d46, c46). Decode uses drafts on and off, two public prompts and three measured repetitions after warmup. Cold measurements use 8192 and 32768 token fixtures, no prompt retention, one initial request, one repeated request and one concurrent request. The first TTFT and median of all three requests are recorded separately.

Every receipt was validated as complete with the expected token count. Matching token_sha values are required at each corresponding request across all four holds. Cold receipts also require zero cached tokens, cold cache state, and equal fixture hashes. Binary hashes bind each measurement to its source commit as recorded by the runner.

Decode JSON preserves the entire original list under results; the wrapper names its original schema. Cold JSON preserves every original field and adds binding annotations. No evidence arrays are shortened. The original prompt arrays are copied byte-for-byte under ids/ and checked against both startup file SHA256 and canonical fixture SHA256. The request bodies under requests/ reproduce bench_concurrent._stream's json.dumps payload exactly; each cold phase uses the same body with model m, 16 tokens, temperature 0 and seed 1234. Endpoint paths and request hashes are in the manifest. Raw-file hashes in manifest.json refer to pre-annotation files; packaged hashes refer to these published files. No workstation logs or paths are included.

To validate a published receipt with validation/ab_validate.py, first remove annotation fields from a cold receipt, or extract results from a decode wrapper, into a temporary JSON file. Run `python3 validation/ab_validate.py decode 0 FILE` or `python3 validation/ab_validate.py cold 8192 FILE` (32768 as applicable). Supply four files in hold order to check quartet equality. Validator scripts are identified by SHA256 in manifest.json. When supplied, the exact original clients and all nonstandard imports are archived under tools/decode and tools/cold with SHA256 hashes. Run these clients with Python 3.11 or later; use the packaged ids arrays and the request settings above. The manifest records whether clients were included.
''')
    print(f'Packaged {len(pending)} receipts; manifest.json, summary.json, summary.md and validation scripts ready')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.exit('FAIL: ' + str(e))
