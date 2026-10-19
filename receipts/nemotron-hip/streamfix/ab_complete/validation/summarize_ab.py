#!/usr/bin/env python3
"""Summarize two completed ab.sh runs from locally copied, bound receipts."""
import argparse
import contextlib
import hashlib
import io
import json
import math
from pathlib import Path
import re
import statistics
import sys
import ab_validate as v

COMMITS = json.loads((Path(__file__).parent / 'commits.json').read_text())
assert set(COMMITS) == {'c46', 'd46'} and all(re.fullmatch(r'[0-9a-f]{40}', x) for x in COMMITS.values())

ORDER = ['c46', 'd46', 'd46', 'c46']
CONDITIONS = ['decode_on', 'decode_off', 'cold_8192', 'cold_32768']
CLIENTS = {'decode/bench_openai.py', 'decode/bench_evidence.py', 'cold/bench_cache.py',
           'cold/bench_concurrent.py', 'cold/bench_evidence.py', 'cold/bench_auth.py'}
HARNESSES = {'ab.sh', 'ab_validate.py', 'capture_identity.py'}


def check(ok, why):
    if not ok:
        raise ValueError(why)


def compare(values):
    old = statistics.mean([values[0], values[3]])
    new = statistics.mean([values[1], values[2]])
    return {'hold_order': values, 'old_mean': old, 'new_mean': new,
            'delta_percent': 100 * (new / old - 1)}


def resolve_directory(root, remote):
    name = Path(remote).name
    check(re.fullmatch(r'[A-Za-z0-9_-]+_\d{8}T\d{6}_\d+', name), 'invalid output directory name')
    # Accept root as the copied run itself, its ab parent, or a higher copied logs root.
    found = [p.resolve() for p in [root, *root.rglob(name)] if p.is_dir() and p.name == name]
    found = list(dict.fromkeys(found))
    check(len(found) == 1, f'expected one local output directory {name}, found {len(found)}')
    return found[0]


def summarize(root, log):
    text = log.read_text()
    check(not re.search(r'\bAB_FAILED\b', text), f'{log}: AB_FAILED')
    check(len(re.findall(r'^\d\d:\d\d:\d\d AB_DONE$', text, re.M)) == 1, f'{log}: missing/duplicate AB_DONE')
    check(text.rstrip().endswith('AB_DONE'), f'{log}: output after AB_DONE')
    outputs = re.findall(r'^\d\d:\d\d:\d\d output=(.+)$', text, re.M)
    check(len(outputs) == 1, f'{log}: expected one output= directory')
    directory = resolve_directory(root, outputs[0])
    client_pairs = re.findall(r'^\d\d:\d\d:\d\d client (\S+) sha256=([0-9a-f]{64})$', text, re.M)
    harness_pairs = re.findall(r'^\d\d:\d\d:\d\d harness (\S+) sha256=([0-9a-f]{64})$', text, re.M)
    check(len(client_pairs) == 6 and set(dict(client_pairs)) == CLIENTS, 'missing/duplicate benchmark client hash bindings')
    check(len(harness_pairs) == len(HARNESSES) and set(dict(harness_pairs)) == HARNESSES, 'missing/duplicate harness hash bindings')
    client_hashes, harness_hashes = dict(client_pairs), dict(harness_pairs)
    bindings = {}
    for line in text.splitlines():
        if ' tree=' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
        tree = fields.get('tree')
        check(tree in COMMITS and tree not in bindings, f'{log}: invalid/duplicate tree binding')
        check(fields.get('commit') == COMMITS[tree], f'{log}: wrong source commit for {tree}')
        check(re.fullmatch(r'[0-9a-f]{64}', fields.get('binary_sha256', '')), f'{log}: missing binary SHA256')
        check(fields.get('binary', '').endswith(f'/out-{fields.get("arch")}/native-hip/bin/tensorfold-native'), 'wrong native prefix')
        bindings[tree] = fields
    check(set(bindings) == set(COMMITS), f'{log}: missing source bindings')
    arches = {(b.get('arch'), b.get('world'), b.get('devs')) for b in bindings.values()}
    check(len(arches) == 1, 'old/new rig mismatch')
    arch, world, devs = arches.pop()
    check((arch, world, devs) in [('gfx1030', '1', '2'), ('gfx1100', '2', '0,1')], 'unexpected rig')
    rig = 'V620tp1' if world == '1' else 'W7800tp2'
    fixtures = dict(re.findall(r'prompt(8192|32768)\.json sha256=([0-9a-f]{64})', text))
    check(len(fixtures) == 2, 'missing prompt fixture SHA256 bindings')
    result = {'rig': rig, 'log': str(log), 'remote_output': outputs[0], 'local_output': str(directory),
              'source_binary_bindings': bindings, 'client_sha256': client_hashes, 'harness_sha256': harness_hashes, 'prompt_file_sha256': fixtures, 'conditions': {}}
    all_ports = []
    for condition in CONDITIONS:
        names = [p for p in directory.glob(condition + '_*.json') if not p.name.endswith('.identity.json')]
        parsed = []
        for path in names:
            match = re.fullmatch(re.escape(condition) + r'_(c46|d46)_(\d+)\.json', path.name)
            check(match is not None, f'unexpected receipt name {path}')
            parsed.append((int(match[2]), match[1], path))
        parsed.sort()
        check(len(parsed) == 4 and [x[1] for x in parsed] == ORDER, f'{condition}: missing/incorrect quartet')
        paths = [x[2] for x in parsed]
        ports = [x[0] for x in parsed]
        all_ports.extend(ports)
        logged = re.findall(r'^\d\d:\d\d:\d\d ' + condition + r' (c46|d46) file=(.+)$', text, re.M)
        check([(t, Path(f).name) for t, f in logged] == [(t, p.name) for _, t, p in parsed], f'{condition}: receipts not bound by log')
        mode = 'decode' if condition.startswith('decode') else 'cold'
        n = 0 if mode == 'decode' else int(condition.split('_')[1])
        with contextlib.redirect_stdout(io.StringIO()):
            hashes = [v.read(p, mode, n) for p in paths]
        check(all(h == hashes[0] for h in hashes), f'{condition}: unequal token hashes/fixtures')
        documents = [json.loads(p.read_text()) for p in paths]
        item = {'order': ORDER, 'ports': ports, 'receipts': [p.name for p in paths],
                'receipt_sha256': [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]}
        if mode == 'decode':
            item['prompts'] = {}
            for prompt, temperature in hashes[0]:
                rows = [next(r for r in d if r['prompt'] == prompt and r['temperature'] == temperature) for d in documents]
                check(all(isinstance(r.get('ttft_s_median'), (int, float)) and math.isfinite(r['ttft_s_median']) and r['ttft_s_median'] > 0 for r in rows), 'invalid decode TTFT median')
                item['prompts'][prompt] = {'decode_tps': compare([r['decode_tps_median'] for r in rows]),
                                           'ttft_s_median': compare([r['ttft_s_median'] for r in rows]),
                                           'all_repetition_tps': [r['decode_tps_all'] for r in rows],
                                           'token_sha_hold_order': [h[(prompt, temperature)] for h in hashes]}
        else:
            item.update({'first_ttft_s': compare([d['rows'][0]['ttft_s'] for d in documents]),
                         'all3_median_ttft_s': compare([statistics.median(r['ttft_s'] for r in d['rows']) for d in documents]),
                         'first_prefill_tps': compare([n / d['rows'][0]['ttft_s'] for d in documents]),
                         'all3_ttft_s': [[r['ttft_s'] for r in d['rows']] for d in documents],
                         'token_sha_hold_order': [h['hashes'] for h in hashes], 'fixture_sha256': hashes[0]['fixture']})
        result['conditions'][condition] = item
    check(all_ports == list(range(all_ports[0], all_ports[0] + 16)), 'ports are not consecutive across conditions')
    return result


def pr_line(runs):
    parts = []
    for r in runs:
        c = r['conditions']
        bits = []
        for mode in ['on', 'off']:
            for prompt, d in c['decode_' + mode]['prompts'].items():
                m = d['decode_tps']
                bits.append(f'{mode}/{prompt} {m["old_mean"]:.2f}→{m["new_mean"]:.2f} tok/s ({m["delta_percent"]:+.2f}%)')
        for n in [8192, 32768]:
            m = c['cold_' + str(n)]['first_ttft_s']
            bits.append(f'cold{n} first TTFT {m["old_mean"]:.3f}→{m["new_mean"]:.3f}s ({m["delta_percent"]:+.2f}%)')
        parts.append(r['rig'] + ': ' + ', '.join(bits))
    return f'A/B {COMMITS["c46"][:8]}→{COMMITS["d46"][:8]}, old/new/new/old; ' + '; '.join(parts) + '; complete replies and token_sha equal in every quartet.'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('logs', type=Path, nargs=2)
    p.add_argument('--output', type=Path, help='write JSON here; otherwise JSON goes to stdout')
    args = p.parse_args()
    runs = [summarize(args.root.resolve(), log.resolve()) for log in args.logs]
    check({r['rig'] for r in runs} == {'V620tp1', 'W7800tp2'}, 'expected one run from each rig')
    check(runs[0]['client_sha256'] == runs[1]['client_sha256'], 'rigs used different benchmark clients')
    check(runs[0]['harness_sha256'] == runs[1]['harness_sha256'], 'rigs used different harnesses')
    check(runs[0]['prompt_file_sha256'] == runs[1]['prompt_file_sha256'], 'rigs used different prompt files')
    for n in ['8192', '32768']:
        check(runs[0]['conditions']['cold_' + n]['fixture_sha256'] == runs[1]['conditions']['cold_' + n]['fixture_sha256'], 'rigs used different prompt fixtures')
    report = {'schema_version': 1, 'validated': True, 'runs': runs, 'pr46_line': pr_line(runs)}
    encoded = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(encoded)
        print(report['pr46_line'])
    else:
        print(encoded, end='')

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.exit('FAIL: ' + str(e))
