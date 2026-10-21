"""Raw-ID streaming matrix client. No model/server launch; default is a local request preview."""
import argparse, hashlib, json, math, re, statistics, sys, threading, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'clients'))
import bench_concurrent as bench

p = argparse.ArgumentParser(__doc__)
p.add_argument('base'); p.add_argument('--model', default='m')
p.add_argument('--fixture', type=Path, required=True)
p.add_argument('--prompt-tokens', type=int, choices=[1024, 16384], required=True)
p.add_argument('--tokens', type=int, choices=[512, 1024], required=True)
p.add_argument('--concurrency', type=int, choices=[1, 8], required=True)
p.add_argument('--mode', choices=['off', 'on'], required=True)
p.add_argument('--reps', type=int, default=3); p.add_argument('--warmups', type=int, default=1)
p.add_argument('--reference-output', type=Path, help='Required complete matching C1 output for C8 equality')
p.add_argument('--output', type=Path, required=True); p.add_argument('--execute', action='store_true')
a = p.parse_args()
assert (a.prompt_tokens, a.tokens) in [(1024, 512), (16384, 1024)]
assert a.reps > 0 and a.warmups > 0
ids = json.loads(a.fixture.read_text())
assert len(ids) == a.prompt_tokens and all(type(x) is int and 0 <= x <= 0xffffffff for x in ids)
body = {'model': a.model, 'max_tokens': a.tokens, 'temperature': 0, 'stream': True,
        'stream_options': {'include_usage': True}, 'ignore_eos': True, 'seed': 1234}
if a.mode == 'off': body['draft'] = False
body['prompt'] = ids
sha = lambda b: hashlib.sha256(b).hexdigest()
if not a.execute:
    print(json.dumps({'execute': False, 'prompt_tokens': len(ids), 'tokens': a.tokens,
        'concurrency': a.concurrency, 'mode': a.mode, 'body_sha256': sha(json.dumps(body).encode()),
        'note': 'Requires separately allocated source-bound native server: parallel8/context65536/promptcache0; no server started.'}, indent=2))
    raise SystemExit
assert not a.output.exists(), 'Preserve previous output'
a.output.parent.mkdir(parents=True, exist_ok=True)
report = {'schema_version': 1, 'complete': False, 'mode': a.mode, 'concurrency': a.concurrency,
          'prompt_tokens': a.prompt_tokens, 'requested_tokens': a.tokens,
          'fixture_sha256': sha(a.fixture.read_bytes()), 'exact_request_body': body,
          'request_body_sha256': sha(json.dumps(body).encode()), 'batches': [],
          'true_token_itl_distribution_available': False, 'pure_gpu_prefill_timer_available': False}
def save():
    tmp = a.output.with_suffix(a.output.suffix + '.tmp')
    tmp.write_text(json.dumps(report, indent=2) + '\n'); tmp.replace(a.output)
def positive(x):
    return type(x) in (int, float) and math.isfinite(x) and x > 0
fingerprints = set()
reference_sha = None
if a.concurrency == 8:
    assert a.reference_output is not None, 'C8 requires one actual prior C1 reference output'
    reference_raw = a.reference_output.read_bytes(); reference_sha = sha(reference_raw)
    reference = json.loads(reference_raw)
    assert reference['complete'] and not reference.get('error') and reference['concurrency'] == 1
    assert reference['mode'] == a.mode and reference['fixture_sha256'] == report['fixture_sha256']
    assert reference['requested_tokens'] == a.tokens
    ref_hashes = {(r['raw_runtime']['token_sha256'], r['output_sha256']) for b in reference['batches'] for r in b['runs']}
    assert len(ref_hashes) == 1
    fingerprints.update(ref_hashes)
    report['c1_reference_output_sha256'] = reference_sha

try:
    for index in range(a.warmups + a.reps):
        samples, sample_errors = [], []
        stop = threading.Event()
        def observe():
            while not stop.is_set():
                try:
                    with urllib.request.urlopen(a.base.rstrip('/') + '/stats', timeout=2) as response:
                        value = json.load(response)
                    samples.append({'at': time.perf_counter(), 'stats': value})
                except Exception as exc:
                    sample_errors.append({'at': time.perf_counter(), 'error': type(exc).__name__})
                stop.wait(.2)
        observer = threading.Thread(target=observe)
        observer.start()
        try:
            runs = bench.together(a.base.rstrip('/'), a.model,
                [({'name': 'fixture', 'kind': 'completion', 'prompt': ids}, 1234)] * a.concurrency,
                a.tokens, 0, draft=(a.mode == 'on'))
        finally:
            stop.set(); observer.join(timeout=3)
        row = {'index': index, 'phase': 'warmup' if index < a.warmups else 'measured',
               'runs': runs, 'status_samples': samples, 'status_sample_errors': sample_errors,
               'status_sampling_seconds': .2}
        report['batches'].append(row); save()
        assert len(runs) == a.concurrency and samples and not sample_errors
        prefills = []
        for r in runs:
            assert r.get('complete') and not r.get('error') and r['tokens'] == a.tokens
            assert r['prompt_tokens'] == a.prompt_tokens and r['cached_tokens'] == 0
            assert positive(r['decode_tps']) and positive(r['ttft_s'])
            runtime = r['raw_runtime']
            assert r['actual_request_body'] == body and r['actual_request_body_sha256'] == report['request_body_sha256']
            assert runtime.get('drafts') is (a.mode == 'on')
            finish = [c.get('finish_reason') for e in r['raw_sse_events'] for c in e['chunk'].get('choices', []) if c.get('finish_reason') is not None]
            assert finish == ['length'], 'Expected exact length termination'
            assert positive(runtime.get('prefill_seconds')) and positive(runtime.get('seconds'))
            assert positive(runtime.get('time_to_first_token'))
            assert positive(runtime.get('tokens_per_second')) and math.isclose(runtime['tokens_per_second'], (r['tokens'] - 1) / runtime['seconds'], rel_tol=1e-9, abs_tol=1e-9)
            token_id_hash = runtime.get('token_sha256')
            assert isinstance(token_id_hash, str) and re.fullmatch('[0-9a-f]{64}', token_id_hash)
            assert r['token_sha'] in {token_id_hash, token_id_hash[:12]}
            fingerprints.add((token_id_hash, r['output_sha256']))
            assert len(fingerprints) == 1, 'Greedy repeat/concurrency output mismatch'
            span = r['last'] - r['first']
            r['delivery_tpot_seconds'] = span / (r['tokens'] - 1)
            r['end_to_end_seconds'] = r['completed'] - r['sent']
            r['server_prefill_elapsed_tps'] = r['prompt_tokens'] / runtime['prefill_seconds']
            r['client_ttft_prefill_proxy_tps'] = r['prompt_tokens'] / r['ttft_s']
            r['chunk_gap_seconds'] = [y[0] - x[0] for x, y in zip(r['pieces'], r['pieces'][1:])]
            prefills.append(runtime['prefill_seconds'])
        wall = max(r['completed'] for r in runs) - min(r['sent'] for r in runs)
        delivery = max(r['last'] for r in runs) - min(r['first'] for r in runs)
        first_window = max(r['first'] for r in runs) - min(r['sent'] for r in runs)
        prompt_total = sum(r['prompt_tokens'] for r in runs)
        completion_total = sum(r['tokens'] for r in runs)
        row['aggregate'] = {'actual_requests': len(runs), 'prompt_tokens_total': prompt_total,
            'completion_tokens_total': completion_total, 'wall_seconds': wall,
            'send_spread_seconds': max(r['sent'] for r in runs) - min(r['sent'] for r in runs),
            'completion_end_to_end_tps': completion_total / wall,
            'prompt_end_to_end_tps': prompt_total / wall,
            'total_end_to_end_tps': (prompt_total + completion_total) / wall,
            'completion_delivery_window_tps': sum(r['tokens'] - 1 for r in runs) / delivery,
            'weighted_server_prefill_service_tps': prompt_total / sum(prefills),
            'batch_first_content_window_prefill_proxy_tps': prompt_total / first_window}
        save()
    if reference_sha is not None: assert sha(a.reference_output.read_bytes()) == reference_sha
    report['complete'] = True; save()
except BaseException as exc:
    report['error'] = type(exc).__name__ + ': ' + str(exc); save(); raise
print(a.output)
