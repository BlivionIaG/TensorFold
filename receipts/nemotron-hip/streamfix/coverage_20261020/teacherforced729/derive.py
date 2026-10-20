import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('raw', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
D = a.output
assert D.is_dir() and not (D / 'result.json').exists()
OLD = '8649e9858b38e9ef3dcd8d579879c6512a363050'
NEW = '729833615595df66c9f9a438ce9a85061843f010'
VOCAB = 131072
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()

def top2(x):
    first = np.argmax(x, axis=1)
    scratch = x.copy()
    scratch[np.arange(len(x)), first] = -np.inf
    second = np.argmax(scratch, axis=1)
    one = x[np.arange(len(x)), first].astype(np.float64)
    two = x[np.arange(len(x)), second].astype(np.float64)
    return first, second, one, two

result = dict(schema_version=1, baseline_source=OLD, candidate_source=NEW,
              review_top='60a8da5243ef313fcab23bdb0d85c790a5cda935',
              scope='CPU-only derivation of retained same-history baseline/current model logits. Not an independent FP64/MLX/HF oracle or a new inference run.',
              dtype='<f4', vocabulary=VOCAB,
              nonfinite_policy='Count and refuse any NaN/Inf; no replacement, masking or NaN-equality exemption. Raw-byte comparison preserves signed zero and NaN payload distinctions.',
              tie_policy='First/lower token index wins exact equal maxima, matching numpy.argmax; second is the first/lower remaining index. No tolerance or tie waiver.',
              tool=dict(python=sys.version, executable=sys.executable, numpy=np.__version__, platform=platform.platform(), script_sha256=sha(Path(__file__))),
              arrays=[], comparisons=[], validated=False)
for folder in sorted((a.raw / 'logs').glob('nem_*')):
    summary_path = folder / 'summary.json'
    s = json.loads(summary_path.read_text())
    assert s['done'] and s['checkpoint_revision'] == 'd9d758fb83953437f7263256b0d96157e2a348b8'
    by = {(r['tree'], r['name']): r for r in s['runs']}
    for name in ('decode', 'chat_full', 'code_full', 'story_full'):
        old, new = by['d46', name], by['sfopt', name]
        assert old['source_commit'] == OLD and new['source_commit'] == NEW
        assert old['input_sha256'] == new['input_sha256']
        rank_count = len(old['ranks'])
        assert rank_count == len(new['ranks']) and rank_count in (1, 2)
        for rank in range(rank_count):
            suffix = '' if rank == 0 else '.r' + str(rank)
            paths = []
            records = []
            for run in (old, new):
                q = next(q for q in run['ranks'] if q['rank'] == rank)
                assert q['exit_status'] == 0
                path = folder / (Path(run['output']['path']).name + suffix)
                assert path.stat().st_size == run['output']['bytes'] and path.stat().st_size % (VOCAB * 4) == 0
                digest = sha(path)
                if rank == 0:
                    assert digest == run['output']['sha256']
                log = folder / q['log']
                assert f'rank {rank} of {rank_count}' in log.read_text()
                identity = folder / Path(run['runtime_identity']['file']).name
                assert sha(identity) == run['runtime_identity']['sha256']
                record = dict(rig=s['rig'], case=name, rank=rank, source_commit=run['source_commit'],
                              file=str(path.relative_to(a.raw)), sha256=digest, bytes=path.stat().st_size,
                              shape=[path.stat().st_size // (VOCAB * 4), VOCAB],
                              primary_output_digest_previously_recorded=(rank == 0),
                              rank_log_sha256=sha(log), summary_sha256=sha(summary_path),
                              test_binary_sha256=run['binary_sha256'], input_sha256=run['input_sha256'],
                              runtime_identity_sha256=sha(identity))
                result['arrays'].append(record)
                records.append(record)
                paths.append(path)
            x, y = [np.memmap(p, dtype='<f4', mode='r').reshape(-1, VOCAB) for p in paths]
            rows = len(x)
            max_abs = np.zeros(VOCAB, dtype=np.float64)
            total_abs = np.zeros(VOCAB, dtype=np.float64)
            total_sq = np.zeros(VOCAB, dtype=np.float64)
            bitdiff = valuesdiff = nonfinite = ties_old = ties_new = 0
            row_records = []
            for start in range(0, rows, 16):
                b, c = np.array(x[start:start + 16]), np.array(y[start:start + 16])
                nonfinite += int(np.count_nonzero(~np.isfinite(b)) + np.count_nonzero(~np.isfinite(c)))
                assert nonfinite == 0
                bitdiff += int(np.count_nonzero(b.view(np.uint32) != c.view(np.uint32)))
                delta = c.astype(np.float64) - b.astype(np.float64)
                valuesdiff += int(np.count_nonzero(delta))
                absolute = np.abs(delta)
                max_abs = np.maximum(max_abs, absolute.max(axis=0))
                total_abs += absolute.sum(axis=0)
                total_sq += np.square(delta).sum(axis=0)
                bo, bs, bv, bz = top2(b)
                co, cs, cv, cz = top2(c)
                for i in range(len(b)):
                    row_records.append(dict(row=start+i, baseline_top1=int(bo[i]), baseline_top2=int(bs[i]),
                                            candidate_top1=int(co[i]), candidate_top2=int(cs[i]),
                                            baseline_top1_logit=float(bv[i]), baseline_top2_logit=float(bz[i]),
                                            candidate_top1_logit=float(cv[i]), candidate_top2_logit=float(cz[i]),
                                            baseline_margin=float(bv[i]-bz[i]), candidate_margin=float(cv[i]-cz[i]),
                                            top1_changed=bool(bo[i] != co[i]), top2_changed=bool(bs[i] != cs[i])))
                ties_old += int(np.count_nonzero(bv == bz))
                ties_new += int(np.count_nonzero(cv == cz))
            stem = s['rig'] + '_' + name + '_r' + str(rank)
            rp = D / (stem + '_rows.json')
            rp.write_text(json.dumps(row_records, indent=2, allow_nan=False) + '\n')
            cp = D / (stem + '_columns.npz')
            np.savez_compressed(cp, max_abs_error=max_abs, mean_abs_error=total_abs/rows, rmse_error=np.sqrt(total_sq/rows), compared_rows=np.array(rows))
            comparison = dict(rig=s['rig'], case=name, rank=rank, shape=[rows, VOCAB],
                              baseline=records[0]['sha256'], candidate=records[1]['sha256'],
                              byte_equal=records[0]['sha256'] == records[1]['sha256'],
                              different_float_bits=bitdiff, different_finite_values=valuesdiff, nonfinite_values=nonfinite,
                              top1_differences=[r for r in row_records if r['top1_changed']],
                              top2_differences=[r for r in row_records if r['top2_changed']],
                              baseline_exact_top_ties=ties_old, candidate_exact_top_ties=ties_new,
                              min_baseline_margin=min(r['baseline_margin'] for r in row_records),
                              min_candidate_margin=min(r['candidate_margin'] for r in row_records),
                              max_abs_error=float(max_abs.max()), mean_abs_error=float(total_abs.sum()/(rows*VOCAB)),
                              rmse_error=float(np.sqrt(total_sq.sum()/(rows*VOCAB))),
                              per_row_artifact=rp.name, per_row_sha256=sha(rp), per_column_artifact=cp.name, per_column_sha256=sha(cp))
            assert comparison['byte_equal'] and bitdiff == 0 and valuesdiff == 0
            result['comparisons'].append(comparison)
assert len(result['arrays']) == 24 and len(result['comparisons']) == 12
result['validated'] = True
(D / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps(dict(validated=True, arrays=len(result['arrays']), comparisons=len(result['comparisons']), compared_rows=sum(q['shape'][0] for q in result['comparisons']), result_sha256=sha(D/'result.json'))))
