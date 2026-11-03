import argparse, json, math, re, struct
from pathlib import Path


def value(text):
    text = text.strip()
    if text == 'null':
        return None
    if text in ('true', 'false'):
        return text == 'true'
    if text.startswith('{') and text.endswith('}'):
        body = text[1:-1].strip()
        return [] if not body else [value(v) for v in body.split(',')]
    try:
        return int(text)
    except ValueError:
        try:
            return float(text)
        except ValueError:
            return text


def fields(line):
    return {m[1]: value(m[2]) for m in re.finditer(r'(\w+)=(.*?)(?=\s+\w+=|$)', line)}


def number(bits):
    return None if bits is None else struct.unpack('<d', struct.pack('<Q', bits))[0]


def parse_log(path):
    result = {'copy_phases': {}, 'startup': None, 'startup_prices': {}, 'shape_snapshots': [], 'gpu_phases': [], 'gpu_init': []}
    current_shape = {}
    current_gpu = []
    snapshot = None
    with path.open() as stream:
        for line in stream:
            if 'DSPARK_PRIVATE_COPY_' in line:
                text = line[line.index('DSPARK_PRIVATE_COPY_'):].strip()
                kind = text.split()[0].removeprefix('DSPARK_PRIVATE_COPY_')
                record = fields(text)
                if kind == 'STARTUP':
                    result['startup'] = record
                elif kind == 'STARTUP_PRICE':
                    result['startup_prices'][record['width']] = record
                elif kind == 'PHASE':
                    assert record['phase'] not in result['copy_phases']
                    result['copy_phases'][record['phase']] = {'header': record, 'request': [], 'plan': [], 'allocation': [], 'decision': [], 'overhead': [], 'prices': []}
                else:
                    assert record['phase'] in result['copy_phases']
                    result['copy_phases'][record['phase']][kind.lower()].append(record)
            elif 'LFM2_PRIVATE_TARGET_GPU_INIT ' in line:
                result['gpu_init'].append(fields(line[line.index('LFM2_PRIVATE_TARGET_GPU_INIT '):]))
            elif 'LFM2_PRIVATE_TARGET_GPU ' in line:
                record = fields(line[line.index('LFM2_PRIVATE_TARGET_GPU '):])
                assert record['invalid'] is False, record
                assert record['cpu_delta_ns'] > 0 and record['gpu_delta_ticks'] > 0
                assert 0 < record['counter_gpu_ms'] <= record['cpu_delta_ns'] / 1e6
                record['command_gpu_ms_valid'] = math.isfinite(record['command_gpu_ms']) and 0 < record['command_gpu_ms'] <= record['cpu_delta_ns'] / 1e6
                current_gpu.append(record)
            elif 'DSPARK_PRIVATE_SNAPSHOT ' in line:
                assert snapshot is None, 'previous shape snapshot not finished'
                snapshot = fields(line[line.index('DSPARK_PRIVATE_SNAPSHOT '):])['index']
                result['gpu_phases'].append({'snapshot': snapshot, 'calls': current_gpu})
                current_gpu = []
            elif 'DSPARK_PRIVATE_TARGET_SHAPE ' in line:
                record = fields(line[line.index('DSPARK_PRIVATE_TARGET_SHAPE '):])
                key = (record['phase'], record['rows'])
                if snapshot is None:
                    assert current_shape.get(key) == record, 'unanchored report changed target counters'
                else:
                    current_shape[key] = record
            elif 'DSPARK_PRIVATE_ROW_COUNTS phase=decode ' in line and snapshot is not None:
                result['shape_snapshots'].append({'snapshot': snapshot, 'shapes': dict(current_shape)})
                snapshot = None
    assert snapshot is None and not current_gpu
    return result


def summarize(root):
    observed = json.loads((root / 'observed.json').read_text())
    output = {'binding': observed['binding'], 'arms': []}
    for arm in observed['records']:
        name = arm['arm']
        log = root / name / 'private-logs' / ('exact-history-' + name + '.server.log')
        parsed = parse_log(log)
        assert len(parsed['shape_snapshots']) == len(parsed['gpu_phases']) == len(arm['phases']) == 46
        summaries = []
        previous = {}
        for phase, shape_snapshot, gpu_phase in zip(arm['phases'], parsed['shape_snapshots'], parsed['gpu_phases'], strict=True):
            current = shape_snapshot['shapes']
            delta = {}
            for key, row in current.items():
                delta[key] = {field: row[field] - previous.get(key, {}).get(field, 0) for field in ('calls', 'captured_calls', 'encode_cpu_ms', 'gpu_ms', 'timed_gpu_calls')}
            previous = current
            if phase['phase'] == 'measured':
                copy = next((v for v in parsed['copy_phases'].values() if v['header']['snapshot'] == shape_snapshot['snapshot']), None)
                summaries.append({'cell': phase['cell'], 'snapshot': shape_snapshot['snapshot'], 'activity': phase['activity'], 'enrollments': phase['enrollments'], 'target_shapes': [{'phase': key[0], 'rows': key[1], **values} for key, values in delta.items() if values['calls']], 'copy_header': copy['header'] if copy else None, 'copy_decisions': len(copy['decision']) if copy else 0, 'gpu_counter_calls': len(gpu_phase['calls']), 'invalid_command_stopwatches': sum(not v['command_gpu_ms_valid'] for v in gpu_phase['calls'])})
        output['arms'].append({'arm': name, 'startup': parsed['startup'], 'startup_prices': parsed['startup_prices'], 'measured': summaries})
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'arms': [{'arm': r['arm'], 'measured': len(r['measured'])} for r in result['arms']], 'output': str(args.output)}))
