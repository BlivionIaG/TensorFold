#!/usr/bin/env python3
"""Replace only stream usage arrays from the contract limits tool's actual RECORD dump."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

FIELDS = ('vgprs', 'sgprs', 'lds', 'private', 'bound', 'wave')
WIDTHS = (2, 3, 4, 5, 6, 8)
SHAPES = ((1, 2), (1, 4), (1, 8), (2, 2), (2, 4), (2, 8), (4, 2), (4, 4))
CHIPS = ('gfx1030', 'gfx1100', 'gfx1151', 'gfx1201')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def usage_end(text, start):
    if text.startswith('@splat(', start):
        opening, closing, pos = '(', ')', text.index('(', start)
    else:
        require(text.startswith('.{', start), 'unsupported usage expression')
        opening, closing, pos = '{', '}', start + 1
    depth = 0
    for i in range(pos, len(text)):
        depth += text[i] == opening
        depth -= text[i] == closing
        if depth == 0:
            require(text[i + 1] == ',', 'usage initializer has no trailing comma')
            return i + 1
    raise ValueError('unterminated usage array')


def parse_record(line):
    match = re.fullmatch(r'RECORD (.+): \.usage = \.\{ (.*)\}', line)
    if not match:
        return None
    name, body = match.groups()
    fields = re.findall(r'\.\{([^{}]+)\}', body)
    require(len(fields) == len(CHIPS), 'record lacks four chip rows: ' + name)
    rows = []
    for field in fields:
        items = re.findall(r'\.(\w+) = (\d+)', field)
        require(tuple(key for key, _ in items) == FIELDS, 'unexpected usage fields: ' + name)
        row = {key: int(value) for key, value in items}
        require(row['vgprs'] > 0 and row['bound'] > 0 and row['wave'] == 32, 'invalid compiled usage: ' + name)
        rows.append(row)
    return name, rows


def replace_arrays(text, launch_pattern, name_of, actual):
    changes = []
    for match in re.finditer(r'(\.launch = ' + launch_pattern + r',\s*\.usage = )', text):
        name = name_of(match)
        require(name in actual, 'missing RECORD for ' + name)
        start = match.end()
        end = usage_end(text, start)
        rows = ['.{ ' + ', '.join('.' + key + ' = ' + str(row[key]) for key in FIELDS) + ' },'
                for row in actual[name]]
        replacement = '.{\n' + ''.join('            ' + row + '\n' for row in rows) + '        }'
        changes.append((start, end, replacement, name))
    for start, end, replacement, _ in reversed(changes):
        text = text[:start] + replacement + text[end:]
    return text, [name for _, _, _, name in changes]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dump', type=Path)
    ap.add_argument('tree', type=Path)
    ap.add_argument('--include-relu', action='store_true', help='also refresh existing #41/#46 relu2 records')
    ap.add_argument('--write', action='store_true', help='otherwise validate and report without editing')
    ap.add_argument('--source-commit', help='full actual build-source commit; required for --write')
    ap.add_argument('--manifest', type=Path, help='optional new local provenance JSON; must not exist')
    args = ap.parse_args()
    if args.write:
        require(re.fullmatch(r'[0-9a-f]{40}', args.source_commit or ''), 'full source commit required for write')
        head = subprocess.check_output(['git', '-C', str(args.tree), 'rev-parse', 'HEAD'], text=True).strip()
        require(head == args.source_commit, 'tree HEAD differs from declared metadata build source')
    require(args.manifest is None or not args.manifest.exists(), 'manifest must be a new path')
    raw = args.dump.read_bytes()
    actual = {}
    for line in raw.decode().splitlines():
        record = parse_record(line)
        if record:
            name, rows = record
            require(name not in actual, 'duplicate RECORD; do not concatenate builds: ' + name)
            actual[name] = rows
    plans = []
    root = args.tree / 'zig/src/core/contract'
    for mode in ('plain', 'swiglu'):
        path = root / f'hip_records_stream_{mode}.zig'
        before = path.read_text()
        pattern = r'tile\((\d+), (\d+), (\d+), ' + ('true' if mode == 'swiglu' else 'false') + r'\)'
        after, names = replace_arrays(before, pattern,
            lambda m: f'{mode} stream tile {m[2]}-bit {m[3]}x{m[4]}', actual)
        expected = {f'{mode} stream tile {bits}-bit {rows}x{cb}' for bits in WIDTHS for rows, cb in SHAPES}
        require(set(names) == expected and len(names) == len(expected), 'stream declaration inventory differs')
        plans.append((path, before, after, names))
    if args.include_relu:
        for file, function, prefix in [('hip_records.zig', 'relu2Tile', 'relu2 stream tile'),
                                       ('hip_records_pair.zig', 'pairTile', 'relu2 pair tile')]:
            path = root / file
            before = path.read_text()
            after, names = replace_arrays(before, function + r'\((\d+), (\d+)\)',
                lambda m: f'{prefix} 4-bit {m[2]}x{m[3]}', actual)
            require(set(names) == {f'{prefix} 4-bit {rows}x{cb}' for rows, cb in SHAPES}
                    and len(names) == len(SHAPES), 'relu declaration inventory differs')
            plans.append((path, before, after, names))
    # Every selected name must have an actual per-chip diagnostic as well as its emitted RECORD row.
    for _, _, _, names in plans:
        for name in names:
            for chip in CHIPS:
                require(re.search(r'^(?:PASS|FAIL) limits ' + re.escape(chip + ' ' + name) + ':', raw.decode(), re.M),
                        'missing per-chip object audit diagnostic for ' + name + ' ' + chip)
    report = {'schema_version': 1, 'artifact_kind': 'resource_metadata_hydration_only',
              'dump_sha256': sha(raw), 'declared_build_source_commit': args.source_commit,
              'chip_order': CHIPS, 'qualification_claim': False, 'written': args.write, 'files': []}
    for path, before, after, names in plans:
        report['files'].append({'path': str(path.relative_to(args.tree)), 'records': len(names),
                                'before_sha256': sha(before.encode()), 'after_sha256': sha(after.encode())})
    if args.write:
        for path, _, after, _ in plans:
            path.write_text(after)
    if args.manifest:
        args.manifest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        sys.exit('FAIL: ' + str(exc))
