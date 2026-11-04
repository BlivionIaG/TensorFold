import hashlib
import json
import os
from pathlib import Path
import platform
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check_source(root, manifest):
    root = Path(root)
    for name, digest in manifest['source_files_sha256'].items():
        require(sha(root / name) == digest, 'Frozen source changed: ' + name)
    for name, target in manifest['source_symlinks'].items():
        require((root / name).is_symlink() and os.readlink(root / name) == target,
                'Frozen source symlink changed: ' + name)


def check_packet(root, manifest):
    for name, digest in manifest['packet_files_sha256'].items():
        require(sha(Path(root) / name) == digest, 'Frozen packet changed: ' + name)


def check_python(manifest):
    expected = manifest['python_runtime']
    require(platform.python_version() == expected['version'], 'Python version differs from declared runtime')
    require(Path(sys.executable).resolve() == Path(expected['executable']).resolve(),
            'Python executable differs from declared runtime')
    require(sha(sys.executable) == expected['sha256'], 'Python binary differs from declared runtime')
