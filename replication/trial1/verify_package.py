#!/usr/bin/env python3
"""Verify export integrity and Python/JSON syntax without imports or GPU access."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'FILE_MANIFEST.json').read_text())
    expected = manifest['files']
    actual = {
        p.relative_to(root).as_posix() for p in root.rglob('*')
        if p.is_file() and p.name != 'FILE_MANIFEST.json'
        and '__pycache__' not in p.parts and '.git' not in p.parts
    }
    if actual != set(expected):
        raise ValueError(f'File inventory differs: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}')
    counts = {'files': 0, 'bytes': 0, 'python_files': 0, 'json_files': 0}
    for name, identity in sorted(expected.items()):
        p = root / name
        if p.is_symlink() or root not in p.resolve().parents:
            raise ValueError('Unexpected symlink/path: ' + name)
        data = p.read_bytes()
        if len(data) != identity['bytes'] or hashlib.sha256(data).hexdigest() != identity['sha256']:
            raise ValueError('File integrity failed: ' + name)
        source = data.decode('utf-8')
        if p.suffix == '.py':
            ast.parse(source, filename=name)
            counts['python_files'] += 1
        elif p.suffix == '.json':
            json.loads(source)
            counts['json_files'] += 1
        counts['files'] += 1
        counts['bytes'] += len(data)
    origins = json.loads((root / 'SOURCE_ORIGINS.json').read_text())['files']
    for name, identity in origins.items():
        if expected[name]['sha256'] != identity['sha256']:
            raise ValueError('Reference source was modified: ' + name)
    return {'status': 'passed', **counts, 'exact_reference_files': len(origins),
            'gpu_used': False, 'scope': 'File integrity and static syntax; GPU/model qualification is required separately.'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    print(json.dumps(verify(args.root), indent=2))
