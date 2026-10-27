"""Organize and validate assignment 05 artifacts; no fetched project execution."""
from pathlib import Path
import csv, hashlib, json, shutil

R = Path(__file__).resolve().parents[2]
E = R / 'evidence/05'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Preserve originals while making every reviewed follow-up available in the
# common full-text directory requested in the assignment.
copied = []
for source in sorted((E / 'sources').glob('*')):
    if source.suffix not in {'.pdf', '.txt', '.json'}:
        continue
    target = R / 'followups/sources' / source.name
    if target.exists():
        if sha(source) != sha(target):
            raise RuntimeError(f'Existing shared source differs: {target}')
    else:
        shutil.copy2(source, target)
        copied.append(str(target.relative_to(R)))
(E / 'source_organization.json').write_text(json.dumps({
    'date': '2026-10-22', 'action': 'copy, preserving original evidence files',
    'new_shared_files': copied, 'collision_policy': 'abort on differing existing file'
}, ensure_ascii=False, indent=2), encoding='utf-8')

print(f'Organized {len(copied)} additional shared source files. Run builders, then validate_artifacts.py.')
