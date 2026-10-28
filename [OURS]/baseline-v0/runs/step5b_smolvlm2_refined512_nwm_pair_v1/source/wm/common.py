"""Artifact identities and atomic records for world-model training."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0')
RUN = ROOT / 'runs/step5_smolvlm2_refined512'
CORPUS = DATA / 'wm_groupA_smolvlm2_refined512_v1'
REFERENCE = ROOT / 'runs/step4_smolvlm2_refined512/reference'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()
