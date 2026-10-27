"""Paths and strict serialization shared by Step 6 collection and verification."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/step6_smolvlm2_refined512'
REFERENCE = ROOT / 'runs/step4_smolvlm2_refined512/reference'
DATA = Path('/data/team_a/dave-workspace/RoleVLN/data/processed/baseline_v0/scorer_labels_smolvlm2_refined512_v1')
ACTION_NAMES = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def job_seed(group, dataset, mode, eid):
    key = f'6/{group}/{dataset}/{mode}/{eid}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], 'little') % (2 ** 31)


def verify_protocol():
    from policy.experimental_reference import verify_bundle
    from sim.config import load_manifest
    protocol = read(RUN / 'protocol.json')
    ref = verify_bundle(REFERENCE)
    assert ref['policy_identity_sha256'] == protocol['policy_identity_sha256']
    assert ref['decoder'] == protocol['decoder']
    assert load_manifest()['simulator'] == protocol['simulator']
    for name, expected in protocol['source_sha256'].items():
        assert sha(ROOT / name) == expected, ('Source changed', name)
    for name, expected in protocol['input_sha256'].items():
        assert sha(name) == expected, ('Input changed', name)
    assert sha(RUN / 'selection.json') == protocol['selection_sha256']
    return protocol
