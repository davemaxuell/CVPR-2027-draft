import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/step7_smolvlm2_refined512_dino_pair_v1'
UPSTREAM = ROOT / 'runs/step6_smolvlm2_refined512'
REFERENCE = ROOT / 'runs/step4_smolvlm2_refined512/reference'
FROZEN = ROOT / 'runs/step5_smolvlm2_refined512/frozen'
DATA_ROOT = ROOT.parents[1] / 'data/processed/baseline_v0'
LABELS = DATA_ROOT / 'scorer_labels_smolvlm2_refined512_v1'
CACHE = DATA_ROOT / 'scorer_features_smolvlm2_refined512_dino_pair_v1'
TRIALS = ('dinowm_transfer', 'dinov2_fresh_dynamics')
POLICY_ID = '729480bf5c8402b07cad653a7ef0f7124be212d50255077290316ba686777ca7'
ENCODER_ID = '072c391c8764fbdcdb9e00bfbf9ec016873fa9707c573e400f44ff385f01f9dd'
LIVE_TO_CANONICAL = (3, 0, 1, 2)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def torch_save(path, value):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def verify_index():
    m = read(RUN / 'index/manifest.json')
    for name, expected in m['files'].items():
        assert sha(RUN / 'index' / name) == expected, name
    assert sha(LABELS / 'manifest.json') == m['upstream_manifest_sha256']
    assert sha(UPSTREAM / 'result.json') == m['upstream_result_sha256']
    assert sha(UPSTREAM / 'full/verification.json') == m['upstream_verification_sha256']
    for group, expected in m['upstream_catalog_sha256'].items():
        assert sha(LABELS / f'catalog_{group}.jsonl') == expected
    return m


def verify_protocol():
    p = read(RUN / 'protocol.json')
    assert p['policy_identity_sha256'] == POLICY_ID and p['encoder_identity_sha256'] == ENCODER_ID
    assert sha(RUN / 'index/manifest.json') == p['index_manifest_sha256']
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT / name) == sha(RUN / 'source' / name) == expected, name
    for name, expected in p['native_source_sha256'].items():
        assert sha(name) == expected, name
    verify_index()
    return p


def native_flags():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    return {'matmul_tf32': False, 'cudnn_tf32': True, 'cudnn_benchmark': False,
            'deterministic_algorithms': False, 'policy_batch_size': 1}
