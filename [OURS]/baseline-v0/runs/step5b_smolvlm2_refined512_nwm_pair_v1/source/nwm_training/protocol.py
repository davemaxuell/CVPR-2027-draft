"""Locked repeat settings and source verification."""
from nwm_training.common import CACHE, INDEX, POLICY_ID, ROOT, RUN, read, sha


def verify_protocol():
    plan = read(RUN / 'protocol.json')
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT / name) == sha(RUN / 'source' / name) == digest, name
    for path, digest in plan['input_sha256'].items():
        assert sha(path) == digest, path
    return plan


def verify_cache():
    record = read(CACHE / 'manifest.json'); assert record['status'] == 'complete'
    assert sha(CACHE / 'config.json') == record['config_sha256']
    config = read(CACHE / 'config.json')
    assert config['selected_policy_identity_sha256'] == POLICY_ID
    assert config['index_manifest_sha256'] == sha(INDEX / 'manifest.json')
    for name, digest in config['source_sha256'].items(): assert sha(ROOT / name) == digest, name
    assert sha(CACHE / 'posteriors.npy') == record['posteriors_sha256']
    return record
