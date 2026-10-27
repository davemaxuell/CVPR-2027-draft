"""Matched scorer repeat identities; reuse the native, already checked math."""
from scorer_training.common import flags
from scorer_four.common import (ROOT, BASE_CACHE as CACHE, CACHE as EXTENSION_CACHE, FEATURES_RUN,
    POLICY_ID, TRIALS, now, read, sha, torch_save, write, verify_features_protocol)

RUN = ROOT / 'runs/step8b_smolvlm2_refined512_four_models_v1'


def verify_protocol():
    verify_features_protocol()
    protocol = read(RUN / 'protocol.json')
    assert read(EXTENSION_CACHE / 'manifest.json')['status'] == 'complete'
    for path, digest in protocol['input_sha256'].items(): assert sha(path) == digest, path
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == sha(RUN / 'source' / name) == digest, name
    return protocol
