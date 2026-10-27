"""A versioned extension of immutable Step 7 features, with no duplicate labels."""
from scorer_features.common import (ROOT, DATA_ROOT, LABELS, POLICY_ID, ENCODER_ID,
    CACHE as BASE_CACHE, RUN as BASE_RUN, now, read, sha, torch_save, write,
    verify_protocol as verify_base_protocol)
from nwm_training.common import RUN as NWM_RUN

FEATURES_RUN = ROOT / 'runs/step7b_smolvlm2_refined512_four_models_v1'
CACHE = DATA_ROOT / 'scorer_features_smolvlm2_refined512_four_models_v1'
TRIALS = ('dinowm_transfer','dinov2_fresh_dynamics','nwm_transfer','cdit_fresh_dynamics')
ADDITIONS = TRIALS[2:]


def verify_features_protocol():
    verify_base_protocol()
    protocol = read(FEATURES_RUN / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == sha(FEATURES_RUN / 'source' / name) == digest, name
    for name, digest in protocol['input_sha256'].items(): assert sha(name) == digest, name
    assert protocol['policy_identity_sha256'] == POLICY_ID and protocol['encoder_identity_sha256'] == ENCODER_ID
    return protocol
