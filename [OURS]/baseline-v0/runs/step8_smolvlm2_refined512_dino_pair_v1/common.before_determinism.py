from pathlib import Path
import torch

from scorer_features.common import (ROOT, CACHE, RUN as FEATURES_RUN, POLICY_ID, ENCODER_ID,
                                    TRIALS, read, sha, write, now, torch_save, verify_protocol as verify_features)

RUN = ROOT / 'runs/step8_smolvlm2_refined512_dino_pair_v1'


def flags():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    # Use the same attention implementation with and without autograd.
    torch.backends.mha.set_fastpath_enabled(False)
    return {'dtype': 'float32', 'matmul_tf32': False, 'cudnn_tf32': True,
            'cudnn_benchmark': False, 'mha_fastpath': False, 'threads': 1}


def verify_inputs():
    verify_features()
    m=read(CACHE/'manifest.json')
    assert m['status']=='complete' and m['policy_identity_sha256']==POLICY_ID
    assert sha(FEATURES_RUN/'verification.json')==m['verification_sha256']
    assert read(FEATURES_RUN/'verification.json')['status']=='passed'
    assert not m['trainable_memory_cached'] and not m['scorer_trained']
    return m


def verify_protocol():
    p=read(RUN/'protocol.json');m=verify_inputs()
    assert sha(CACHE/'manifest.json')==p['feature_manifest_sha256']
    for name,digest in p['source_sha256'].items():
        assert sha(ROOT/name)==sha(RUN/'source'/name)==digest,name
    return p
