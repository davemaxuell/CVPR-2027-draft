"""Four-model evaluation identities and deterministic, outcome-free sample seeds."""
import hashlib
from scorer_four.common import (ROOT, BASE_RUN, BASE_CACHE, FEATURES_RUN, CACHE, NWM_RUN,
    ADDITIONS, TRIALS, POLICY_ID, ENCODER_ID, now, read, sha, torch_save, write)
from scorer_four.training_common import RUN as TRAIN

CALIBRATION = ROOT / 'runs/step9b_smolvlm2_refined512_four_models_v1'
EVALUATION = ROOT / 'runs/step10b_smolvlm2_refined512_four_models_v1'
PRIOR_CALIBRATION = ROOT / 'runs/step9_smolvlm2_refined512_dino_pair_v1'
PRIOR_EVALUATION = ROOT / 'runs/step10_smolvlm2_refined512_dino_pair_v1'


def sample_seeds(root_id):
    prefix = 'NWM_CE_v1/5/' + root_id
    return {kind + '_seed': int.from_bytes(hashlib.sha256((prefix + '/' + kind).encode()).digest()[:8], 'little') % (2**63)
            for kind in ('context', 'diffusion')}
