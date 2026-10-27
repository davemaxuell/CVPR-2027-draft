"""Paths and identities for the matched NWM/CDiT repeat."""
from pathlib import Path

from wm.common import DATA, ROOT, now, read, sha, write

RUN = ROOT / 'runs/step5b_smolvlm2_refined512_nwm_pair_v1'
INDEX = ROOT / 'runs/step5_smolvlm2_refined512/data_index'
CACHE = DATA / 'wm_vae_posteriors_refined512_v1'
POLICY_ID = '729480bf5c8402b07cad653a7ef0f7124be212d50255077290316ba686777ca7'
NWM_REVISION = '0821a1a7b1ae938539e32f13f5ad82465e0f3fde'
NWM_FILE = Path('/data/shared/hf_cache/hub/models--facebook--nwm/snapshots') / NWM_REVISION / 'cdit_s_100000.pth.tar'
NWM_SHA = '8ed9478d40e49809a1bbe6aec4df8af7f6f48c3bb427307dd8a423efb347ce6e'
VAE_REVISION = 'f04b2c4b98319346dad8c65879f680b1997b204a'
VAE_DIRECTORY = Path('/data/shared/hf_cache/hub/models--stabilityai--sd-vae-ft-ema/snapshots') / VAE_REVISION


def verify_index():
    manifest = read(INDEX / 'manifest.json')
    assert manifest['status'] == 'complete'
    assert manifest['selected_policy_identity_sha256'] == POLICY_ID
    assert manifest['training_group'] == 'A' and manifest['validation_group'] == 'C'
    assert manifest['train_samples'] == 2422469 and manifest['validation_samples'] == 3072
    for name, digest in manifest['files'].items():
        assert sha(INDEX / name) == digest, name
    return manifest


def verify_assets():
    record = read(RUN / 'assets/selection.json')
    assert record['status'] == 'verified_cpu'
    assert record['selected_policy_identity_sha256'] == POLICY_ID
    for path, digest in record['files_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in record['source_sha256'].items():
        assert sha(ROOT / path) == digest, path
    return record
