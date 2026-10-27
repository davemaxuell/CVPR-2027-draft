"""Check new frozen inputs, all-model updates, exact restart, and inherited warm-up."""
import copy
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from Scorer.scorer import NavigationScorer
from scorer_four.common import CACHE as EXTRA, read, sha, torch_save, write, now
from scorer_four.data import ScorerData
from scorer_four.evaluate import evaluate
from scorer_four.train import schedule, update
from scorer_four.training_common import CACHE, RUN, flags
from scorer_training.train import schedule as parent_schedule


def main():
    flags(); recipe = read(RUN / 'recipe.json')
    for epoch in range(16):
        stage, order, reps = schedule(epoch, recipe); assert order == parent_schedule(epoch, recipe)[1]
        counts = {-1:1024} if stage == 'actual' else {-1:512,0:128,1:128,2:128,3:128} if stage == 'mixed' else {0:256,1:256,2:256,3:256}
        assert dict(Counter(reps)) == counts
    torch.manual_seed(8); data = ScorerData()
    assert data.groups['B'] == list(range(1024)) and data.groups['C'] == list(range(1024,1280))
    ids = torch.arange(1280, device='cuda'); stop = data.pooled[data.actual_ids[:,3]]
    for model in range(4): assert torch.equal(data.predictions[model,:,3], stop)
    assert torch.isfinite(data.predictions).all() and not data.predictions.requires_grad
    scorer = NavigationScorer.from_config(CACHE / 'scorer_config.json', 'smolvlm2').cuda().train()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=.01)
    # A complete actual-only epoch must agree with the previously executed fit.
    _, order, reps = schedule(0, recipe)
    started = time.perf_counter(); times = []
    for start in range(0,1024,8):
        tick = time.perf_counter(); update(scorer, optimizer, data, order[start:start+8], reps[start:start+8], 'pooled')
        torch.cuda.synchronize(); times.append(time.perf_counter() - tick)
    parent = torch.load(Path(recipe['DINO_training_run']) / 'pooled/checkpoints/step_000128.pt', map_location='cpu', weights_only=True)
    for name, value in scorer.state_dict().items(): assert torch.equal(value.detach().cpu(), parent['model'][name]), name
    del parent
    stage, order, reps = schedule(8, recipe)
    indices = order[:8]; representations = [0,1,2,3,0,1,2,3]
    update(scorer, optimizer, data, indices, representations, 'pooled')
    path = RUN / 'diagnostics/restart.pt'
    torch_save(path, {'model': scorer.state_dict(), 'optimizer': optimizer.state_dict(),
        'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
    expected = update(scorer, optimizer, data, indices, representations, 'pooled')
    state = {name: value.detach().cpu().clone() for name, value in scorer.state_dict().items()}
    saved = torch.load(path, map_location='cpu', weights_only=True)
    scorer.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    actual = update(scorer, optimizer, data, indices, representations, 'pooled')
    assert actual == expected
    for name, value in scorer.state_dict().items(): assert torch.equal(value.cpu(), state[name]), name
    before = {k: v.detach().cpu().clone() for k,v in scorer.state_dict().items()}
    validation = evaluate(scorer, data)
    assert set(validation['representations']) == {'actual','dinowm_transfer','dinov2_fresh_dynamics','nwm_transfer','cdit_fresh_dynamics'}
    assert all(len(rows) == 256 for rows in validation['rows'].values())
    for name, value in scorer.state_dict().items(): assert torch.equal(value.cpu(), before[name]), name
    report = {'status': 'passed', 'completed_utc': now(), 'recipe_sha256': sha(RUN / 'recipe.json'),
        'feature_manifest_sha256': sha(EXTRA / 'manifest.json'), 'all_stages_root_order_and_balance_verified': True,
        'first_actual_only_epoch_bit_exact_to_DINO_trial': True, 'all_model_update_and_optimizer_resume_bit_exact': True,
        'C_evaluation_all_four_models_and_actual_preserves_weights': True, 'all_STOP_features_current_identity': True,
        'B_fitting_only': True, 'no_E_metrics_used': True, 'seconds': time.perf_counter() - started,
        'steady_seconds_per_update': sum(times[8:]) / len(times[8:]),
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'scorer_four').glob('*.py')) + [Path(__file__).resolve()]}}
    write(RUN / 'check_training.json', report); print(report, flush=True)


if __name__ == '__main__':
    main()
