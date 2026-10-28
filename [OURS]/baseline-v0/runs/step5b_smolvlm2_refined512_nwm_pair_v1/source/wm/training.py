"""Matched fixed-order dynamics updates, held-out evaluation and exact restart state."""
import math
import os
import random
from pathlib import Path

import numpy as np
import torch


def seed_training(seed):
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def learning_rate(step, total, peak, warmup):
    if step <= warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return peak * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def update(model, optimizer, batch, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr
    x = batch['features'].cuda(non_blocking=True)
    actions = batch['actions'].cuda(non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(x[:, :3], actions[:, :2], actions[:, 2])
        loss = (prediction.float() - x[:, 3].float()).square().mean()
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite dynamics loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), norm.item()


@torch.inference_mode()
def evaluate(model, loader, *, command_diagnostics=False):
    was_training = model.training
    model.eval()
    records = []
    for batch in loader:
        x = batch['features'].cuda(non_blocking=True)
        actions = batch['actions'].cuda(non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            pred = model(x[:, :3], actions[:, :2], actions[:, 2])
        errors = (pred.float() - x[:, 3].float()).square().mean((1, 2))
        persistence = (x[:, 2].float() - x[:, 3].float()).square().mean((1, 2))
        assert torch.isfinite(errors).all()
        diagnostics = None
        if command_diagnostics:
            alternatives = []
            for command in range(3):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    alt = model(x[:, :3], actions[:, :2], torch.full_like(actions[:, 2], command))
                alternatives.append((alt.float() - x[:, 3].float()).square().mean((1, 2)))
            diagnostics = torch.stack(alternatives, dim=1).cpu().numpy()
        for i, (error, baseline) in enumerate(zip(errors.cpu().tolist(), persistence.cpu().tolist())):
            records.append({'slot': int(batch['slot'][i]), 'dataset': int(batch['dataset'][i]),
                            'action': int(batch['actions'][i, 2]), 'collided': bool(batch['collided'][i]),
                            'blocked': bool(batch['blocked'][i]), 'mse': error, 'copy_current_mse': baseline,
                            **({'candidate_errors_against_observed_target': diagnostics[i].tolist()} if diagnostics is not None else {})})
    assert len(records) == len(loader.dataset)
    groups = {}
    for dataset, label in enumerate(['r2r', 'rxr']):
        for action, command in enumerate(['FORWARD', 'LEFT', 'RIGHT']):
            rows = [r for r in records if r['dataset'] == dataset and r['action'] == action]
            assert rows
            groups[f'{label}/{command}'] = {'samples': len(rows), 'mse': float(np.mean([r['mse'] for r in rows])),
                                           'copy_current_mse': float(np.mean([r['copy_current_mse'] for r in rows]))}
    report = {'samples': len(records), 'macro_mse': float(np.mean([r['mse'] for r in groups.values()])),
              'micro_mse': float(np.mean([r['mse'] for r in records])),
              'copy_current_macro_mse': float(np.mean([r['copy_current_mse'] for r in groups.values()])),
              'by_dataset_action': groups, 'blocked_samples': sum(r['blocked'] for r in records),
              'collided_samples': sum(r['collided'] for r in records), 'selection_group': 'C',
              'selection_domain': 'fixed expert-path transitions; no simulator navigation-success claim'}
    if command_diagnostics:
        margins, hits = [], []
        for r in records:
            values, a = r['candidate_errors_against_observed_target'], r['action']
            margins.append(float(np.mean([v for i, v in enumerate(values) if i != a]) - values[a]))
            hits.append(int(np.argmin(values) == a))
        report['command_sensitivity'] = {'closest_prediction_matches_executed_command': float(np.mean(hits)),
                                          'mean_other_command_error_minus_correct': float(np.mean(margins)),
                                          'interpretation': 'Conditioning diagnostic against the observed target, not measured alternate-action ground truth'}
    if was_training:
        model.train()
    return report, records


def checkpoint_state(model, optimizer, *, step, seen, plan_sha256, trial, metrics):
    return {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'seen': seen,
            'plan_sha256': plan_sha256, 'trial': trial, 'metrics': metrics,
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    torch.set_rng_state(state['torch_rng'].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(state, temporary)
    with temporary.open('rb') as file:
        os.fsync(file.fileno())
    temporary.replace(path)
