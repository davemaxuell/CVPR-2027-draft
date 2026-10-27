"""One matched full-data pass of native NWM diffusion fitting and EMA selection."""
import argparse
import copy
import fcntl
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from nwm_training.common import CACHE, RUN, now, read, sha, write
from nwm_training.data import PosteriorSamples
from nwm_training.engine import configure, diffusion, evaluate, update
from nwm_training.model import build_model
from nwm_training.protocol import verify_cache, verify_protocol
from wm.training import restore_rng, save_checkpoint


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--trial', choices=['transfer','fresh'], required=True)
    args = parser.parse_args(); plan = verify_protocol(); cache = verify_cache()
    configure(plan['seed']); identity = sha(RUN / 'protocol.json')
    out = RUN / args.trial; out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'train.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['status'] == 'complete'; return
    model = build_model(args.trial, 'cuda:0', seed=plan['seed']).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    spec = plan['optimizer']; optimizer = torch.optim.AdamW(model.parameters(), lr=spec['lr'],
        weight_decay=spec['weight_decay'], betas=tuple(spec['betas']), eps=spec['epsilon'])
    process = diffusion(); step = seen = 0; best = None
    if (out / 'latest.json').exists():
        latest = read(out / 'latest.json'); path = out / latest['checkpoint']
        assert sha(path) == latest['checkpoint_sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['protocol_sha256'] == identity and state['trial'] == args.trial
        model.load_state_dict(state['model'], strict=True); ema.load_state_dict(state['ema'], strict=True)
        optimizer.load_state_dict(state['optimizer']); restore_rng(state)
        step, seen, best = state['step'], state['seen'], latest['best']; del state
    else:
        configure(plan['seed'] + 1000)
    order = np.load(plan['sample_order'], mmap_mode='r')
    train_loader = DataLoader(PosteriorSamples('train', order=order, start=seen), batch_size=plan['batch_size'],
        num_workers=plan['data_workers'], pin_memory=True, persistent_workers=True,
        generator=torch.Generator().manual_seed(plan['seed'] + 2000))
    validation = DataLoader(PosteriorSamples('validation'), batch_size=plan['batch_size'], num_workers=2,
        pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(plan['seed'] + 3000))
    started = time.monotonic(); initial_seen = seen; totals = dict(loss=0., mse=0., vb=0.); count = 0

    def savepoint():
        nonlocal best
        report, records = evaluate(ema, process, validation)
        relative = f'checkpoints/step_{step:06d}.pt'; path = out / relative
        state = {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'step': step, 'seen': seen, 'trial': args.trial, 'protocol_sha256': identity,
            'metrics': report, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
        save_checkpoint(path, state)
        candidate = {'step': step, 'seen': seen, 'checkpoint': relative, 'checkpoint_sha256': sha(path),
                     'macro_loss': report['macro_loss'], 'metrics': report}
        write(out / f'validation/step_{step:06d}.json', {'candidate': candidate, 'rows': records})
        write(path.with_suffix('.json'), candidate)
        if best is None or (candidate['macro_loss'], step) < (best['macro_loss'], best['step']): best = candidate
        write(out / 'latest.json', {**candidate, 'best': best})
        write(out / 'best.json', best)
        print(json.dumps({'event': 'validation', 'trial': args.trial, 'step': step, 'macro_loss': report['macro_loss'],
                          'macro_mse': report['macro_mse'], 'macro_vb': report['macro_vb'], 'best_step': best['step']}), flush=True)

    if not (out / 'latest.json').exists(): savepoint()
    torch.cuda.reset_peak_memory_stats()
    with (out / 'train.jsonl').open('a') as log:
        for batch in train_loader:
            values, norm = update(model, ema, optimizer, process, batch)
            size = len(batch['action']); step += 1; seen += size; count += size
            for key in totals: totals[key] += values[key] * size
            if step % 100 == 0 or seen == len(order):
                elapsed = time.monotonic() - started
                record = {'time_utc': now(), 'trial': args.trial, 'step': step, 'updates': plan['updates'],
                    'samples_seen': seen, 'samples_total': len(order), 'gradient_norm': norm,
                    'train': {key: value / count for key, value in totals.items()}, 'lr': spec['lr'],
                    'seconds_this_run': elapsed, 'samples_per_second_including_validation': (seen - initial_seen) / elapsed,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                log.write(json.dumps(record) + '\n'); log.flush(); print(json.dumps(record), flush=True)
                write(out / 'progress.json', {'status': 'training', **record})
                totals = dict(loss=0., mse=0., vb=0.); count = 0
            if step % plan['eval_every'] == 0 or seen == len(order): savepoint()
    assert seen == plan['samples_per_trial'] == len(order) and step == plan['updates']
    path = out / best['checkpoint']; assert sha(path) == best['checkpoint_sha256']
    state = torch.load(path, map_location='cpu', weights_only=True)
    ema.load_state_dict(state['ema'], strict=True); del state
    report, rows = evaluate(ema, process, validation)
    assert report == best['metrics'], 'Frozen selected EMA must reproduce its validation result exactly'
    write(out / 'selected_validation.json', {'metrics': report, 'rows': rows})
    save_checkpoint(out / 'selected_model.pt', {k: v.detach().cpu() for k, v in ema.state_dict().items()})
    result = {'status': 'complete', 'completed_utc': now(), 'trial': args.trial, 'protocol_sha256': identity,
        'updates': step, 'samples_seen': seen, 'epochs': 1, 'selected_step': best['step'], 'selected_state': 'EMA',
        'selected_checkpoint': str(path), 'selected_checkpoint_sha256': best['checkpoint_sha256'],
        'selected_model': str(out / 'selected_model.pt'), 'selected_model_sha256': sha(out / 'selected_model.pt'),
        'selected_validation_sha256': sha(out / 'selected_validation.json'), 'metrics': report,
        'cache_manifest_sha256': sha(CACHE / 'manifest.json'), 'trainable_after_freeze': 0,
        'seconds_this_run': time.monotonic() - started, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    write(out / 'result.json', result); write(out / 'progress.json', {'status': 'complete', **result})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
