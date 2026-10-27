"""One identical full-data pass for the transferred and freshly initialized predictors."""
import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import ROOT, RUN, now, read, sha, write
from wm.dino import build_dynamics
from wm.features import CACHE, FeatureSamples
from wm.training import checkpoint_state, evaluate, learning_rate, restore_rng, save_checkpoint, seed_training, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', choices=['transfer', 'fresh'], required=True)
    args = parser.parse_args()
    plan = read(RUN / 'plan.json')
    plan_hash = sha(RUN / 'plan.json')
    data = read(RUN / 'data_index/manifest.json')
    cache = read(CACHE / 'manifest.json')
    assert cache['status'] == 'complete'
    assert cache['selected_policy_identity_sha256'] == plan['selected_policy_identity_sha256'] == data['selected_policy_identity_sha256']
    for name, expected in read(RUN / 'source_sha256.json').items():
        assert sha(ROOT / name) == expected, ('Pinned training source changed', name)
    for name, expected in data['files'].items():
        assert sha(RUN / 'data_index' / name) == expected
    assert sha(RUN / 'sample_order.npy') == plan['sample_order_sha256']
    assert sha(CACHE / 'config.json') == cache['config_sha256']
    seal = read(RUN / 'cache_seal.json')
    stat = (CACHE / 'features.npy').stat()
    assert [stat.st_size, stat.st_mtime_ns] == seal['features_size_mtime_ns']
    assert sha(CACHE / 'manifest.json') == seal['cache_manifest_sha256']
    out = RUN / args.trial
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'train.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'result.json').exists():
        assert read(out / 'result.json')['status'] == 'complete'
        print((out / 'result.json').read_text()); return
    seed_training(plan['seed'])
    model = build_dynamics(args.trial, seed=plan['seed']).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['peak_lr'], weight_decay=plan['weight_decay'], betas=(.9, .999), eps=1e-8)
    step, seen, best = 0, 0, None
    if (out / 'latest.json').exists():
        latest = read(out / 'latest.json')
        path = out / latest['checkpoint']
        assert sha(path) == latest['sha256']
        # Keep Adam's non-capturable step counters on CPU; load_state_dict moves
        # parameter moments to the parameter device without changing counter semantics.
        state = torch.load(path, map_location='cpu', weights_only=True)
        assert state['plan_sha256'] == plan_hash and state['trial'] == args.trial
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        step, seen = state['step'], state['seen']
        restore_rng(state)
        best = read(out / 'best.json')
        del state
    else:
        # Predictor initialization consumes random numbers; reset training dropout RNG identically.
        seed_training(plan['seed'] + 1000)
    order = np.load(RUN / 'sample_order.npy', mmap_mode='r')
    train_set = FeatureSamples('train', order=order, start=seen)
    val_set = FeatureSamples('validation')
    train_loader = DataLoader(train_set, batch_size=plan['batch_size'], num_workers=plan['data_workers'],
                              pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(plan['seed'] + 2000))
    val_loader = DataLoader(val_set, batch_size=plan['batch_size'], num_workers=2,
                            pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(plan['seed'] + 3000))
    log = (out / 'train.jsonl').open('a')
    started, resumed_seen = time.time(), seen
    loss_sum, loss_samples = 0.0, 0

    def checkpoint(report):
        nonlocal best
        relative = f'checkpoints/step_{step:06d}.pt'
        path = out / relative
        state = checkpoint_state(model, optimizer, step=step, seen=seen, plan_sha256=plan_hash, trial=args.trial, metrics=report)
        save_checkpoint(path, state)
        digest = sha(path)
        selection = {'step': step, 'seen': seen, 'checkpoint': relative, 'sha256': digest,
                     'macro_mse': report['macro_mse'], 'plan_sha256': plan_hash, 'metrics': report}
        write(out / f'evaluations/step_{step:06d}.json', selection)
        # Publish best before latest so a resumed run always has a valid selection.
        if best is None or (selection['macro_mse'], selection['step']) < (best['macro_mse'], best['step']):
            best = selection
            write(out / 'best.json', best)
        write(out / 'latest.json', selection)
        print(json.dumps({'event': 'validation', 'trial': args.trial, 'step': step, 'seen': seen, 'mse': report['macro_mse'], 'copy_current_mse': report['copy_current_macro_mse'], 'best_step': best['step']}), flush=True)

    if seen == 0 and step == 0 and not (out / 'latest.json').exists():
        report, _ = evaluate(model, val_loader)
        checkpoint(report)
    torch.cuda.reset_peak_memory_stats()
    for batch in train_loader:
        step += 1
        lr = learning_rate(step, plan['updates'], plan['peak_lr'], plan['warmup_updates'])
        loss, norm = update(model, optimizer, batch, lr)
        batch_count = len(batch['features'])
        seen += batch_count
        loss_sum += loss * batch_count
        loss_samples += batch_count
        if step % 100 == 0 or seen == len(order):
            elapsed = time.time() - started
            record = {'time_utc': now(), 'trial': args.trial, 'step': step, 'updates': plan['updates'],
                      'samples_seen': seen, 'samples_total': len(order), 'train_mse': loss_sum / loss_samples,
                      'lr': lr, 'grad_norm': norm, 'elapsed_seconds_this_run': elapsed,
                      'samples_per_second_including_validation': (seen - resumed_seen) / elapsed,
                      'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
            log.write(json.dumps(record) + '\n'); log.flush()
            write(out / 'progress.json', {'status': 'training', **record})
            print(json.dumps(record), flush=True)
            loss_sum, loss_samples = 0.0, 0
        if step % plan['eval_every'] == 0 or seen == len(order):
            report, _ = evaluate(model, val_loader)
            checkpoint(report)
    assert seen == len(order) == plan['samples_per_trial'] and step == plan['updates']
    stat = (CACHE / 'features.npy').stat()
    assert [stat.st_size, stat.st_mtime_ns] == seal['features_size_mtime_ns']
    best_path = out / best['checkpoint']
    assert sha(best_path) == best['sha256']
    state = torch.load(best_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    model.requires_grad_(False).eval()
    final, rows = evaluate(model, val_loader, command_diagnostics=True)
    assert abs(final['macro_mse'] - best['macro_mse']) < 1e-7
    write(out / 'selected_validation.json', {'metrics': final, 'records': rows})
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / 'selected_model.pt')
    result = {'status': 'complete', 'completed_utc': now(), 'trial': args.trial,
              'updates': step, 'samples_seen': seen, 'epochs': 1,
              'selected_step': best['step'], 'metrics': final, 'selected_checkpoint': str(best_path),
              'selected_checkpoint_sha256': best['sha256'], 'model_sha256': sha(out / 'selected_model.pt'),
              'trainable_parameters_after_freeze': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'total_parameters': sum(p.numel() for p in model.parameters()),
              'selected_policy_identity_sha256': plan['selected_policy_identity_sha256'],
              'encoder_identity_sha256': cache['encoder_identity_sha256'], 'plan_sha256': plan_hash,
              'seconds_this_run': time.time() - started}
    write(out / 'result.json', result)
    write(out / 'progress.json', {'status': 'complete', 'step': step, 'samples_seen': seen, 'selected_step': best['step']})
    log.close()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
