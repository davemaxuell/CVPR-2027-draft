#!/usr/bin/env python3
"""Local entry point for the released ScaleVLN agents. Run through ../run.sh."""
import argparse
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
R2R = ROOT / 'VLN-DUET/datasets/R2R'
RVR = ROOT / 'VLN-DUET-RVR/datasets/REVERIE'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def first_rows(path, count):
    with path.open() as stream:
        return [json.loads(line) for _, line in zip(range(count), stream)]


def small_dataset(destination):
    """Separate tiny annotations/graphs; original datasets remain untouched."""
    scans = {'ZMojNkEp431'}  # upstream angle-feature simulator reference scan
    for source, target in [(R2R, destination / 'R2R'), (RVR, destination / 'REVERIE')]:
        for directory in ['features', 'trained_models']:
            (target / directory).parent.mkdir(parents=True, exist_ok=True)
            (target / directory).symlink_to(source / directory, target_is_directory=True)
        for path in (source / 'annotations').rglob('*'):
            if not path.is_file():
                continue
            output = target / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == '.jsonl':
                rows = first_rows(path, 2)
                output.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            elif path.name == 'R2R_scalevln_ft_aug_enc.json':
                rows = []
                for name in ['R2R_hm3d_aug_envdrop_generated_enc.jsonl',
                             'R2R_gibson_aug_envdrop_generated_enc.jsonl']:
                    for item in first_rows(source / 'annotations/pretrain_map' / name, 1):
                        rows.append(dict(item, path_id=item['instr_id'], instructions=['Synthetic instruction'],
                                         instr_encodings=[item['instr_encoding']]))
                write_json(output, rows)
            elif path.name.startswith(('R2R_', 'REVERIE_')) and path.suffix == '.json':
                rows = json.loads(path.read_text())[:2]
                write_json(output, rows)
            else:
                output.symlink_to(path)
                continue
            scans.update(row['scan'] for row in rows)
    (destination / 'pretrained').symlink_to(ROOT / 'VLN-DUET/datasets/pretrained', target_is_directory=True)
    for name in ['connectivity', 'connectivity_mp3d']:
        directory = destination / 'R2R' / name
        directory.mkdir()
        for scan in sorted(scans):
            path = R2R / 'connectivity' / (scan + '_connectivity.json')
            if not path.exists():
                raise FileNotFoundError(path)
            (directory / path.name).symlink_to(path)
        (directory / 'scans.txt').write_text('\n'.join(sorted(scans)) + '\n')
    return destination


def check():
    import torch
    import h5py
    report = {'python': sys.executable, 'torch': torch.__version__,
              'cuda_devices': torch.cuda.device_count(), 'checks': {},
              'reverie_pretraining_scope': 'HM3D only: released object LMDB has 800 HM3D scenes; no Gibson objects'}
    for module in ['MatterSim', 'transformers', 'h5py', 'networkx', 'jsonlines',
                   'tensorboardX', 'lmdb', 'msgpack_numpy', 'line_profiler']:
        importlib.import_module(module)
        report['checks'][module] = 'ok'
    from transformers import AutoTokenizer, PretrainedConfig
    AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    PretrainedConfig.from_pretrained('bert-base-uncased', local_files_only=True)
    for path in sorted((R2R / 'features').glob('*.hdf5')):
        with h5py.File(path, 'r') as handle:
            value = handle[next(iter(handle))][...]
            report['checks'][path.name] = {'entries': len(handle), 'sample_shape': list(value.shape)}
    for path in [*R2R.glob('trained_models/*/*'), *RVR.glob('trained_models/*'),
                 ROOT / 'VLN-DUET/datasets/pretrained/LXMERT/model_LXRT.pth',
                 RVR / 'features/obj_fts/data.mdb', RVR / 'features/obj_gtmax_timm_imagenet_vitb16/data.mdb']:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        report['checks'][str(path.relative_to(ROOT))] = path.stat().st_size
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required for demonstration and training')
    report['gpu'] = torch.cuda.get_device_name(0)
    report['free_gpu_gib'] = round(torch.cuda.mem_get_info()[0] / 2**30, 2)
    return report


def navigation_command(args, data, output):
    rvr = args.task == 'reverie'
    project = ROOT / ('VLN-DUET-RVR' if rvr else 'VLN-DUET')
    model = args.model
    batch = args.batch_size or (2 if args.smoke else (8 if model == 'h14' else 16))
    flags = ['--root_dir', str(data), '--dataset', args.task, '--output_dir', str(output),
             '--world_size', '1', '--seed', '0', '--tokenizer', 'bert', '--enc_full_graph',
             '--graph_sprels', '--fusion', 'dynamic', '--train_alg', 'dagger',
             '--num_l_layers', '9', '--num_x_layers', '4', '--num_pano_layers', '2',
             '--max_action_len', '15', '--max_instr_len', '100' if rvr else '200',
             '--batch_size', str(batch), '--features', 'clip-h14' if rvr else 'clip.' + model,
             '--image_feat_size', '1024' if model == 'h14' else '512', '--angle_feat_size', '4',
             '--feat_dropout', '0.4', '--dropout', '0.5', '--optim', 'adamW', '--lazy_features']
    if rvr:
        flags += ['--obj_features', 'timm_vitb16', '--obj_feat_size', '768', '--max_objects', '50',
                  '--multi_endpoints', '--dagger_sample', 'sample']
    else:
        flags += ['--expert_policy', 'spl']
    if args.command == 'eval':
        if rvr and not args.checkpoint:
            raise ValueError('REVERIE has a pretraining checkpoint only. Pass a fine-tuned --checkpoint for evaluation.')
        checkpoint = args.checkpoint or R2R / 'trained_models/finetune' / f'duet_vit-{model}_ft_best_val_unseen'
        flags += ['--resume_file', str(checkpoint), '--test', '--submit', '--detailed_output']
    else:
        steps = args.steps or (2 if args.smoke else 200000)
        interval = args.log_every or (steps if args.smoke else 500)
        aug_times = 1 if args.smoke else (1 if rvr else 9)
        if steps % (aug_times + 1) or interval % (aug_times + 1):
            raise ValueError(f'--steps and --log-every must be multiples of {aug_times + 1} for mixed fine-tuning')
        flags += ['--iters', str(steps), '--log_every', str(interval), '--lr', '2e-5' if rvr else '1e-5',
                  '--ml_weight', '0.2' if rvr else '0.15', '--gamma', '0.']
        if not rvr:
            flags += ['--aug_times', str(aug_times)]
            if model == 'h14':
                flags += ['--env_aug']
        if args.resume:
            flags += ['--resume_file', str(args.resume), '--resume_optimizer']
        else:
            checkpoint = args.checkpoint or (RVR / 'trained_models/model_step_40000.pt' if rvr else
                         R2R / 'trained_models/pretrain' / f'duet_vit-{model}_model_step_{190000 if model == "h14" else 140000}.pt')
            flags += ['--bert_ckpt_file', str(checkpoint)]
        aug = data / ('REVERIE/annotations/ade20k_pseudo3d_depth2_epoch_94_beam0_sample10.jsonl' if rvr else
                      'R2R/annotations/R2R_scalevln_ft_aug_enc.json')
        flags += ['--aug', str(aug)]
    entry = 'reverie/main_nav_obj_hm3d.py' if rvr else 'r2r/main_nav.py'
    return project / 'map_nav_src', [sys.executable, '-u', entry, *flags]


def pretrain_command(args, data, output):
    rvr = args.task == 'reverie'
    project = ROOT / ('VLN-DUET-RVR' if rvr else 'VLN-DUET')
    source = project / 'pretrain_src'
    name = 'configs/training_args.json' if rvr else f'config/r2r_pretrain_hm3d+mp3d+gibson_clip-{args.model}.json'
    config = json.loads((source / name).read_text())

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith('../datasets/'):
            return str(data / value[len('../datasets/'):])
        return value

    config = resolve(config)
    batch = args.batch_size or (2 if args.smoke else 8)
    effective = 128 if rvr else 256
    accum = args.accumulation or (2 if args.smoke else max(1, effective // (batch * args.num_gpus)))
    steps = args.steps or ((12 if rvr else 2) if args.smoke else 200000)
    config.update(train_batch_size=batch, val_batch_size=batch, gradient_accumulation_steps=accum,
                  num_train_steps=steps, valid_steps=steps if args.smoke else config['valid_steps'],
                  warmup_steps=1 if args.smoke else config['warmup_steps'],
                  log_steps=1 if args.smoke else config['log_steps'], n_workers=0,
                  lazy_features=True, fp16=args.fp16, world_size=args.num_gpus)
    model_config = source / ('configs/model_config.json' if rvr else f'config/r2r_model_config_clip-{args.model}.json')
    config_path = output / 'config.json'
    write_json(config_path, config)
    entry = 'train_hm3d_reverie.py' if rvr else 'train_r2r.py'
    command = [sys.executable, '-u']
    if args.num_gpus > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(args.num_gpus)]
    command += [entry, '--vlnbert', 'cmt', '--model_config', str(model_config),
                '--config', str(config_path), '--output_dir', str(output)]
    if args.resume:
        step = args.resume.stem.rsplit('_', 1)[-1]
        state = args.resume.with_name(f'train_state_{step}.pt')
        if not state.is_file():
            raise FileNotFoundError(f'Optimizer resume state missing: {state}. Use --checkpoint for a weights-only start.')
        command += ['--checkpoint', str(args.resume), '--resume_state', str(state)]
    elif args.checkpoint:
        command += ['--checkpoint', str(args.checkpoint)]
    print(f'Effective pretraining batch: {batch} × {accum} × {args.num_gpus} = {batch * accum * args.num_gpus}', flush=True)
    return source, command


def run_logged(command, cwd, log):
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        return 130


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'demo', 'eval', 'finetune', 'pretrain'])
    parser.add_argument('--task', choices=['r2r', 'reverie'], default='r2r')
    parser.add_argument('--model', choices=['b16', 'h14'])
    parser.add_argument('--gpu', help='CUDA_VISIBLE_DEVICES override (index, GPU UUID or MIG UUID)')
    parser.add_argument('--num-gpus', type=int, default=1, help='Pretraining processes; default 1')
    parser.add_argument('--smoke', action='store_true', help='Real updates and validation on tiny separate dataset views')
    parser.add_argument('--steps', type=int, help='Pretraining total updates; fine-tuning additional iterations')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation', type=int)
    parser.add_argument('--log-every', type=int)
    parser.add_argument('--fp16', action='store_true', help='Opt-in pretraining AMP')
    parser.add_argument('--checkpoint', type=lambda x: Path(x).resolve(), help='Initialization checkpoint')
    parser.add_argument('--resume', type=lambda x: Path(x).resolve(), help='Training checkpoint with optimizer state')
    parser.add_argument('--output', type=lambda x: Path(x).resolve())
    parser.add_argument('--episodes', default='3', help="Demo: number of evenly spaced instructions, or 'all'")
    parser.add_argument('--split', default='val_unseen', choices=['val_unseen', 'val_seen', 'val_train_seen', 'test', 'train'],
                        help='Demo: R2R split to run')
    parser.add_argument('--ids', nargs='+', help='Demo: specific instruction ids, e.g. 601_2 6620_2')
    parser.add_argument('--export-only', action='store_true', help='Demo: save the offline player without starting the visualizer')
    parser.add_argument('--no-browser', action='store_true', help='Demo: serve the visualizer without opening a browser')
    parser.add_argument('--host', default='127.0.0.1', help='Demo visualizer bind address')
    parser.add_argument('--port', type=int, default=0, help='Demo visualizer port; 0 chooses a free port')
    parser.add_argument('--dry-run', action='store_true', help='Prepare and print a launch without starting it')
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error('--port must be between 0 and 65535')
    args.model = args.model or ('h14' if args.task == 'reverie' else 'b16')
    if args.task == 'reverie' and args.model != 'h14':
        parser.error('REVERIE requires --model h14')
    for name in ['num_gpus', 'steps', 'batch_size', 'accumulation', 'log_every']:
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.episodes != 'all' and not (args.episodes.isdigit() and int(args.episodes) > 0):
        parser.error("--episodes must be a positive integer or 'all'")
    if args.export_only and (args.split != 'val_unseen' or args.ids or args.episodes == 'all'):
        parser.error('--export-only builds the offline player from N val_unseen episodes; drop --split/--ids/all')
    if args.checkpoint and args.resume:
        parser.error('Choose --checkpoint or --resume')
    if args.num_gpus > 1 and args.command != 'pretrain':
        parser.error('--num-gpus applies to pretraining; navigation uses one GPU')
    if args.command == 'demo' and args.task != 'r2r':
        parser.error('The released fine-tuned demonstration checkpoints are for R2R')
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.command == 'check':
        report = check()
        if args.output:
            write_json(args.output, report)
        print(json.dumps(report, indent=2))
        return
    import torch
    if not torch.cuda.is_available() or args.num_gpus > torch.cuda.device_count():
        raise RuntimeError(f'Requested {args.num_gpus} GPUs; CUDA exposes {torch.cuda.device_count()}. Select --gpu explicitly if needed.')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or ROOT / 'runs' / f'{args.command}_{args.task}_{args.model}_{"smoke_" if args.smoke else ""}{timestamp}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new output directory to preserve existing runs: {output}')
    output.mkdir(parents=True, exist_ok=True)
    data = ROOT / ('VLN-DUET-RVR/datasets' if args.task == 'reverie' else 'VLN-DUET/datasets')
    if args.smoke:
        data = small_dataset(output / 'data')
    if args.command == 'demo' and args.export_only:
        cwd, command = ROOT, [sys.executable, '-u', str(ROOT / 'tools/demo.py'),
                             '--model', args.model, '--episodes', str(args.episodes), '--output', str(output)]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
    elif args.command == 'demo':
        # Fresh inference that also logs every decision (scores, graph memory, attention) for the viewer.
        cwd, command = ROOT, [sys.executable, '-u', str(ROOT / 'analysis/vln_viz/log_duet_steps.py'),
                             '--model', args.model, '--split', args.split, '--out', str(output)]
        command += ['--ids', *args.ids] if args.ids else ['--episodes', args.episodes]
        if args.checkpoint:
            command += ['--checkpoint', str(args.checkpoint)]
    elif args.command == 'pretrain':
        cwd, command = pretrain_command(args, data, output)
    else:
        cwd, command = navigation_command(args, data, output)
    metadata = dict(started_utc=timestamp, cwd=str(cwd), command=command,
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    python=sys.executable, smoke=args.smoke, dry_run=args.dry_run, dataset_root=str(data),
                    reverie_pretraining_scope='HM3D only' if args.task == 'reverie' else None)
    write_json(output / 'launch.json', metadata)
    print('Working directory:', cwd, '\nCommand:', shlex.join(command), '\nOutput:', output, flush=True)
    viewer = None
    if args.command == 'demo' and not args.export_only:
        viewer = [sys.executable, '-u', str(ROOT / 'analysis/vln_viz/visualizer_app.py'),
                  '--preds', f'demo_{args.model}={output / f"detail_{args.split}.json"}',
                  '--steplog', f'demo_{args.model}={output / f"steps_{args.split}.jsonl"}',
                  '--anno', str(R2R / f'annotations/R2R_{args.split}_enc.json'),
                  '--host', args.host, '--port', str(args.port), '--autoplay']
        if not args.no_browser:
            viewer.append('--open-browser')
        metadata['visualizer_command'] = viewer
        write_json(output / 'launch.json', metadata)
        print('After inference:', shlex.join(viewer), flush=True)
    if args.dry_run:
        return
    with (output / 'console.log').open('w') as log:
        status = run_logged(command, cwd, log)
    metadata.update(exit_code=status, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'launch.json', metadata)
    if status:
        raise SystemExit(status)
    print('Completed:', output, flush=True)
    if viewer:
        # The inference subprocess has exited, releasing its GPU memory before playback.
        print('Starting navigation playback (Ctrl-C to stop the visualizer).', flush=True)
        with (output / 'console.log').open('a') as log:
            viewer_status = run_logged(viewer, ROOT, log)
        metadata['visualizer_exit_code'] = viewer_status
        write_json(output / 'launch.json', metadata)
        if viewer_status not in (0, 130):
            raise SystemExit(viewer_status)


if __name__ == '__main__':
    main()
