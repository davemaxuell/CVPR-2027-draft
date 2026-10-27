"""Durable matched DINO transfer/fresh training, selection, freezing and verification."""
import fcntl
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wm.common import REFERENCE, ROOT, RUN, now, read, sha, write
from wm.features import CACHE


def main():
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    runtime, training = read(RUN / 'check_runtime.json'), read(RUN / 'check_training.json')
    assert runtime['status'] == training['status'] == 'passed'
    for report in [runtime, training]:
        for relative, expected in report['source_sha256'].items():
            assert sha(ROOT / relative) == expected
    data = read(RUN / 'data_index/manifest.json')
    assert training['data_index_manifest_sha256'] == sha(RUN / 'data_index/manifest.json')
    reference = read(REFERENCE / 'reference.json')
    identity = reference['policy_identity_sha256']
    assert identity == data['selected_policy_identity_sha256'] == runtime['policy_identity_sha256']
    assert reference['decoder']['stop_logit_bias'] == -.25
    order_path = RUN / 'sample_order.npy'
    if not order_path.exists():
        np.save(order_path, np.random.default_rng(5).permutation(data['train_samples']).astype(np.int32))
    order = np.load(order_path, mmap_mode='r')
    assert len(order) == data['train_samples'] and np.array_equal(np.sort(order), np.arange(len(order)))
    plan = {'scope': 'Step 5A: DINO-WM transferred PointMaze predictor and matched freshly initialized predictor',
            'scope_order': 'DINO pair first, as in the earlier sequence; NWM/CDiT retained for its later paired repeat',
            'selected_policy_identity_sha256': identity, 'policy_reference': str(REFERENCE),
            'policy_checkpoint': reference['source_checkpoint'], 'policy_stop_logit_bias': -.25,
            'policy_selection_caveat': 'Explicit user selection despite failed stopping safeguard; Step 2 preserved as control',
            'transfer_selection_sha256': sha(RUN / 'assets/transfer_selection.json'),
            'data_index_manifest_sha256': sha(RUN / 'data_index/manifest.json'),
            'sample_order_sha256': sha(order_path), 'seed': 5, 'training_group': 'A', 'validation_group': 'C',
            'samples_per_trial': data['train_samples'], 'epochs': 1, 'updates': math.ceil(data['train_samples'] / 64),
            'batch_size': 64, 'data_workers': 8, 'optimizer': 'AdamW', 'peak_lr': 5e-4, 'weight_decay': .01,
            'betas': [.9, .999], 'epsilon': 1e-8, 'clip_grad_norm': 1.0, 'warmup_updates': 512,
            'lr_schedule': 'linear warmup then cosine to 10% peak; identical in both trials',
            'precision': 'float32 predictor weights with BF16 autocast; native float32 frozen encoder; float16 feature storage',
            'deterministic_algorithms': True, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            'history': 'last three actual frames from matched roots having at least four observations',
            'early_DINO_extra_used': False, 'loss': 'MSE on final predicted 196x384 raw visual patches only',
            'eval_every': 4096, 'validation_samples': 3072,
            'selection': 'minimum dataset/action-macro group-C feature MSE, then earliest update; include initialization',
            'validation_limit': data['validation_limit'],
            'budget_rationale': 'One complete pass over every matched sample in each trial, fixed before validation results; measured batch64 throughput 761.98 samples/s, about 53 minutes model compute per trial',
            'physical_gpu': 0, 'training_profile': 'check_runtime.json'}
    if (RUN / 'plan.json').exists():
        assert read(RUN / 'plan.json') == plan
    else:
        write(RUN / 'plan.json', plan)
    files = ['wm/common.py', 'wm/dino.py', 'wm/features.py', 'wm/training.py', 'wm/frozen.py',
             'tools/prepare_step5_data.py', 'tools/prepare_dino_transfer.py', 'tools/cache_dino_features.py',
             'tools/check_step5_runtime.py', 'tools/check_step5_training.py', 'tools/train_dino_world_model.py',
             'tools/finalize_step5.py', 'tools/run_step5.py', 'Scorer/world_models.py', 'Scorer/interfaces.py',
             'third_party/dino_wm/models/vit.py', 'policy/experimental_reference.py', 'manifest/scene_partition.json']
    hashes = {relative: sha(ROOT / relative) for relative in files}
    if (RUN / 'source_sha256.json').exists():
        assert read(RUN / 'source_sha256.json') == hashes, 'Review source changes before resuming'
    else:
        write(RUN / 'source_sha256.json', hashes)
    for relative in files:
        target = RUN / 'source' / relative
        if target.exists():
            assert sha(target) == hashes[relative]
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
    if (RUN / 'status.json').exists() and read(RUN / 'status.json')['status'] == 'complete':
        print((RUN / 'status.json').read_text()); return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               HF_HOME='/data/shared/hf_cache', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    children, logs = [], []
    status = {'status': 'running', 'pid': os.getpid(), 'started_or_resumed_utc': now(), 'scope': plan['scope']}
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Step 5 interrupted')
    signal.signal(signal.SIGTERM, interrupted)

    def stage(name, **values):
        status.update(stage=name, updated_utc=now(), **values)
        write(RUN / 'status.json', status)

    def launch(name, command):
        logfile = (RUN / f'{name}.log').open('a'); logs.append(logfile)
        child = subprocess.Popen(command, env=env, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        stage(name, child_pid=child.pid)
        while child.poll() is None:
            time.sleep(5)
        assert child.returncode == 0, f'{name} failed with {child.returncode}; inspect {name}.log'

    try:
        if not (CACHE / 'manifest.json').exists():
            job_path = RUN / 'cache_job.json'
            job = read(job_path) if job_path.exists() else None
            if job and Path(f"/proc/{job['pid']}").exists():
                stage('cache_features', attached_cache_pid=job['pid'])
                while not (CACHE / 'manifest.json').exists():
                    assert Path(f"/proc/{job['pid']}").exists(), 'Attached feature extraction stopped; inspect cache_features.log'
                    time.sleep(5)
            else:
                launch('cache_features', [sys.executable, '-u', 'tools/cache_dino_features.py'])
        cache = read(CACHE / 'manifest.json')
        assert cache['status'] == 'complete' and cache['selected_policy_identity_sha256'] == identity
        stat = (CACHE / 'features.npy').stat()
        seal = {'cache_manifest_sha256': sha(CACHE / 'manifest.json'), 'features_size_mtime_ns': [stat.st_size, stat.st_mtime_ns],
                'features_sha256': cache['features_sha256'], 'encoder_identity_sha256': cache['encoder_identity_sha256']}
        if (RUN / 'cache_seal.json').exists():
            assert read(RUN / 'cache_seal.json') == seal
        else:
            write(RUN / 'cache_seal.json', seal)
        for trial in ['transfer', 'fresh']:
            if not (RUN / trial / 'result.json').exists():
                launch(f'train_{trial}', [sys.executable, '-u', 'tools/train_dino_world_model.py', '--trial', trial])
            assert read(RUN / trial / 'result.json')['status'] == 'complete'
        launch('finalize', [sys.executable, '-u', 'tools/finalize_step5.py'])
        assert read(RUN / 'final_verification.json')['status'] == 'passed'
        status.update(status='complete', stage='complete', completed_utc=now(), result=str(RUN / 'result.json'))
        status.pop('child_pid', None); status.pop('attached_cache_pid', None)
        write(RUN / 'status.json', status)
        print(__import__('json').dumps(status), flush=True)
    except BaseException as error:
        status.update(status='failed', error=repr(error), updated_utc=now())
        write(RUN / 'status.json', status)
        raise
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        for logfile in logs:
            logfile.close()


if __name__ == '__main__':
    main()
