"""Shared queue bookkeeping. Standard library only; importing this never touches CUDA."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(os.environ.get('ROLEVLN_QUEUE_OUTPUT',str(Path(__file__).resolve().parent))).resolve()
QUEUE_CODE = Path(os.environ.get('ROLEVLN_QUEUE_CODE_ROOT',str(Path(__file__).resolve().parent))).resolve()
WORKSPACE = OUTPUT.parent
CODE = Path(os.environ.get('ROLEVLN_NATIVE_CODE_ROOT',str(WORKSPACE / '[OURS]' / 'baseline-v0'))).resolve()
DATA = WORKSPACE / 'data'
POLICY_PYTHON = '/home/bufsgpu/yes/envs/rolevln-qwen35/bin/python'
SIM_PYTHON = '/home/bufsgpu/yes/envs/rolevln/bin/python'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def frozen_policy(verify=True):
    identity = read_json(OUTPUT/'POLICY_FINAL/identity.json')
    if not identity or identity.get('status')!='complete':
        raise RuntimeError('POLICY_FINAL has not been frozen.')
    if verify:
        actual = {p.name:sha256(p) for p in (OUTPUT/'POLICY_FINAL/weights').glob('*.safetensors')}
        if actual!=identity['sha256']:
            raise RuntimeError('Frozen policy weights changed. Skip dependent work; never repair or overwrite them.')
    return identity


def append_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())


def decision(phase, choice, reason, alternative, key=None):
    # Single queue process owns this ledger; agent preparation writes are sequential.
    ledger = OUTPUT / 'decisions.jsonl'
    if key and ledger.exists():
        if any(json.loads(x).get('key') == key for x in ledger.read_text().splitlines()):
            return
    row = dict(time=utc(), phase=phase, decision=choice, reason=reason,
               alternative_not_taken=alternative, key=key)
    append_json(ledger, row)
    with (OUTPUT / 'DECISIONS.md').open('a') as f:
        f.write(f"\n- **{row['time']} | {phase}** — {choice}\n  Reason: {reason}\n  Alternative not taken: {alternative}\n")


def process_identity(pid):
    try:
        p = Path('/proc') / str(int(pid))
        fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': fields[19],
                'command': (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()}
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None


def same_process(identity):
    if not identity:
        return False
    current = process_identity(identity['pid'])
    return bool(current and current['start_ticks'] == identity['start_ticks'] and
                (not identity.get('command') or current['command'] == identity['command']))


def capture_process(pid, expected_fragment):
    """Allow a new Popen child to finish exec before persisting its command identity."""
    for _ in range(50):
        identity = process_identity(pid)
        if identity and expected_fragment in identity['command']:
            return identity
        if identity is None:
            return None
        time.sleep(0.02)
    raise RuntimeError('New child command identity could not be verified.')


def gpu_snapshot():
    """Read-only NVML via nvidia-smi. Never reserves or changes a device."""
    def query(args):
        r = subprocess.run(['nvidia-smi', *args, '--format=csv,noheader,nounits'],
                           text=True, capture_output=True, timeout=20)
        if r.returncode:
            raise RuntimeError(r.stderr.strip())
        return [x.strip().split(', ') for x in r.stdout.splitlines() if x.strip()]
    return {'time': utc(), 'gpus': query(['--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu']),
            'processes': query(['--query-compute-apps=gpu_uuid,pid,process_name,used_memory'])}


def source_identity():
    result = subprocess.run(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True, capture_output=True)
    paths = sorted(p for d in ['stream_policy', 'sim', 'tools', 'manifest']
                   for p in (CODE / d).glob('*') if p.is_file() and p.suffix in {'.py', '.yaml', '.json'})
    hashes = {str(p.relative_to(CODE)): sha256(p) for p in paths}
    return {'commit': result.stdout.strip() if result.returncode == 0 else None,
            'commit_unavailable_reason': None if result.returncode == 0 else 'Local code is an archive without git metadata.',
            'source_sha256': hashes,
            'source_manifest_sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}


def snapshot_queue_code(job_dir,revision=1):
    """Preserve and execute one immutable code version while the goal develops later stages."""
    destination = Path(job_dir)/f'source_v{int(revision)}'
    if destination.exists():
        manifest = read_json(destination/'SNAPSHOT.json')
        if not manifest or any(sha256(destination/name)!=value for name,value in manifest['files'].items()):
            raise RuntimeError('An existing job code snapshot changed; no job is launched.')
        return destination,manifest
    temporary = Path(job_dir)/('.source_partial_'+str(time.time_ns()))
    temporary.mkdir(parents=True)
    for path in sorted(OUTPUT.glob('*.py')):
        shutil.copy2(path,temporary/path.name)
    os.symlink(OUTPUT/'dependencies',temporary/'dependencies',target_is_directory=True)
    manifest = dict(created_utc=utc(),revision=int(revision),files={p.name:sha256(p) for p in temporary.glob('*.py')},
                    output_root=str(OUTPUT),native_source=source_identity(),
                    note='This code snapshot is immutable; output/cache paths are supplied separately in environment variables.')
    atomic_json(temporary/'SNAPSHOT.json',manifest)
    os.replace(temporary,destination)
    return destination,manifest


def refresh_sync():
    wanted = []
    excluded = {'cache', 'checkpoints', '__pycache__', 'source_snapshot', 'dependencies', 'episodes', 'workers'}
    for directory, children, files in os.walk(OUTPUT):
        children[:] = [name for name in children if name not in excluded and not name.startswith('source_v')]
        for name in files:
            p = Path(directory) / name
            if p.suffix in {'.md', '.json', '.csv', '.png'} and p.stat().st_size <= 20 * 1024**2:
                wanted.append(str(p.relative_to(OUTPUT)))
    atomic_text(OUTPUT / 'FOR_USER_SYNC.md', '# Manual sync\n\nCopy these small files by hand. '
                'No Google Drive access, upload, git push, or public posting is performed. '
                'Large local caches, full episode traces, features and checkpoints stay here. '
                'Per-episode scalar metrics are exported as CSV for manual sync.\n\n' +
                '\n'.join('- `' + x + '`' for x in sorted(wanted)) + '\n')


def status(state, persist=True):
    if persist:
        atomic_json(OUTPUT / 'state.json', state)
    rows = [f"# Autonomous queue\n\nUpdated: {utc()}\n",
            f"State: **{state.get('status', 'preparing')}**. Phase: **{state.get('phase', '0')}**.\n",
            state.get('message', '') + '\n',
            '\nOne AI agent, observed runtime gpt-6.1-sol / max. GPU 0 only. '
            'Original DAgger run is read-only. No backtracking controller.\n',
            '\n| Job | State | Evidence |\n|---|---|---|']
    settings = read_json(OUTPUT/'settings.json',{})
    if settings.get('current_run_dir'):
        progress_path = Path(settings['current_run_dir'])/'progress.json'
        progress = read_json(progress_path,{})
        if progress:
            rows.insert(4,f"MEASURED protected-run progress: step {progress.get('step')} / {progress.get('max_steps')}; "
                          f"reported at {progress.get('updated_utc')}. Evidence: `{progress_path}`.\n")
    rows.insert(4,'Implementation and execution are separate. Registered commands and current job states are listed below; '
                  'remaining work is tracked in `WORKPLAN.md`. A queued job is not evidence that its GPU checks or experiment completed.\n')
    configured = {x['id']:x for x in read_json(OUTPUT/'jobs.json',{}).get('jobs',[])}
    for key in dict.fromkeys([*configured,*state.get('jobs',{})]):
        row = state.get('jobs',{}).get(key,{})
        evidence = row.get('evidence',configured.get(key,{}).get('completion',''))
        rows.append(f"| {key} | {row.get('status','queued')} | {evidence} |")
    preparation = [
        ('Video recipe index', OUTPUT / 'preparation/video_recipe_index.json'),
        ('Co-training transfers', OUTPUT / 'preparation/cotraining_downloads_progress.json'),
        ('Video frame caching', OUTPUT / 'preparation/video_decode_progress.json'),
        ('Video corpus', OUTPUT / 'preparation/video_corpus_ready.json'),
        ('MMC4 preparation', OUTPUT / 'preparation/mmc4_prepare_progress.json'),
        ('MMC4 corpus', OUTPUT / 'preparation/mmc4_corpus_ready.json'),
        ('Past-junction training labels',OUTPUT/'preparation/slot_labels_ready.json'),
        ('Noun parser environment',OUTPUT/'preparation/nlp_ready.json'),
        ('Noun mapping first manual agent audit',OUTPUT/'preparation/noun_mapper_precision_v1.json')]
    rows.append('\nCPU preparation (may run alongside the protected training job):\n')
    for name, path in preparation:
        value = read_json(path)
        if value:
            summary = {k: value[k] for k in ['status','completed','failed','expected','counts','selected_examples','unique_videos','usable_examples','totals','instructions_reviewed','mapped_claims','semantic_precision'] if k in value}
            rows.append(f'- MEASURED {name}: `{json.dumps(summary, ensure_ascii=False)}` — `{path.relative_to(OUTPUT)}`')
    atomic_text(OUTPUT / 'STATUS.md', '\n'.join(rows) + '\n')
    refresh_sync()
