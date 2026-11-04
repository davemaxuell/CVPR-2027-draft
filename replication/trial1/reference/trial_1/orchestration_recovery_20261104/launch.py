"""Repair child identity capture without changing any frozen experiment source."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TARGETS = {
    'F': HERE.parent / 'continuation_from_990_20261103',
    'suite': HERE.parent / 'strategy_comparison_20261103',
}


def install(pipeline):
    """Match each owned child against its actual Python script, including evals."""
    worker = pipeline.worker_stage
    original_stage = worker.stage
    original_capture = worker.capture_process

    def checked_stage(name, command, ready, progress_path=None, gpu=False, extra_env=None):
        if len(command) < 4 or list(command[1:3]) != ['-B', '-u']:
            raise RuntimeError('Recovery wrapper requires an explicit Python script command.')
        script = Path(command[3])
        if not script.is_absolute() or not script.is_file():
            raise RuntimeError('Child script must be an existing absolute path.')

        def capture_actual_script(pid, legacy_directory):
            return original_capture(pid, str(script))

        # These pipelines hold an exclusive lock and launch stages sequentially.
        # Preserve the frozen stage's adoption, attempt budget and completion gates.
        worker.capture_process = capture_actual_script
        try:
            return original_stage(name, command, ready, progress_path, gpu, extra_env)
        finally:
            worker.capture_process = original_capture

    worker.stage = checked_stage
    # F cached the function at import; the suite resolves worker.stage dynamically
    # through its own authorization wrapper, which must remain in place.
    if pipeline.stage is original_stage:
        pipeline.stage = checked_stage


def verify_recovery():
    import hashlib

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest_path = HERE / 'SOURCE_SNAPSHOT.json'
    manifest = json.loads(manifest_path.read_text())
    checks = json.loads((HERE / 'checks.json').read_text())
    if checks.get('status') != 'complete' or checks.get('source_snapshot_sha256') != digest(manifest_path):
        raise RuntimeError('Recovery verification is missing or stale.')
    for name, expected in manifest['files'].items():
        if digest(HERE / name) != expected:
            raise RuntimeError('Recovery source checksum changed: ' + name)
    for name, expected in manifest['frozen_experiment_manifests'].items():
        if digest(Path(name)) != expected:
            raise RuntimeError('Frozen experiment source manifest changed: ' + name)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TARGETS:
        raise SystemExit('Usage: launch.py F|suite')
    target = sys.argv[1]
    verify_recovery()
    sys.path.insert(0, str(TARGETS[target]))
    pipeline = importlib.import_module('pipeline')
    install(pipeline)
    print(json.dumps({'event': 'verified_orchestration_recovery', 'target': target,
                      'recovery_source': str(HERE / 'SOURCE_SNAPSHOT.json'),
                      'original_pipeline': pipeline.__file__}), flush=True)
    pipeline.main()


if __name__ == '__main__':
    main()
