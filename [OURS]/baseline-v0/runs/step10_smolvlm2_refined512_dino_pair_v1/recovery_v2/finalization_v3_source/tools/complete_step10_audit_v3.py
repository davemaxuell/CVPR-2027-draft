"""Finish the existing 1,120 rollouts with a versioned audit-only correction."""
import fcntl
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scorer_features.common import now, read, sha, write
from tools.run_step10_metric_recovery import ORIGINAL_RUN, RUN, verify


def main():
    lock = (RUN / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    protocol = verify()
    record_path = RUN / 'finalization_v3.json'
    if not record_path.exists():
        sources = {name: sha(ROOT / name) for name in (
            'tools/complete_step10_audit_v3.py', 'tools/finalize_step10_audit_v3.py')}
        inputs = {}
        for system in protocol['systems']:
            directory = RUN / 'evaluation' / system['name']
            assert read(directory / 'summary.json')['status'] == 'complete'
            for name in ('summary.json', 'job.json', 'gpu_memory.json'):
                path = directory / name
                inputs[str(path)] = sha(path)
        for name in sources:
            target = RUN / 'finalization_v3_source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write(record_path, {
            'format': 'step10_audit_only_amendment_v3', 'declared_utc': now(),
            'reason': 'The imported-policy audit reused expected for a normalized row, overwriting the expected episode-ID set before the next system. Use separate variables and rerun every audit.',
            'evaluation_protocol_sha256': sha(RUN / 'protocol.json'),
            'previous_finalizer_sha256': sha(ROOT / 'tools/finalize_step10_metric_recovery.py'),
            'source_sha256': sources, 'input_sha256': inputs,
            'evaluation_records_unchanged': True, 'selection_unchanged': True,
            'no_E_tuning': True, 'all_original_checks_retained': True,
        })
    record = read(record_path)
    assert record['evaluation_protocol_sha256'] == sha(RUN / 'protocol.json')
    for name, digest in record['source_sha256'].items():
        assert sha(ROOT / name) == sha(RUN / 'finalization_v3_source' / name) == digest, name
    for name, digest in record['input_sha256'].items():
        assert sha(name) == digest, name
    try:
        if not (RUN / 'result.json').exists():
            status = {'status': 'running', 'stage': 'final_audit_v3', 'pid': os.getpid(),
                      'updated_utc': now(), 'reporting_recovery': str(RUN)}
            for directory in (RUN, ORIGINAL_RUN):
                path = directory / 'status.json'
                if path.exists():
                    archive = directory / 'history' / 'final_audit_failure_v2.json'
                    if not archive.exists():
                        archive.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, archive)
                write(path, status)
            subprocess.run([sys.executable, '-u', 'tools/finalize_step10_audit_v3.py'], cwd=ROOT, check=True)
        result = read(RUN / 'result.json')
        assert result['status'] == 'complete'
        assert result['finalization_amendment_sha256'] == sha(record_path)
        assert result['verification_sha256'] == sha(RUN / 'verification.json')
        assert read(RUN / 'verification.json')['status'] == 'passed'
        verify()
        # Publish reports first and the completion marker last. Resuming after a
        # publication interruption still finishes publishing the same result.
        for name in ('verification.json', 'metrics.csv', 'RESULTS.md', 'finalization_v3.json'):
            target = ORIGINAL_RUN / name
            if target.exists():
                assert sha(target) == sha(RUN / name), name
            else:
                shutil.copy2(RUN / name, target)
        result = {**result, 'recovery_directory': str(RUN),
                  'original_protocol_sha256': sha(ORIGINAL_RUN / 'protocol.json')}
        if (ORIGINAL_RUN / 'result.json').exists():
            assert read(ORIGINAL_RUN / 'result.json') == result
        else:
            write(ORIGINAL_RUN / 'result.json', result)
        write(ORIGINAL_RUN / 'status.json', {'status': 'complete', 'completed_utc': now(),
              'reporting_recovery': str(RUN), 'finalization': str(record_path)})
        write(ROOT / 'runs/automatic_smolvlm2/status.json', {
            'status': 'initial_smol_dino_trial_complete', 'completed_utc': now(),
            'reporting_recovery': str(RUN), 'finalization': str(record_path)})
        print('Verified initial DINO Step 10 results published.', flush=True)
    except BaseException as error:
        status = {'status': 'failed', 'stage': 'final_audit_v3', 'error': repr(error),
                  'reporting_recovery': str(RUN), 'updated_utc': now()}
        write(RUN / 'status.json', status)
        write(ORIGINAL_RUN / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
