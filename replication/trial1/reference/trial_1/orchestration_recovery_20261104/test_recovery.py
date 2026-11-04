"""CPU subprocess integration checks for the actual frozen stage implementation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest

from launch import install, HERE, TARGETS

WORKSPACE = HERE.parents[4]
sys.path.insert(0, str(WORKSPACE / 'queue_after_stageA'))
from common import atomic_json, read_json, utc, process_identity, capture_process


class RecoveryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='trial1_orchestration_check_')
        self.root = Path(self.temp.name)
        self.work = self.root / 'metadata'
        self.work.mkdir()
        self.external = self.root / 'separate_evaluation_source'
        self.external.mkdir()
        self.script = self.external / 'evaluate.py'
        self.receipt = self.root / 'complete.json'
        self.script.write_text(
            'import json,os,sys,time\nfrom pathlib import Path\n'
            'time.sleep(0.2)\n'
            "Path(sys.argv[1]).write_text(json.dumps({'status':'complete',"
            "'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}))\n")
        self.command = [sys.executable, '-B', '-u', str(self.script), str(self.receipt)]
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)
        self.temp.cleanup()

    def pipeline(self, suite=False):
        path = TARGETS['suite' if suite else 'F'] / 'worker_stage.py'
        spec = importlib.util.spec_from_file_location('frozen_test_worker', path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        worker.HERE, worker.SOURCE = self.work, self.root
        for name in ('atomic_json', 'read_json', 'utc', 'process_identity', 'capture_process'):
            setattr(worker, name, globals()[name])
        worker.progress = lambda *args, **kwargs: None
        # Shorten only test polling intervals; the production stage stays unchanged.
        worker.time = types.SimpleNamespace(time_ns=time.time_ns, sleep=lambda _: time.sleep(0.02))
        pipeline = types.SimpleNamespace(worker_stage=worker, stage=worker.stage)
        pipeline.authorization_checks = 0
        if suite:
            def authorized_stage(*args, **kwargs):
                pipeline.authorization_checks += 1
                return worker.stage(*args, **kwargs)
            pipeline.stage = authorized_stage
        install(pipeline)
        return pipeline

    def ready(self):
        return read_json(self.receipt, {}).get('status') == 'complete'

    def test_original_wrong_directory_failure_is_reproduced(self):
        script = self.external / 'long_child.py'
        script.write_text('import time; time.sleep(5)\n')
        child = subprocess.Popen([sys.executable, '-B', '-u', str(script)])
        self.children.append(child)
        with self.assertRaisesRegex(RuntimeError, 'command identity could not be verified'):
            capture_process(child.pid, str(self.work))
        self.assertEqual(capture_process(child.pid, str(script))['pid'], child.pid)

    def test_f_external_script_and_completed_work_resume(self):
        pipeline = self.pipeline()
        pipeline.stage('val_seen', self.command, self.ready, gpu=True)
        state_path = self.work / 'stages/val_seen/state.json'
        state = read_json(state_path)
        receipt = read_json(self.receipt)
        self.assertEqual(state['status'], 'complete')
        self.assertEqual(state['attempts'], 1)
        self.assertEqual(receipt['gpu'], '0')
        self.assertIn(str(self.script), state['attempt_history'][0]['process']['command'])
        pipeline.stage('val_seen', self.command, self.ready, gpu=True)
        self.assertEqual(read_json(self.receipt), receipt)
        self.assertEqual(read_json(state_path)['attempts'], 1)

    def test_suite_keeps_authorization_and_cpu_nice_wrapper(self):
        pipeline = self.pipeline(suite=True)
        pipeline.stage('verify', self.command, self.ready, gpu=False)
        self.assertEqual(pipeline.authorization_checks, 1)
        self.assertEqual(read_json(self.receipt)['gpu'], '')
        self.assertEqual(read_json(self.work / 'stages/verify/state.json')['status'], 'complete')

    def test_recorded_child_is_adopted_without_duplicate_launch(self):
        pipeline = self.pipeline()
        child = subprocess.Popen(self.command, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
        self.children.append(child)
        identity = capture_process(child.pid, str(self.script))
        state_path = self.work / 'stages/val_unseen/state.json'
        atomic_json(state_path, dict(status='running', attempts=1, process=identity,
                                    log=str(self.root / 'adopted.log')))
        pipeline.stage('val_unseen', self.command, self.ready, gpu=True)
        self.assertEqual(read_json(self.receipt)['pid'], child.pid)
        self.assertEqual(read_json(state_path)['attempts'], 1)

    def test_failure_retains_evidence_and_attempt_limit(self):
        pipeline = self.pipeline()
        self.script.write_text('import time; time.sleep(0.2); raise RuntimeError("intentional-test-failure")\n')
        state_path = self.work / 'stages/failing/state.json'
        atomic_json(state_path, {'attempts': 2, 'status': 'failed'})
        with self.assertRaisesRegex(RuntimeError, 'Stage failed'):
            pipeline.stage('failing', self.command, self.ready, gpu=True)
        state = read_json(state_path)
        self.assertEqual((state['status'], state['attempts']), ('failed', 3))
        self.assertIn('intentional-test-failure', Path(state['log']).read_text())
        with self.assertRaisesRegex(RuntimeError, 'Bounded recovery exhausted'):
            pipeline.stage('failing', self.command, self.ready, gpu=True)
        self.assertFalse(self.receipt.exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
