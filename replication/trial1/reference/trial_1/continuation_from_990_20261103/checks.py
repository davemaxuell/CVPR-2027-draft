"""Checks for initializer, held scope, data identity and checkpoint publication."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline
from train_entry import validate_args, atomic_checkpoint_writer, verify_source
from common import read_json, atomic_json, sha256, utc


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pipeline.SOURCE / 'tools/train_stream_policy.py'
        spec = importlib.util.spec_from_file_location('checked_continuation_trainer', path)
        cls.trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.trainer)

    def test_initializer_is_990_with_full_expert_data_and_stage_two_schedule(self):
        command = pipeline.training_command()
        args = self.trainer.parser('streamvln_stage2').parse_args(command[4:])
        validate_args(args, pipeline.AUTH)
        self.assertEqual(args.init, pipeline.AUTH['initial_checkpoint'])
        self.assertTrue(args.init.endswith('step_000990'))
        self.assertEqual(args.train_root, [pipeline.AUTH['expert_root']])
        self.assertEqual((args.batch_size * args.grad_accum, args.epochs), (128, 1))
        self.assertEqual((args.lr, args.vision_lr, args.weight_decay, args.warmup_frac), (2e-5, 5e-6, 0., .03))
        args.init = pipeline.AUTH['preserved_stage_one_checkpoint']
        with self.assertRaises(RuntimeError):
            validate_args(args, pipeline.AUTH)
        args.init = ''
        with self.assertRaises(RuntimeError):
            validate_args(args, pipeline.AUTH)

    def test_exact_resume_uses_new_run_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck = Path(tmp) / 'checkpoints/step_000156'
            ck.mkdir(parents=True)
            (ck / 'complete.json').write_text('{}')
            with patch.dict(pipeline.AUTH, run_dir=tmp):
                command = pipeline.training_command()
            self.assertEqual(command[command.index('--resume') + 1], str(ck))
            self.assertNotIn('--init', command)

    def test_atomic_save_rejects_overwrite_and_retains_interrupted_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'step_000001'
            def save(path, *args):
                self.assertFalse(target.exists())
                path.mkdir()
                for name in ('model.safetensors', 'master_weights.safetensors', 'trainer_state.pt'):
                    (path / name).write_bytes(b'fixture')
            writer = atomic_checkpoint_writer(save, 'identity')
            writer(target, None, None, None, 1, 128, 10000, root)
            self.assertEqual(read_json(target / 'complete.json')['training_identity_sha256'], 'identity')
            with self.assertRaises(RuntimeError):writer(target, None, None, None, 1, 128, 10000, root)
            def fail(path, *args):
                path.mkdir(); (path / 'partial').write_bytes(b'evidence')
                raise OSError('interrupted')
            with self.assertRaises(OSError):
                atomic_checkpoint_writer(fail, 'identity')(root / 'step_000002', None, None, None, 2, 256, 10000, root)
            self.assertFalse((root / 'step_000002').exists())
            self.assertEqual(len(list(root.glob('.step_000002.partial_*'))), 1)

    def test_data_native_encoding_and_epoch_budget(self):
        audit = read_json(HERE / 'evidence/expert_data_audit.json')
        identity = read_json(HERE / 'evidence/data_identity.json')
        self.assertTrue(audit['passed'])
        self.assertEqual(identity['status'], 'verified')
        self.assertEqual(audit['train']['windows']['windows'], 109784)
        self.assertEqual(audit['train']['windows']['episodes'], 30221)
        self.assertEqual(audit['one_pass_sampler']['updates'], 857)
        inv = read_json(HERE / 'preparation/inventory.json')
        self.assertEqual(inv['val_unseen_scene_overlap'], [])
        self.assertEqual(inv['counts']['rxr/train']['languages'], {'en-US':6660, 'en-IN':13336})
        previous = HERE.parent / 'full_training_20261103/source_v1'
        for p in (pipeline.SOURCE / 'stream_policy').glob('*.py'):
            self.assertEqual(sha256(p), sha256(previous / 'stream_policy' / p.name))
        self.assertEqual(sha256(pipeline.SOURCE / 'tools/train_stream_policy.py'), sha256(previous / 'tools/train_stream_policy.py'))
        for p, info in identity['assets'].items():self.assertEqual(sha256(p), info['sha256'])

    def test_user_holds_and_old_checkpoint_roles_preserved(self):
        holds = read_json(pipeline.QUEUE / 'expanded_stage2/USER_JOB_HOLDS.json')['jobs']
        for key in ('trial1_full_pipeline', 'expanded_hm3d', 'expanded_gpu_checks', 'expanded_full_train',
                    'expanded_verify_checkpoint', 'expanded_val_seen', 'expanded_val_unseen', 'expanded_report_handoff'):
            self.assertEqual(holds[key]['status'], 'held')
        self.assertEqual(holds['expanded_mmc4']['status'], 'released')
        self.assertEqual(read_json(pipeline.QUEUE / 'settings.json')['execution_mode'], 'prepare_only')
        self.assertTrue((HERE.parent / 'checkpoints/stage_one_final').resolve().name == 'step_000623')
        self.assertTrue((HERE.parent / 'checkpoints/stage_two_final').resolve().name == 'step_000990')
        self.assertNotIn('dagger_collect', pipeline.STAGES)
        self.assertEqual(pipeline.STAGES, ['continuation_train','checkpoint_verify','val_seen','val_unseen','report'])

    def test_evaluation_outputs_are_fresh_and_separate(self):
        root = Path(pipeline.AUTH['evaluation_root'])
        self.assertNotEqual(root, pipeline.WORKSPACE / 'dagger_r1_full_eval_20261031')
        self.assertEqual(read_json(root / 'settings.json')['current_run_dir'], pipeline.AUTH['run_dir'])
        self.assertFalse(list(root.glob('phase1/*/episodes/*.json')))
        self.assertFalse(list(root.glob('phase1/*/complete.json')))


if __name__ == '__main__':
    for p in HERE.glob('*.py'):ast.parse(p.read_text())
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    proof = dict(status='complete' if result.wasSuccessful() else 'failed', time=utc(), tests=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), source_snapshot_sha256=verify_source(),
        native_gpu_checks_reused=str(pipeline.BASE / 'trials/trial_1/steps/00_setup/check_stream_backbone.json'),
        note='Unchanged native model/trainer; continuation initializer, data binding, schedule, resume path and publication checked. Actual new GPU optimizer steps will be observed after launch.')
    atomic_json(HERE / 'evidence/implementation_checks.json', proof)
    raise SystemExit(not result.wasSuccessful())
