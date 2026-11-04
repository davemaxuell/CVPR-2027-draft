"""CPU checks for branch assignments, data mixing, recovery and scoped execution."""
import ast
import importlib.util
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import context
from context import *
import pipeline
from train_entry import atomic_checkpoint_writer


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_spec=importlib.util.spec_from_file_location('checked_strategy_trainer',SOURCE/'tools/train_stream_policy.py')
        cls.trainer=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(cls.trainer)

    def test_four_total_with_existing_F_reused_once(self):
        self.assertEqual(AUTH['selected'],['E','F','G','J'])
        self.assertEqual(BRANCHES,('E','G','J'))
        with self.assertRaises(ValueError):spec('F')
        self.assertNotIn('F_train',pipeline.STAGES)
        self.assertEqual(AUTH['new_dagger_rounds'],1)
        self.assertEqual([s for s in pipeline.STAGES if s.endswith('_collect')],['J_collect'])

    def test_E_G_match_data_schedule_and_sampler_but_not_initializer(self):
        e=binding('E',create=True);g=binding('G',create=True)
        self.assertEqual(e['train_roots'],g['train_roots'])
        self.assertEqual(e['recipe'],g['recipe'])
        self.assertEqual((e['windows'],g['windows'],e['updates'],g['updates']),(156691,156691,1224,1224))
        self.assertEqual(e['initial_checkpoint'],AUTH['stage_one_checkpoint'])
        self.assertEqual(g['initial_checkpoint'],AUTH['stage_two_checkpoint'])
        from stream_policy.data import MixtureSampler
        first=list(itertools.islice(MixtureSampler([109784,46907],seed=0),156672))
        second=list(itertools.islice(MixtureSampler([109784,46907],seed=0),156672))
        self.assertEqual(first,second)
        self.assertEqual(len({index for draw,index in first}),156672)
        self.assertTrue(any(index<109784 for _,index in first) and any(index>=109784 for _,index in first))

    def test_actual_commands_and_reject_wrong_initializers_or_lora(self):
        for branch in ('E','G'):
            args=self.trainer.parser('streamvln_stage2').parse_args(pipeline.training_command(branch)[4:])
            identity=binding(branch)
            training_args_check(args,branch,identity)
            args.init=AUTH['stage_two_checkpoint'] if branch=='E' else AUTH['stage_one_checkpoint']
            with self.assertRaises(RuntimeError):training_args_check(args,branch,identity)
            args.init=identity['initial_checkpoint'];args.mode='lora'
            with self.assertRaises(AssertionError):training_args_check(args,branch,identity)

    def test_J_cannot_bind_a_future_or_placeholder_F(self):
        original=context.read_json
        def missing(path,default=None):
            if Path(path).parent == Path(AUTH['existing_F_preparation']) and Path(path).name=='completion.json':return {}
            return original(path,default)
        with patch('context.read_json',side_effect=missing):
            with self.assertRaises(RuntimeError):binding('J',create=True)
        self.assertFalse((HERE/'bindings/J.json').exists())
        self.assertEqual(spec('J')['initialization'],'F_final')
        self.assertEqual(spec('J')['train_roots'],[AUTH['expert_root'],AUTH['fresh_dagger_root']])
        self.assertNotIn(AUTH['original_dagger_root'],spec('J')['train_roots'])

    def test_recovery_command_stays_in_its_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/'checkpoints/step_000156';ck.mkdir(parents=True);(ck/'complete.json').write_text('{}')
            item=dict(spec('E'),run_dir=tmp)
            with patch('pipeline.spec',return_value=item):command=pipeline.training_command('E')
            self.assertEqual(command[command.index('--resume')+1],str(ck))
            self.assertNotIn('--init',command)

    def test_atomic_checkpoint_publication_preserves_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);target=root/'step_000156'
            def save(path,*args):
                self.assertFalse(target.exists());path.mkdir()
                for name in ('model.safetensors','master_weights.safetensors','trainer_state.pt'):(path/name).write_bytes(b'fixture')
            writer=atomic_checkpoint_writer(save,'identity')
            writer(target,None,None,None,156,19968,10000,root)
            self.assertEqual(read_json(target/'complete.json')['training_identity_sha256'],'identity')
            with self.assertRaises(RuntimeError):writer(target,None,None,None,156,19968,10000,root)
            def fail(path,*args):path.mkdir();(path/'failure_evidence').write_text('preserved');raise OSError('interrupted')
            with self.assertRaises(OSError):atomic_checkpoint_writer(fail,'id')(root/'step_000312',None,None,None,312,39936,10000,root)
            self.assertEqual(len(list(root.glob('.step_000312.partial_*'))),1)
            self.assertFalse((root/'step_000312').exists())

    def test_native_science_and_fresh_collection_rule_unchanged(self):
        prior=Path(AUTH['existing_F_preparation'])/'source_v1'
        for path in SOURCE.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:self.assertEqual(sha256(path),sha256(prior/path.relative_to(SOURCE)))
        plan=read_json(HERE/'collection/plan.json')
        self.assertEqual(plan['counts'],{'r2r':10819,'rxr':19408})
        self.assertEqual((plan['beta'],plan['seed'],plan['stop_bias']),(0,1,0))
        from stream_policy.controller import DaggerRule
        self.assertEqual(plan['rule'],DaggerRule().as_dict())
        self.assertNotEqual(AUTH['fresh_dagger_root'],AUTH['original_dagger_root'])

    def test_separate_fresh_evaluation_and_no_unseen_selection(self):
        roots=[Path(spec(branch)['evaluation_root']) for branch in BRANCHES]
        self.assertEqual(len(set(roots)),3)
        for branch,root in zip(BRANCHES,roots):
            self.assertEqual(read_json(root/'settings.json')['current_run_dir'],spec(branch)['run_dir'])
            self.assertFalse(list(root.glob('phase1/*/episodes/*.json')))
            self.assertFalse(list(root.glob('phase1/*/complete.json')))
            self.assertTrue((root/'source_v1/dependencies/sim/dtw/__init__.py').is_file())
        self.assertFalse(AUTH['unseen_used_for_selection'])

    def test_unrelated_and_stopped_jobs_remain_held(self):
        holds=read_json(QUEUE/'expanded_stage2/USER_JOB_HOLDS.json')['jobs']
        for key in ['trial1_full_pipeline','expanded_hm3d','expanded_gpu_checks','expanded_full_train','expanded_verify_checkpoint','expanded_val_seen','expanded_val_unseen','expanded_report_handoff']:
            self.assertEqual(holds[key]['status'],'held')
        self.assertEqual(holds['expanded_mmc4']['status'],'released')
        self.assertEqual(read_json(QUEUE/'settings.json')['execution_mode'],'prepare_only')
        self.assertEqual(read_json(QUEUE/'PAUSED_BY_USER.json')['status'],'paused')


if __name__=='__main__':
    for path in HERE.glob('*.py'):ast.parse(path.read_text())
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    atomic_json(HERE/'evidence/implementation_checks.json',dict(status='complete' if result.wasSuccessful() else 'failed',
        time=utc(),tests=result.testsRun,errors=len(result.errors),failures=len(result.failures),source_snapshot_sha256=verify_source(),
        gpu_validation='Unchanged native navigation model/trainer; F has verified finite GPU forward/backward and optimizer updates. Each new branch remains queued, with no claim of its training completion.'))
    raise SystemExit(not result.wasSuccessful())
