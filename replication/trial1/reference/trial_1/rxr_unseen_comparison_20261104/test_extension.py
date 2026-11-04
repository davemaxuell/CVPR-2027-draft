"""Regression checks for added evaluation order, recovery and result attribution."""
import contextlib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run
import rxr_context as context


class ExtensionChecks(unittest.TestCase):
    def execute_fake_suite(self, completed=False):
        events, commands = [], []
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); here=root/'extension'; suite=root/'suite'
            here.mkdir();suite.mkdir()
            def read(path, default=None):
                path=Path(path)
                if path.name=='checks.json':return {'status':'complete','source_snapshot_sha256':'new'}
                if path.name=='implementation_checks.json':return {'status':'complete','source_snapshot_sha256':'old'}
                if path.name=='existing_mixture.json':return {'passed':True}
                if path.name=='split_audit.json':return {'assets':{}}
                if path.name=='status.json':return {'status':'complete' if completed else 'pending'}
                return default
            def stage(name,command,ready,*args,**kwargs):
                events.append(name);commands.append((name,command,kwargs))
            def training(branch):
                if completed:raise AssertionError('Completed training was regenerated.')
                return [run.POL,'-B','-u',str(root/(branch+'.py')),'--init',{'E':'623','G':'990','J':'F'}[branch]]
            with contextlib.ExitStack() as stack:
                for name,value in dict(HERE=here,SUITE=suite,read_json=read,authorize=lambda:None,
                    verify_source=lambda:'new',stage=stage,evaluate_rxr=lambda model:events.append(model+'_rxr_val_unseen')).items():
                    stack.enter_context(patch.object(run,name,value))
                for name,value in dict(check_authorization=lambda:None,verify_source=lambda:'old',
                    f_checkpoint=lambda:{'status':'complete'},spec=lambda branch:{'run_dir':str(root/branch)},
                    collect_j=lambda:events.extend(['J_collect','J_audit']),training_command=training,
                    evaluate=lambda branch,split:events.append(branch+'_'+split)).items():
                    stack.enter_context(patch.object(run.original,name,value))
                stack.enter_context(patch.dict(run.os.environ,CUDA_VISIBLE_DEVICES='0'))
                run.main()
        return events,commands

    def test_full_order_and_original_initializers(self):
        events,commands=self.execute_fake_suite()
        self.assertEqual(events,run.STAGES)
        self.assertEqual([name for name in events if name.endswith('_rxr_val_unseen')],
            ['F_rxr_val_unseen','990_rxr_val_unseen','623_rxr_val_unseen','E_rxr_val_unseen','G_rxr_val_unseen','J_rxr_val_unseen'])
        train={name:cmd[-1] for name,cmd,_ in commands if name.endswith('_train')}
        self.assertEqual(train,{'E_train':'623','G_train':'990','J_train':'F'})
        self.assertEqual(events.count('J_collect'),1)

    def test_resume_does_not_request_training_again(self):
        events,commands=self.execute_fake_suite(completed=True)
        self.assertEqual(events,run.STAGES)
        for name,command,kwargs in commands:
            if name.endswith('_train'):
                self.assertEqual(command[1:3],['-B','-u'])
                self.assertTrue(command[3].endswith('/train_entry.py'))

    def test_no_comparison_completion_when_an_rxr_result_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def verify(model):
                if model=='J':raise RuntimeError('Full RxR result is not complete.')
                return {'status':'complete'}
            with patch.object(run,'HERE',root),patch.object(run,'authorize',lambda:None),patch.object(run,'verify_result',verify):
                with self.assertRaisesRegex(RuntimeError,'not complete'):run.comparison()
            self.assertFalse((root/'completion.json').exists())

    def result_fixture(self, root):
        here=root/'extension';here.mkdir();out=root/'model';folder=out/'phase1/val_unseen'
        (folder/'episodes').mkdir(parents=True);(folder/'workers').mkdir()
        native=root/'code';native.mkdir();(native/'evaluate.py').write_text('test fixture\n')
        text='Walk to the chair.'
        audit={'selected_episode_ids':['4'],'episodes':{'4':{'scene_id':'mp3d/scene/scene.glb',
            'language':'en-IN','role':'guide','instruction_id':'72545','instruction_sha256':hashlib.sha256(text.encode()).hexdigest()}}}
        context.atomic_json(here/'split_audit.json',audit);context.atomic_json(here/'protocol.json',{'test':True})
        proof={'checkpoint':'/test/final','run_dir':'/test','checkpoint_sha256':{'model.safetensors':'test-sha'}}
        context.atomic_json(out/'binding.json',proof)
        cfg=dict(dataset='rxr',split='val_unseen',run_dir=proof['run_dir'],weights=proof['checkpoint'],checkpoint_sha256=proof['checkpoint_sha256'],
            selected_episodes=['4'],protocol_sha256=context.sha256(here/'protocol.json'),split_audit_sha256=context.sha256(here/'split_audit.json'),
            memory_mode='empty',force_slot=False,capture_features=False,save_frames=False,evaluation_sha256=context.sha256(native/'evaluate.py'),
            simulator={'habitat':{'simulator':{'turn_angle':15},'environment':{'max_episode_steps':700}}})
        context.atomic_json(folder/'config.json',cfg)
        row=dict(episode_id='4',dataset='rxr',split='val_unseen',scene_id='/data/mp3d/scene/scene.glb',language='en-IN',role='guide',instruction_id='72545',
            instruction=text,executed_actions=[0],action_steps=[{'action':0}],termination='STOP',metrics={'steps_taken':1.,'success':0.})
        context.atomic_json(folder/'episodes/4.json',row)
        context.atomic_json(folder/'workers/0.identity.json',dict(weights=proof['checkpoint'],weight_sha256=proof['checkpoint_sha256'],saved_places_slot=False,stop_logit_bias=0))
        return here,out,native,proof,folder

    def test_result_integrity_rejects_wrong_policy_instruction_and_extra_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            here,out,native,proof,folder=self.result_fixture(Path(tmp))
            with patch.object(context,'HERE',here),patch.object(context,'EVAL_CODE',native),patch.object(context,'model_spec',lambda _: {'output_root':str(out)}),patch.object(context,'resolve_model',lambda _:proof):
                self.assertEqual(context.verify_result('F',partial=True)['verified_episodes'],1)
                with self.assertRaisesRegex(RuntimeError,'not complete'):context.verify_result('F')
                cfg=context.read_json(folder/'config.json');cfg['weights']='/wrong/model';context.atomic_json(folder/'config.json',cfg)
                with self.assertRaisesRegex(RuntimeError,'configuration mismatch'):context.verify_result('F',partial=True)
                cfg['weights']=proof['checkpoint'];context.atomic_json(folder/'config.json',cfg)
                row=context.read_json(folder/'episodes/4.json');row['instruction']='Wrong instruction';context.atomic_json(folder/'episodes/4.json',row)
                with self.assertRaisesRegex(RuntimeError,'instruction mismatch'):context.verify_result('F',partial=True)
                context.atomic_json(folder/'episodes/999.json',row)
                with self.assertRaisesRegex(RuntimeError,'unexpected'):context.verify_result('F',partial=True)

    def test_future_model_requires_a_real_verified_receipt(self):
        with patch.object(context,'model_spec',lambda _: {'run_dir':'/future/E','checkpoint_receipt':'/nonexistent/receipt.json'}):
            with self.assertRaisesRegex(RuntimeError,'unavailable'):context.resolve_model('E')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ExtensionChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    context.atomic_json(context.HERE/'evidence/extension_tests.json',dict(status='complete' if result.wasSuccessful() else 'failed',
        time=context.utc(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors)))
    raise SystemExit(not result.wasSuccessful())
