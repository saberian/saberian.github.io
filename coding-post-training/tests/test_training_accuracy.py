import copy
import unittest
from training_accuracy import training_problems
from modal_app import training_eval_destination

class TrainingAccuracyContracts(unittest.TestCase):
    def test_exact_training_membership_excludes_development(self):
        ps=[{'id':str(i),'split':'sft_train' if i<64 else 'development'} for i in range(96)]
        records=[{'id':str(i)} for i in range(64)]
        self.assertEqual(training_problems({'problems':ps},records),ps[:64])
        for change in ('id','split'):
            bad=copy.deepcopy(ps);bad[0][change]='development' if change=='split' else 'different'
            with self.assertRaises(ValueError):training_problems({'problems':bad},records)
    def test_results_are_separate_from_checkpoint_development_evaluation(self):
        run='train-eval-20261009T160000Z-1234abcd'
        self.assertEqual(str(training_eval_destination(run,7)),f'/results/{run}/batch-7.json')
        for bad,index in [(run,8),(run,-1),('../adapter',0),('sft-20261009T160000Z-1234abcd',0)]:
            with self.assertRaises(ValueError):training_eval_destination(bad,index)

    def test_base_policy_uses_unadapted_loader_and_preserves_saved_batches(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from modal_app import base_training_generate
        with tempfile.TemporaryDirectory() as d:
            destination=Path(d)/'batch-0.json'
            def generate(problems,*,samples,checkpoint):
                self.assertEqual(samples,1)
                outputs=[{'id':problems[0]['id'],'code':'def f(): return 1'}]
                checkpoint(outputs)
                return {'outputs':outputs}
            with patch('modal_app.training_eval_destination',return_value=destination),patch('modal_app.results_volume') as volume,patch('modal_app.generate_batch',side_effect=generate) as loader:
                result=base_training_generate.local([{'id':'a'}],'train-eval-20261009T160000Z-1234abcd',0)
                self.assertEqual(json.loads(destination.read_text()),result)
                volume.commit.assert_called_once()
                loader.assert_called_once()
                with self.assertRaises(ValueError):base_training_generate.local([{'id':'a'}],'train-eval-20261009T160000Z-1234abcd',0)

    def test_test_batch_bounds_and_disjointness(self):
        from test_data import validate_test
        run='test-eval-20261009T160000Z-1234abcd'
        self.assertIn('batch-34.json',str(training_eval_destination(run,34)))
        with self.assertRaises(ValueError):training_eval_destination(run,35)
        ps=[{'id':str(i),'family_id':str(i),'implementation_key':str(i)} for i in range(300)]
        pools={'assignments':{p['id']:{'family_id':p['family_id'],'split':'final_test'} for p in ps}}
        validate_test(ps,pools,[])
        with self.assertRaises(ValueError):validate_test(ps,pools,[ps[0]])
        pools['assignments']['0']['split']='sft_train'
        with self.assertRaises(ValueError):validate_test(ps,pools,[])

    def test_every_planned_batch_is_accepted_including_tail(self):
        from data import evaluation_batches
        for split,n in [('train',64),('test',300),('validation',32)]:
            batches=evaluation_batches(list(range(n)))
            self.assertEqual([x for batch in batches for x in batch],list(range(n)))
            for i,batch in enumerate(batches):
                self.assertTrue(1<=len(batch)<=9)
                training_eval_destination(f'{split}-eval-20261009T160000Z-1234abcd',i)
            with self.assertRaises(ValueError):training_eval_destination(f'{split}-eval-20261009T160000Z-1234abcd',len(batches))

    def test_resume_preserves_answers_and_rejects_partial_batches(self):
        from training_accuracy import resume_report
        from data import evaluation_batches
        ps=[{'id':str(i)} for i in range(300)]
        keys=('policy','split','model_revision','source_sft_run_id','evaluation_problems_sha256','training_export_sha256','adapter_files','evaluator_version','decoding','reference_runtime','budget_usd')
        expected={k:None for k in keys};expected['revision']='new'
        old=expected|{'status':'failed','app_id':'old-app','error':'boundary','revision':'old',
            'gpu_batches':[{}]*34,'raw_outputs':[{'id':str(i),'sample_index':0} for i in range(298)],'results':[{'id':'0'}]}
        recovered=resume_report(copy.deepcopy(old),expected,ps)
        self.assertEqual(recovered['raw_outputs'],old['raw_outputs'])
        self.assertEqual(len(evaluation_batches(ps)[len(recovered['gpu_batches']):][0]),2)
        self.assertEqual(recovered['attempts'][0]['revision'],'old')
        for key in ('raw_outputs','gpu_batches'):
            bad=copy.deepcopy(old);bad[key].pop()
            with self.assertRaises(ValueError):resume_report(bad,expected,ps)
        bad=copy.deepcopy(old);bad['model_revision']='changed'
        with self.assertRaises(ValueError):resume_report(bad,expected,ps)

    def test_adapter_stages_are_separate_and_scope_is_validation_only(self):
        from modal_app import adapter_destination
        from training_accuracy import main
        run='sft-20261009T160000Z-1234abcd'
        self.assertNotEqual(adapter_destination(run,'train'),adapter_destination(run,'overfit'))
        self.assertEqual(str(adapter_destination(run,'overfit')),f'/results/{run}/overfit/adapter')
        with self.assertRaises(ValueError):adapter_destination(run,'../train')
        for split in ('train','test'):
            with self.assertRaises(ValueError):main('sft',split,checkpoint='overfit')

    def test_recovery_reuses_completed_remote_batch_after_lost_response(self):
        from training_accuracy import recover_saved_batches
        ps=[{'id':str(i)} for i in range(32)]
        report={'gpu_batches':[{},{}],'raw_outputs':[{'id':str(i),'sample_index':0} for i in range(10)]}
        saved={'outputs':[{'id':str(i),'sample_index':0} for i in range(10,19)]}
        recover_saved_batches(report,ps,lambda i:saved if i==2 else None)
        self.assertEqual(len(report['raw_outputs']),19)
        self.assertEqual(len(report['gpu_batches']),3)
        self.assertTrue(report['gpu_batches'][2]['recovered_from_volume'])
        with self.assertRaises(ValueError):recover_saved_batches(report,ps,lambda i:saved)

    def test_missing_volume_checkpoint_ends_recovery_without_gpu_retry(self):
        from unittest.mock import Mock
        from training_accuracy import read_saved_batch
        volume=Mock()
        volume.read_file.side_effect=FileNotFoundError('No next checkpoint')
        self.assertIsNone(read_saved_batch(volume,'run',3))
        volume.read_file.side_effect=None
        volume.read_file.return_value=iter([b'{"outputs": []}'])
        self.assertEqual(read_saved_batch(volume,'run',3),{'outputs':[]})
        volume.read_file.side_effect=ConnectionError('offline')
        with self.assertRaises(ConnectionError):read_saved_batch(volume,'run',3)
