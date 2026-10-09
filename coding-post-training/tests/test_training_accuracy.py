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
        self.assertIn('batch-33.json',str(training_eval_destination(run,33)))
        with self.assertRaises(ValueError):training_eval_destination(run,34)
        ps=[{'id':str(i),'family_id':str(i),'implementation_key':str(i)} for i in range(300)]
        pools={'assignments':{p['id']:{'family_id':p['family_id'],'split':'final_test'} for p in ps}}
        validate_test(ps,pools,[])
        with self.assertRaises(ValueError):validate_test(ps,pools,[ps[0]])
        pools['assignments']['0']['split']='sft_train'
        with self.assertRaises(ValueError):validate_test(ps,pools,[])
