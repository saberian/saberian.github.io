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
