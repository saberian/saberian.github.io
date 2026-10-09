import unittest
from overfit_data import select

class OverfitContracts(unittest.TestCase):
    def test_selection_has_seven_failures_eight_passes_and_no_holdout(self):
        ps=[{'id':str(i),'split':'sft_train' if i<64 else 'development'} for i in range(96)]
        scores=[{'id':str(i),'reward':int(i>=7)} for i in range(64)]
        chosen=select(ps,scores)
        self.assertEqual(len(chosen),15)
        self.assertEqual(len({p['id'] for p in chosen}),15)
        self.assertEqual(sum(int(p['id'])<7 for p in chosen),7)
        self.assertTrue(all(p['split']=='sft_train' for p in chosen))
        self.assertEqual(chosen,select(list(reversed(ps)),list(reversed(scores))))
        with self.assertRaises(ValueError):select(ps,scores[:-1])
        scores[0]['reward']=1
        with self.assertRaises(ValueError):select(ps,scores)
