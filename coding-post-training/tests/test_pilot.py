import copy
import unittest

from discovery import can_start_batch
from pilot_data import implementation_key, training_record, validate_splits


class Tokenizer:
    eos_token_id = 99
    def __init__(self, ids=None):
        self.ids = ids if ids is not None else [1, 2, 3, 4, 5, 99, 12]
    def apply_chat_template(self, turns, **kwargs):
        return 'full' if turns[-1]['role'] == 'assistant' else 'prompt'
    def __call__(self, text, **kwargs):
        return {'input_ids': self.ids if text == 'full' else [1, 2, 3]}
    def decode(self, ids):
        return '\n' if ids == [12] else 'unexpected'


class PilotContracts(unittest.TestCase):
    def problem(self):
        return {'id': 'a', 'question': 'Add one', 'signature': 'def f(x):',
            'reference_solution': 'def f(x): return x + 1', 'cases': ['HIDDEN_TEST']}

    def test_answer_only_labels_include_eos_without_turn_suffix(self):
        r = training_record(self.problem(), Tokenizer())
        self.assertEqual(r['input_ids'], [1, 2, 3, 4, 5, 99])
        self.assertEqual(r['labels'], [-100, -100, -100, 4, 5, 99])
        self.assertEqual(r['target_tokens'], 3)
        self.assertNotIn('cases', r)
        self.assertNotIn('HIDDEN_TEST', str(r))
        self.assertEqual(r['messages'][-1]['content'], self.problem()['reference_solution'])

    def test_bad_prefix_missing_eos_and_long_targets_are_rejected(self):
        for ids in ([9, 2, 3, 4, 99], [1, 2, 3, 4], [1, 2, 3] + [4] * 512 + [99]):
            with self.subTest(ids_length=len(ids)), self.assertRaises(ValueError):
                training_record(self.problem(), Tokenizer(ids))

    def test_top_level_demonstrations_cannot_become_sft_targets(self):
        p = self.problem(); p['reference_solution'] += '\nprint(f(1))'
        with self.assertRaises(ValueError):
            training_record(p, Tokenizer())

    def test_renamed_implementations_are_detected(self):
        self.assertEqual(implementation_key('def f(x): return x + 1'),
            implementation_key('def renamed(value: int):\n "documentation"\n return value + 1'))
        self.assertNotEqual(implementation_key('def f(x): return x + 1'),
            implementation_key('def f(x): return x - 1'))

    def test_split_counts_family_overlap_and_assignment_guards(self):
        ps = [{'id': str(i), 'family_id': str(i), 'implementation_key': str(i),
            'split': 'sft_train' if i < 64 else 'development'} for i in range(96)]
        pools = {'assignments': {p['id']: {k:p[k] for k in ('family_id','split')} for p in ps}}
        validate_splits(ps, pools)
        for key in ('id', 'family_id', 'implementation_key'):
            bad = copy.deepcopy(ps); bad[64][key] = bad[0][key]
            with self.assertRaises(ValueError): validate_splits(bad, pools)
        bad = copy.deepcopy(ps); bad[64]['split'] = 'final_test'
        with self.assertRaises(ValueError): validate_splits(bad, pools)

    def test_pilot_budget_reserves_whole_next_call(self):
        self.assertTrue(can_start_batch(0, budget_usd=1.5))
        self.assertFalse(can_start_batch(1200, budget_usd=1.5))
