import copy
import unittest

from data import DATA_REVISION, digest
from discovery import can_start_batch, summarize, validate_artifact, validate_outputs
from discovery_data import family_assignments


class DiscoveryContracts(unittest.TestCase):
    def test_prompt_length_counts_tokens_not_batch_encoding_fields(self):
        from data import tokenize_prompt
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.options = kwargs
                return 'rendered chat'
            def __call__(self, rendered, **kwargs):
                self.rendered, self.token_options = rendered, kwargs
                return {'input_ids': list(range(1200)), 'attention_mask': [1] * 1200}
        tokenizer = Tokenizer()
        result = tokenize_prompt({'question': 'test', 'signature': 'def f():'}, tokenizer)
        self.assertEqual(len(result['input_ids']), 1200)
        self.assertEqual(tokenizer.options, {'tokenize': False, 'add_generation_prompt': True})
        self.assertFalse(tokenizer.token_options['add_special_tokens'])


    def test_batched_eos_padding_is_not_counted_as_generated_text(self):
        from modal_app import trim_completion
        self.assertEqual(trim_completion([5, 9, 9, 9], 9), ([5, 9], True))
        self.assertEqual(trim_completion([5, 8, 9, 9], [8, 9]), ([5, 8], True))
        self.assertEqual(trim_completion([5, 6], [8, 9]), ([5, 6], False))


    def test_transitive_families_include_ineligible_bridges_and_reserve_smoke(self):
        def row(i, seeds):
            return {'question_id': i, 'metadata': {'seed_ids': repr(seeds)},
                'question': i, 'solution': f'def {i}(): return {len(i)}'}
        rows = [row('a', [1]), row('bridge', ['1', '2']), row('c', [2]), row('other', [3])]
        assignments = family_assignments(rows, {'a'})
        self.assertEqual(assignments, family_assignments(list(reversed(rows)), {'a'}))
        self.assertEqual(assignments['a'], assignments['c'])
        self.assertEqual(assignments['c']['split'], 'development')
        self.assertNotEqual(assignments['a']['family_id'], assignments['other']['family_id'])

    def test_normalized_duplicates_are_grouped(self):
        rows = [{'question_id': str(i), 'metadata': {'seed_ids': '[]'}, 'question': str(i), 'solution': s}
            for i, s in enumerate(('def f(x): return x', 'def f(x):\n    return x # comment'))]
        result = family_assignments(rows, set())
        self.assertEqual(result['0'], result['1'])

    def test_budget_reserves_next_worst_case_call(self):
        self.assertTrue(can_start_batch(0))
        self.assertTrue(can_start_batch(2500))
        self.assertFalse(can_start_batch(3200))

    def fixture(self):
        problems = [{'id': str(i), 'family_id': str(i), 'split': 'alignment_train',
            'stratum': 'broad' if i < 50 else 'medium_hard'} for i in range(100)]
        pools = {'assignments': {p['id']: {k: p[k] for k in ('family_id', 'split')} for p in problems}}
        artifact = {'problems': problems, 'revision': DATA_REVISION,
            'sample_sha256': digest(problems), 'split_manifest_sha256': digest(pools)}
        return artifact, {'sample_sha256': artifact['sample_sha256']}, pools

    def test_frozen_data_and_holdout_guard(self):
        a, m, p = self.fixture()
        validate_artifact(a, m, p)
        a['problems'][0]['split'] = 'final_test'
        a['sample_sha256'] = m['sample_sha256'] = digest(a['problems'])
        with self.assertRaises(ValueError):
            validate_artifact(a, m, p)
        a, m, p = self.fixture()
        p['assignments']['0']['split'] = 'development'
        with self.assertRaises(ValueError):
            validate_artifact(a, m, p)

    def test_response_accounting_and_mixed_groups(self):
        batch = [{'id': 'a', 'stratum': 'broad', 'split': 'alignment_train', 'difficulty': 'hard'}]
        outputs = [{'id': 'a', 'sample_index': i, 'reward': int(i < 2), 'code': str(i),
            'status': 'passed' if i < 2 else 'wrong_answer', 'generated_tokens': 10} for i in range(4)]
        validate_outputs(batch, outputs)
        self.assertEqual(summarize(outputs, batch)['pass_histogram']['2'], 1)
        self.assertEqual(summarize(outputs, batch)['sample_accuracy'], 0.5)
        bad = copy.deepcopy(outputs)
        bad[3]['sample_index'] = 0
        with self.assertRaises(ValueError):
            validate_outputs(batch, bad)
