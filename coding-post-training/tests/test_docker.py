"""Opt-in real local isolation checks: RUN_DOCKER_TESTS=1 uv run ..."""

import os
import unittest

from evaluator import evaluate_with_runner
from local_runner import image_identity, run_docker


@unittest.skipUnless(os.environ.get('RUN_DOCKER_TESTS') == '1', 'Requires local Docker')
class DockerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = image_identity()['id']

    def grade(self, code):
        problem = {'entry_point': 'f', 'cases': [{'args': [7], 'expected': 7}]}
        return evaluate_with_runner(problem, code, lambda request: run_docker(request, self.image))

    def test_canary_raw_and_fenced_correctness(self):
        for code in ('def f(x): return x', '```python\ndef f(x): return x\n```'):
            self.assertEqual(self.grade(code)['status'], 'passed')
        self.assertEqual(self.grade('def f(x): return 0')['status'], 'wrong_answer')

    def test_network_and_root_writes_blocked(self):
        code = '''def f(x):
 import socket
 try:
  socket.create_connection(('1.1.1.1', 443), timeout=1).close()
  return 0
 except OSError:
  pass
 try:
  open('/must_not_write', 'w').write('bad')
  return 0
 except OSError:
  return x
'''
        self.assertEqual(self.grade(code)['status'], 'passed')

    def test_resource_limits(self):
        self.assertEqual(self.grade('def f(x):\n while True: pass')['status'], 'execution_error')
        self.assertEqual(self.grade("def f(x):\n print('x' * 100000)\n return x")['status'], 'execution_error')
        self.assertEqual(self.grade('def f(x):\n import time\n time.sleep(20)\n return x')['status'], 'timeout')
