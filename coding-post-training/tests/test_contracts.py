import json
import subprocess
import sys
import shutil
import tempfile
import unittest
from pathlib import Path

from data import extract_cases, messages
from evaluator import DRIVER, grade_payload


class Contracts(unittest.TestCase):
    def test_cloud_source_bundle_imports_without_checkout(self):
        from modal_app import RUNTIME_MODULES
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for module in ("modal_app", *RUNTIME_MODULES):
                shutil.copy(root / f"{module}.py", directory)
            subprocess.run([sys.executable, "-I", "-c",
                f"import sys; sys.path.insert(0, {directory!r}); import modal_app"],
                cwd=directory, capture_output=True, text=True, check=True, timeout=10)

    def test_no_solutions_or_cases_in_prompt(self):
        prompt = messages({"question": "Add one", "signature": "def f(x):",
            "reference_solution": "SECRET_SOLUTION", "cases": ["SECRET_TEST"]})
        self.assertNotIn("SECRET", json.dumps(prompt))
        self.assertIn("def f(x):", prompt[1]["content"])

    def test_literal_cases_and_unsupported_test_are_not_silently_dropped(self):
        source = "from solution import f\ndef test_f():\n assert f(1) == 2\n assert f(2) == 3\n assert f(3) == 4\n"
        self.assertEqual(extract_cases(source, "f")[0], {"args": [1], "expected": 2})
        with self.assertRaises(ValueError):
            extract_cases(source + " assert f(4) > 0\n", "f")
        with self.assertRaises(ValueError):
            extract_cases(source.replace("f(1)", "f(make_input())"), "f")

    def test_grader_owns_reward_and_checks_every_case(self):
        cases = [{"expected": 1}, {"expected": [2]}]
        self.assertEqual(grade_payload({"status": "ok", "output": "[1, [2]]"}, cases)["reward"], 1)
        self.assertEqual(grade_payload({"status": "ok", "output": "[1, [9]]", "reward": 1}, cases)["reward"], 0)
        self.assertEqual(grade_payload({"status": "ok", "output": "[1]"}, cases)["status"], "invalid_output")
        self.assertEqual(grade_payload({"status": "ok", "output": "[true, [2]]"}, cases)["reward"], 0)
        self.assertEqual(grade_payload({"status": "ok", "output": "[]"}, [])["reward"], 0)

    def test_malformed_and_oversized_output_fail(self):
        for text in ('{"reward": 1}', '[NaN]', 'x' * 65537):
            self.assertEqual(grade_payload({"status": "ok", "output": text}, [{"expected": 1}])["status"], "invalid_output")

    def test_driver_protocol_with_only_handwritten_trusted_fixture(self):
        # No downloaded or generated program is executed on the local machine.
        request = {"code": "def f(x): return x + 1", "entry_point": "f", "inputs": [[1], [2]]}
        process = subprocess.run([sys.executable, "-I", "-c", DRIVER], input=json.dumps(request),
            text=True, capture_output=True, check=True, timeout=5)
        self.assertEqual(json.loads(process.stdout), [2, 3])


if __name__ == "__main__":
    unittest.main()
