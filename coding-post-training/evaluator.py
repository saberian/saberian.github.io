"""Shared extraction/grading. Execution is delegated to an isolated runtime."""

import ast
import json
import re
from data import json_value

EVALUATOR_VERSION = "python-output-v2"


def extract_python(response):
    """Accept raw source or exactly one whole-response ```python block.

    Strip fence delimiters only, never repair code or search prose for snippets.
    Raw source is returned byte-for-byte unchanged. Outer whitespace around a
    fenced block is permitted; nested/multiple fence lines are not.
    """
    if not isinstance(response, str) or not response.strip() or len(response.encode()) > 65536:
        raise ValueError("Empty or oversized response")
    text = response.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```python[ \t]*\r?\n(.*?)\r?\n```", text, flags=re.DOTALL)
        if not match or not match[1].strip() or re.search(r"(?m)^[ \t]*```", match[1]):
            raise ValueError("Expected one complete Python block with no surrounding prose")
        return match[1], "fenced_python"
    return response, "raw_python"

# This driver and its supervisor run INSIDE the disposable sandbox. Neither sees
# expected answers. The controller outside the sandbox owns all comparisons.
DRIVER = r'''
import contextlib, json, sys
request = json.loads(sys.stdin.read())
namespace = {}
with contextlib.redirect_stdout(sys.stderr):
    exec(compile(request["code"], "candidate.py", "exec"), namespace)
    fn = namespace[request["entry_point"]]
    outputs = [fn(*args) for args in request["inputs"]]
print(json.dumps(outputs, allow_nan=False))
'''

SUPERVISOR = r'''
import json, resource, subprocess, sys, tempfile
driver = sys.argv[1]
request = sys.stdin.read()
def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 3))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
    resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", driver], input=request.encode(),
            stdout=out, stderr=err, timeout=10, preexec_fn=limits)
        out.seek(0)
        payload = out.read(65537)
        result = {"status": "ok" if proc.returncode == 0 else "execution_error",
                  "returncode": proc.returncode, "output": payload.decode(errors="replace")}
    except subprocess.TimeoutExpired:
        result = {"status": "timeout", "output": ""}
    print(json.dumps(result))
'''


def same_value(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(same_value(a, b) for a, b in zip(actual, expected))
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(same_value(actual[k], v) for k, v in expected.items())
    return actual == expected


def grade_payload(payload, cases):
    result = {"status": payload.get("status", "invalid_output"), "passed_cases": 0, "total_cases": len(cases), "reward": 0}
    if result["status"] != "ok":
        if result["status"] not in ("timeout", "execution_error"):
            result["status"] = "invalid_output"
        return result
    try:
        raw = payload["output"]
        if len(raw.encode()) > 65536:
            raise ValueError("Output limit")
        outputs = json.loads(raw)
        if not isinstance(outputs, list) or len(outputs) != len(cases) or not json_value(outputs):
            raise ValueError("Invalid output protocol")
    except (ValueError, TypeError, KeyError):
        result["status"] = "invalid_output"
        return result
    result["passed_cases"] = sum(same_value(output, case["expected"]) for output, case in zip(outputs, cases))
    result["reward"] = int(result["passed_cases"] == len(cases) and len(cases) > 0)
    result["status"] = "passed" if result["reward"] else "wrong_answer"
    return result


def evaluate_with_runner(problem, response, runner):
    """The common policy for local replay and future cloud/RL evaluations."""
    metadata = {"evaluator_version": EVALUATOR_VERSION, "output_format": "rejected", "raw_format_compliant": False}
    failure = {"reward": 0, "passed_cases": 0, "total_cases": len(problem["cases"])}
    try:
        code, output_format = extract_python(response)
        metadata["output_format"] = output_format
    except ValueError:
        return {**metadata, **failure, "status": "format_error"}
    try:
        ast.parse(code)
    except SyntaxError:
        return {**metadata, **failure, "status": "syntax_error"}
    metadata["raw_format_compliant"] = output_format == "raw_python"
    request = {"code": code, "entry_point": problem["entry_point"], "inputs": [c["args"] for c in problem["cases"]]}
    payload, runtime_metadata = runner(request)
    return {**metadata, **grade_payload(payload, problem["cases"]), **runtime_metadata}


def evaluate(problem, response, app, image):
    return evaluate_with_runner(problem, response, lambda request: run_modal(request, app, image))


def run_modal(request, app, image):
    import modal
    # Credentials, model/data volumes, expected answers and scoring never enter.
    sandbox = modal.Sandbox.create("sleep", "60", app=app, image=image,
        timeout=60, cpu=(1, 1), memory=(512, 512), block_network=True,
        include_oidc_identity_token=False)
    try:
        process = sandbox.exec("python", "-I", "-c", SUPERVISOR, DRIVER, timeout=20)
        process.stdin.write(json.dumps(request).encode())
        process.stdin.write_eof()
        process.stdin.drain()
        raw = process.stdout.read()
        process.wait()
        if process.returncode != 0:
            raise RuntimeError("Evaluator supervisor failed; sample unscored")
        try:
            payload = json.loads(raw)
        except ValueError as exc:
            raise RuntimeError("Evaluator protocol failed; sample unscored") from exc
        return payload, {"sandbox_id": sandbox.object_id}
    finally:
        sandbox.terminate()
