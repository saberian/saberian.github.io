"""Replay saved responses locally. Never load a model or contact Modal."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from data import ROOT, digest
from evaluator import EVALUATOR_VERSION, evaluate_with_runner
from local_runner import DEFAULT_IMAGE, image_identity, run_docker


def replay(run_path, data_path, output_path, image):
    if output_path.resolve() in (run_path.resolve(), data_path.resolve()):
        raise ValueError("Replay output must not overwrite source artifacts")
    if output_path.exists():
        raise ValueError("Output already exists; choose a new report path")
    original = json.loads(run_path.read_text())
    artifact = json.loads(data_path.read_text())
    if original["status"] != "completed" or digest(artifact["problems"]) != artifact["sample_sha256"] or original["sample_sha256"] != artifact["sample_sha256"]:
        raise ValueError("Completed run and matching immutable sample required")
    problems = {p["id"]: p for p in artifact["problems"]}
    responses = original["results"]
    ids = [r["id"] for r in responses]
    if len(set(ids)) != len(ids) or set(ids) != set(problems):
        raise ValueError("Response IDs must match the problem set exactly")
    identity = image_identity(image)
    runner = lambda request: run_docker(request, identity["id"])
    report = {
        "evaluator_version": EVALUATOR_VERSION, "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_run_id": original["run_id"], "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "sample_sha256": original["sample_sha256"], "dataset_revision": original["dataset_revision"],
        "model": original["model"], "model_revision": original["model_revision"],
        "evaluator_source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ("evaluator.py", "local_runner.py", "reevaluate.py", "data.py")},
        "checkout_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkout_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
        "runtime": {"backend": "local_docker", "image": identity},
        "new_model_generations": 0, "modal_calls": 0,
        "original_strict_accuracy": original["greedy_accuracy"], "reference_checks": [], "results": [],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(output_path)

    try:
        save()
        # Verify the new runtime against the original oracles before any replay.
        for problem in problems.values():
            result = evaluate_with_runner(problem, problem["reference_solution"], runner)
            report["reference_checks"].append({"id": problem["id"], **result})
            save()
            if result["reward"] != 1:
                raise RuntimeError(f"Reference validation failed: {problem['id']}")
        for response in responses:
            problem = problems[response["id"]]
            if not response["terminated"]:
                result = {"status": "truncated", "reward": 0, "passed_cases": 0,
                    "total_cases": len(problem["cases"]), "raw_format_compliant": False}
            else:
                result = evaluate_with_runner(problem, response["code"], runner)
            report["results"].append({"id": response["id"],
                "response_sha256": hashlib.sha256(response["code"].encode()).hexdigest(),
                "original_status": response["status"], **result})
            save()
            print(f"{response['id']}: {result['status']} ({result['passed_cases']}/{result['total_cases']} cases)", flush=True)
        count = len(responses)
        report["functional_accuracy_after_extraction"] = sum(r["reward"] for r in report["results"]) / count
        report["raw_format_compliance"] = sum(r["raw_format_compliant"] for r in report["results"]) / count
        report["passed_cases"] = sum(r["passed_cases"] for r in report["results"])
        report["total_cases"] = sum(r["total_cases"] for r in report["results"])
        report["status"] = "completed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    print(f"Functional accuracy: {report['functional_accuracy_after_extraction']:.0%}; raw format compliance: {report['raw_format_compliance']:.0%}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/development-smoke.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    replay(args.run, args.data, args.output, args.image)
