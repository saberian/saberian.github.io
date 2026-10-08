"""Ten-problem baseline. No training, exposed endpoint, or persistent GPU service."""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import modal
from data import MODEL, MODEL_REVISION, DATA_REVISION, ROOT, digest, messages
from evaluator import EVALUATOR_VERSION, evaluate

app = modal.App("coding-post-training-baseline")
RUNTIME_MODULES = ("data", "evaluator")
results_volume = modal.Volume.from_name("coding-post-training-results", create_if_missing=True)
sandbox_image = modal.Image.debian_slim(python_version="3.12")
cpu_image = modal.Image.debian_slim(python_version="3.12").uv_sync(str(ROOT), uv_version="0.10.10").add_local_python_source(*RUNTIME_MODULES)


def cache_model():
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from data import MODEL, MODEL_REVISION
    snapshot_download(MODEL, revision=MODEL_REVISION, local_dir="/model",
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"])
    config = AutoConfig.from_pretrained("/model", local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
    assert config.model_type == "qwen3" and tokenizer.eos_token_id is not None


gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), extras=["inference"], uv_version="0.10.10")
    .add_local_python_source(*RUNTIME_MODULES, copy=True)
    .run_function(cache_model, cpu=2, memory=16384, timeout=600)
)
_loaded = None


@app.function(image=gpu_image, gpu="A100-80GB", cpu=(2, 2), memory=(16384, 16384),
    timeout=600, startup_timeout=300, retries=0, max_containers=1, scaledown_window=2)
def generate(problems):
    import importlib.metadata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _loaded
    started = time.monotonic()
    assert 1 <= len(problems) <= 9
    torch.manual_seed(42)
    if _loaded is None:
        tokenizer = AutoTokenizer.from_pretrained("/model", local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained("/model", dtype=torch.bfloat16,
            local_files_only=True, attn_implementation="sdpa").to("cuda").eval()
        _loaded = tokenizer, model
    tokenizer, model = _loaded
    loaded_at = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    outputs = []
    for problem in problems:
        prompt = tokenizer.apply_chat_template(messages(problem), tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens > 1024:
            raise ValueError(f"Prompt exceeds agreed limit: {problem['id']}")
        torch.cuda.synchronize()
        before = time.monotonic()
        with torch.inference_mode():
            sequence = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        completion = sequence[0, prompt_tokens:].tolist()
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        terminated = bool(completion and completion[-1] in eos)
        outputs.append({"id": problem["id"], "code": tokenizer.decode(completion, skip_special_tokens=True),
            "prompt_tokens": prompt_tokens, "generated_tokens": len(completion),
            "terminated": terminated, "generation_seconds": time.monotonic() - before})
        print(json.dumps({k: v for k, v in outputs[-1].items() if k != "code"}), flush=True)
    return {"outputs": outputs, "function_seconds": time.monotonic() - started,
        "load_seconds": loaded_at - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "accelerate", "modal")}}


@app.function(image=cpu_image, timeout=1800, retries=0, cpu=(1, 1), memory=(2048, 2048),
    max_containers=1, scaledown_window=2, volumes={"/results": results_volume})
def baseline(artifact, revision, run_id):
    started = time.monotonic()
    problems = artifact["problems"]
    assert len(problems) == 10 and digest(problems) == artifact["sample_sha256"]
    assert artifact["revision"] == DATA_REVISION
    assert all(p["split"] == "development_smoke" for p in problems)
    report = {"run_id": run_id, "revision": revision, "model": MODEL, "model_revision": MODEL_REVISION,
        "evaluator_version": EVALUATOR_VERSION,
        "dataset_revision": DATA_REVISION, "sample_sha256": artifact["sample_sha256"],
        "status": "running", "smoke_checks": [], "reference_checks": [], "results": [], "gpu_batches": []}
    dest = Path("/results") / f"{run_id}.json"

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = dest.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(dest)
        results_volume.commit()

    def grade(problem, code):
        # Retry a platform failure once; never turn it into a correctness label.
        for attempt in range(2):
            try:
                return evaluate(problem, code, app, sandbox_image)
            except Exception:
                if attempt:
                    raise
                time.sleep(1)

    try:
        save()
        probe = {"entry_point": "identity", "cases": [{"args": [7], "expected": 7}]}
        probes = [
            ("correct", "def identity(x): return x", "passed"),
            ("wrong", "def identity(x): return 0", "wrong_answer"),
            ("syntax", "def identity(:", "syntax_error"),
            ("loop", "def identity(x):\n while True: pass", "execution_error"),
            ("output_limit", "def identity(x):\n print('x' * 100000)\n return x", "execution_error"),
            ("network_block", "def identity(x):\n import socket\n try:\n  socket.create_connection(('1.1.1.1', 443), timeout=1).close()\n  return 0\n except OSError:\n  return x", "passed"),
        ]
        for name, code, expected in probes:
            result = grade(probe, code)
            report["smoke_checks"].append({"name": name, **result})
            save()
            if result["status"] != expected:
                raise RuntimeError(f"Evaluator check {name} failed: {result['status']}")
        print("Evaluator smoke checks passed", flush=True)
        for problem in problems:
            result = grade(problem, problem["reference_solution"])
            report["reference_checks"].append({"id": problem["id"], **result})
            save()
            if result["reward"] != 1:
                raise RuntimeError(f"Reference validation failed: {problem['id']}")
        print("All ten reference solutions passed; starting one-problem GPU canary", flush=True)

        # Only prompt-bearing fields reach the GPU function, never solutions/cases.
        for batch in (problems[:1], problems[1:]):
            prompts = [{k: p[k] for k in ("id", "question", "signature")} for p in batch]
            generated = generate.remote(prompts)
            report["gpu_batches"].append({k: v for k, v in generated.items() if k != "outputs"})
            if len(generated["outputs"]) != len(batch):
                raise RuntimeError("Incomplete inference batch")
            for problem, output in zip(batch, generated["outputs"]):
                if problem["id"] != output["id"]:
                    raise RuntimeError("Inference IDs do not match")
                if not output["terminated"]:
                    result = {"status": "truncated", "reward": 0, "passed_cases": 0, "total_cases": len(problem["cases"])}
                else:
                    result = grade(problem, output["code"])
                report["results"].append({**output, **result})
                save()
                print(f"{problem['id']}: {result['status']}", flush=True)
            # Wrong model answers are baseline observations, not infrastructure failures.
            print(f"Completed and saved {len(report['results'])}/10 baseline problems", flush=True)
        report["status"] = "completed"
        report["greedy_accuracy"] = sum(r["reward"] for r in report["results"]) / 10
        report["functional_accuracy_after_extraction"] = report["greedy_accuracy"]
        report["raw_format_compliance"] = sum(r.get("raw_format_compliant", False) for r in report["results"]) / 10
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        save()
    return report


@app.local_entrypoint()
def main():
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise RuntimeError("Commit and push the exact runner revision before a paid run")
    remote_revision = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=ROOT, text=True).strip()
    if revision != remote_revision:
        raise RuntimeError("Runner revision must match the pushed upstream branch")
    artifact = json.loads((ROOT / "data" / "development-smoke.json").read_text())
    manifest = json.loads((ROOT / "manifests" / "development-smoke.json").read_text())
    if artifact["sample_sha256"] != manifest["sample_sha256"]:
        raise RuntimeError("Local data differs from committed manifest")
    run_id = datetime.now(timezone.utc).strftime("baseline-%Y%m%dT%H%M%SZ-") + revision[:8]
    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    call = baseline.spawn(artifact, revision, run_id)
    (out / f"{run_id}-launch.json").write_text(json.dumps({"run_id": run_id, "revision": revision,
        "app_id": app.app_id, "function_call_id": call.object_id, "volume": "coding-post-training-results"}, indent=2) + "\n")
    print(f"Started {run_id}; call {call.object_id}; results volume coding-post-training-results", flush=True)
    report = call.get()
    (out / f"{run_id}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Baseline completed: {report['greedy_accuracy']:.0%} greedy accuracy (10 development problems)", flush=True)
