"""Prepare a small, reproducible development-only sample without executing code."""

import ast
import hashlib
import json
from pathlib import Path

DATASET = "KodCode/KodCode-Light-RL-10K"
DATA_REVISION = "dcf78a8bbba9a613b596ce993c4921a38687dfcc"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
ROOT = Path(__file__).resolve().parent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def json_value(value):
    """Strict JSON values; reject tuples, nonfinite numbers and ambiguous dict keys."""
    if value is None or type(value) in (str, int, bool):
        return True
    if type(value) is list:
        return all(json_value(x) for x in value)
    if type(value) is dict:
        return all(type(k) is str and json_value(v) for k, v in value.items())
    return False


def extract_cases(source, entry_point):
    """Accept only direct equality assertions with literal inputs/expected values.

    All test bodies must be understood. Never silently drop unsupported assertions.
    This parser is a format restriction, not an execution sandbox.
    """
    cases = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module == "solution":
            if any(a.name != entry_point or a.asname for a in node.names):
                raise ValueError("Unsupported solution import")
            continue
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            raise ValueError("Unsupported test module statement")
        if node.args.args or node.args.posonlyargs or node.args.kwonlyargs or node.args.vararg or node.args.kwarg or node.decorator_list:
            raise ValueError("Fixtures/decorators unsupported")
        for statement in node.body:
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str):
                continue
            if not isinstance(statement, ast.Assert):
                raise ValueError("Unsupported test body")
            comparison = statement.test
            if not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1 or not isinstance(comparison.ops[0], ast.Eq):
                raise ValueError("Only direct equality assertions supported")
            call = comparison.left
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != entry_point or call.keywords:
                raise ValueError("Only positional direct function calls supported")
            args = [ast.literal_eval(arg) for arg in call.args]
            expected = ast.literal_eval(comparison.comparators[0])
            if not json_value(args) or not json_value(expected):
                raise ValueError("Unsupported case value")
            cases.append({"args": args, "expected": expected})
    if not 3 <= len(cases) <= 20:
        raise ValueError("Require 3–20 literal cases")
    return cases


def parse_problem(row):
    """Shared pure-function eligibility and literal-test contract."""
    if any(term in row["question"].lower() for term in ("complexity", "print", "file", "random", "input()", "database", "numpy")):
        raise ValueError("Outside initial pure-function scope")
    info = row["test_info"]
    if len(info) != 1:
        raise ValueError("Ambiguous entry point")
    entry = info[0]["function_name"]
    cases = extract_cases(row["test"], entry)
    tree = ast.parse(row["solution"])
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    if len(definitions) != 1 or definitions[0].decorator_list:
        raise ValueError("Missing/ambiguous entry-point definition")
    node = definitions[0]
    signature = f"def {entry}({ast.unparse(node.args)}):"
    seed_ids = ast.literal_eval(row["metadata"].get("seed_ids", "[]"))
    if not isinstance(seed_ids, list) or not all(type(x) in (str, int) for x in seed_ids):
        raise ValueError("Invalid family metadata")
    return {"id": row["question_id"], "question": row["question"],
        "entry_point": entry, "signature": signature,
        "reference_solution": row["solution"], "cases": cases,
        "source_seed_ids": seed_ids}


def prepare():
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    source_dir = ROOT / "data" / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(DATASET, "data/train-00000-of-00001.parquet", repo_type="dataset", revision=DATA_REVISION, local_dir=source_dir)
    hf_hub_download(DATASET, "README.md", repo_type="dataset", revision=DATA_REVISION, local_dir=source_dir)
    rows = pq.read_table(path).to_pylist()
    exclusions = json.loads((ROOT / "manifests" / "audit-exclusions.json").read_text())
    selected, rejected, families = [], {}, set()
    for row in sorted(rows, key=lambda r: digest([42, r["question_id"]])):
        try:
            if row["question_id"] in exclusions:
                raise ValueError("Manual audit: " + exclusions[row["question_id"]])
            parsed = parse_problem(row)
            seed_ids = parsed["source_seed_ids"]
            # Preserve the original smoke selection; discovery accepts numeric IDs.
            if not all(isinstance(x, str) for x in seed_ids):
                raise ValueError("Invalid family metadata")
            family_keys = set(seed_ids) | {digest(row["question"].strip()), digest(row["solution"].strip())}
            if family_keys & families:
                raise ValueError("Duplicate family in development sample")
            problem = {**parsed, "split": "development_smoke"}
            selected.append(problem)
            families.update(family_keys)
            if len(selected) == 10:
                break
        except (ValueError, SyntaxError, TypeError, KeyError) as exc:
            reason = str(exc)
            rejected[reason] = rejected.get(reason, 0) + 1
    if len(selected) != 10:
        raise RuntimeError(f"Only {len(selected)} eligible problems; no run prepared")
    artifact = {
        "dataset": DATASET, "revision": DATA_REVISION,
        "source_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "selection": "First 10 eligible records sorted by SHA256([42, question_id]); development only",
        "examined_rejections": rejected, "sample_sha256": digest(selected),
        "problems": selected,
    }
    dest = ROOT / "data" / "development-smoke.json"
    dest.write_text(json.dumps(artifact, indent=2) + "\n")
    manifest_dir = ROOT / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    manifest = {k: v for k, v in artifact.items() if k != "problems"}
    manifest["reserved_development"] = [{"id": p["id"], "source_seed_ids": p["source_seed_ids"]} for p in selected]
    (manifest_dir / "development-smoke.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(selected)} development problems: {dest}")
    for p in selected:
        print(p["id"], p["signature"], len(p["cases"]), "cases")


def messages(problem):
    return [
        {"role": "system", "content": "Write only Python source code. Do not include Markdown fences, explanations, tests, or interactive input."},
        {"role": "user", "content": problem["question"] + "\n\nRequired function signature:\n" + problem["signature"]},
    ]


def tokenize_prompt(problem, tokenizer, **kwargs):
    """Use exactly the same chat rendering and tokenization locally and on GPU."""
    rendered = tokenizer.apply_chat_template(messages(problem), tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False, **kwargs)


if __name__ == "__main__":
    prepare()


def evaluation_batches(problems):
    """One canary, followed by batches of at most nine (including the tail)."""
    return [problems[:1]]+[problems[i:i+9] for i in range(1,len(problems),9)]
