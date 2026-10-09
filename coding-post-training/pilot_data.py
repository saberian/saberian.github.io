"""Prepare, audit, then freeze 64 SFT demonstrations and 32 development tasks."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from data import ROOT, DATASET, DATA_REVISION, MODEL, MODEL_REVISION, digest, messages, parse_problem, tokenize_prompt
from discovery_data import write_immutable
from evaluator import evaluate_with_runner
from local_runner import image_identity, run_docker

COUNTS = {'sft_train': 64, 'development': 32}


def implementation_key(source):
    """Conservative duplicate screen: erase user identifier names, retain operations."""
    tree = ast.parse(source)
    names = {}
    def canonical(name):
        if name not in names:
            names[name] = f'identifier_{len(names)}'
        return names[name]
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            node.name = canonical(node.name)
            node.returns = None
            node = self.generic_visit(node)
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
            return node
        def visit_arg(self, node):
            node.arg = canonical(node.arg)
            node.annotation = None
            return node
        def visit_Name(self, node):
            node.id = canonical(node.id)
            return node
    return digest(ast.dump(Normalize().visit(tree), include_attributes=False))


def training_record(problem, tokenizer):
    tree = ast.parse(problem['reference_solution'])
    if any(isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) for n in tree.body):
        raise ValueError('Reference contains top-level demonstration calls')
    prompt = tokenize_prompt(problem, tokenizer)['input_ids']
    turns = messages(problem) + [{'role': 'assistant', 'content': problem['reference_solution']}]
    rendered = tokenizer.apply_chat_template(turns, tokenize=False, add_generation_prompt=False)
    ids = tokenizer(rendered, add_special_tokens=False)['input_ids']
    if ids[:len(prompt)] != prompt:
        raise ValueError('Assistant rendering does not preserve the inference prompt prefix')
    try:
        end = ids.index(tokenizer.eos_token_id, len(prompt)) + 1
    except ValueError as exc:
        raise ValueError('Missing assistant EOS') from exc
    if end < len(ids) and tokenizer.decode(ids[end:]).strip():
        raise ValueError('Unexpected non-whitespace after assistant EOS')
    ids = ids[:end]
    if not 0 < len(prompt) <= 1024 or not 1 < len(ids) - len(prompt) <= 512:
        raise ValueError('Prompt/target exceeds the 1024/512 token contract; never truncate')
    return {'id': problem['id'], 'messages': turns, 'input_ids': ids,
        'attention_mask': [1] * len(ids), 'labels': [-100] * len(prompt) + ids[len(prompt):],
        'prompt_tokens': len(prompt), 'target_tokens': len(ids) - len(prompt)}


def validate_splits(problems, pools):
    if Counter(p['split'] for p in problems) != COUNTS:
        raise ValueError('Expected exactly 64 SFT and 32 development tasks')
    if len({p['id'] for p in problems}) != 96 or len({p['family_id'] for p in problems}) != 96:
        raise ValueError('Duplicate ID or source family')
    if len({p['implementation_key'] for p in problems}) != 96:
        raise ValueError('Duplicate normalized implementation')
    for p in problems:
        if pools['assignments'][p['id']] != {k: p[k] for k in ('family_id', 'split')}:
            raise ValueError('Frozen family assignment changed')


def prepare():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    pools = json.loads((ROOT / 'manifests/family-pools-v1.json').read_text())
    old_audit = json.loads((ROOT / 'manifests/discovery-audit.json').read_text())
    audit = json.loads((ROOT / 'manifests/sft-pilot-audit.json').read_text())
    source = ROOT / 'data/source/data/train-00000-of-00001.parquet'
    if hashlib.sha256(source.read_bytes()).hexdigest() != pools['source_sha256']:
        raise ValueError('Pinned source hash mismatch')
    exclusions = json.loads((ROOT / 'manifests/audit-exclusions.json').read_text()) | old_audit['exclusions'] | audit['exclusions']
    reviewed = set(old_audit['reviewed_ids']) | set(audit['reviewed_ids'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    eligible, rejections = [], Counter()
    for row in pq.read_table(source).to_pylist():
        assignment = pools['assignments'][row['question_id']]
        if assignment['split'] not in COUNTS:
            continue
        try:
            if row['question_id'] in exclusions:
                raise ValueError('Audit exclusion')
            p = parse_problem(row) | assignment | {'difficulty': row['gpt_difficulty']}
            p['implementation_key'] = implementation_key(p['reference_solution'])
            encoded = training_record(p, tokenizer)
            p['prompt_tokens'], p['target_tokens'] = encoded['prompt_tokens'], encoded['target_tokens']
            eligible.append(p)
        except (ValueError, SyntaxError, KeyError, TypeError) as exc:
            rejections[str(exc)] += 1
    eligible.sort(key=lambda p: (p['id'] not in reviewed, digest([42, 'sft-pilot-v1', p['id']])))
    identity = image_identity()
    def validate(p):
        grades = [evaluate_with_runner(p, p['reference_solution'], lambda req: run_docker(req, identity['id'])) for _ in range(2)]
        return p, [g['status'] for g in grades], all(g['reward'] == 1 for g in grades)
    selected, families, implementations, checks = [], set(), set(), []
    # Development is picked first; equivalent implementations are removed from SFT.
    with ThreadPoolExecutor(max_workers=4) as executor:
        for split in ('development', 'sft_train'):
            count = 0
            remaining = iter(p for p in eligible if p['split'] == split)
            while count < COUNTS[split]:
                batch = []
                while len(batch) < min(4, COUNTS[split] - count):
                    p = next(remaining, None)
                    if p is None:
                        break
                    if p['family_id'] in families or p['implementation_key'] in implementations:
                        continue
                    families.add(p['family_id']); implementations.add(p['implementation_key'])
                    batch.append(p)
                if not batch:
                    raise RuntimeError('Insufficient eligible independent examples')
                for p, statuses, passed in executor.map(validate, batch):
                    checks.append({'id': p['id'], 'statuses': statuses})
                    if passed:
                        selected.append(p); count += 1
                        print(f'{split} {count}/{COUNTS[split]}: {p["id"]}', flush=True)
                    else:
                        print('Reference rejected', p['id'], statuses, flush=True)
    validate_splits(selected, pools)
    selected.sort(key=lambda p: (p['split'], p['id']))
    draft = {'dataset': DATASET, 'revision': DATA_REVISION, 'model': MODEL, 'model_revision': MODEL_REVISION,
        'split_manifest_sha256': digest(pools), 'audit_sha256': digest(audit),
        'sample_sha256': digest(selected), 'problems': selected, 'reference_checks': checks,
        'reference_runtime': identity, 'rejections': dict(rejections),
        'needs_review': [p['id'] for p in selected if p['id'] not in reviewed]}
    (ROOT / 'data/sft-pilot-draft.json').write_text(json.dumps(draft, indent=2) + '\n')
    print('Needs prompt/test/target review:', draft['needs_review'])


def freeze():
    from transformers import AutoTokenizer
    draft = json.loads((ROOT / 'data/sft-pilot-draft.json').read_text())
    audit = json.loads((ROOT / 'manifests/sft-pilot-audit.json').read_text())
    old = json.loads((ROOT / 'manifests/discovery-audit.json').read_text())
    pools = json.loads((ROOT / 'manifests/family-pools-v1.json').read_text())
    validate_splits(draft['problems'], pools)
    reviewed = set(old['reviewed_ids']) | set(audit['reviewed_ids'])
    if any(p['id'] not in reviewed or p['id'] in audit['exclusions'] for p in draft['problems']):
        raise ValueError('Every selected task must be audited and not excluded')
    if digest(draft['problems']) != draft['sample_sha256'] or digest(pools) != draft['split_manifest_sha256']:
        raise ValueError('Draft data or pool hash changed')
    checked = {c['id']: c['statuses'] for c in draft['reference_checks']}
    if any(checked.get(p['id']) != ['passed', 'passed'] for p in draft['problems']):
        raise ValueError('Reference validation failed')
    draft['audit_sha256'] = digest(audit)
    draft['needs_review'] = []
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    records = [training_record(p, tokenizer) for p in draft['problems'] if p['split'] == 'sft_train']
    # Training export contains no development questions or execution test cases.
    text = ''.join(json.dumps(row) + '\n' for row in records)
    dest = ROOT / 'data/sft-pilot-v1-train.jsonl'
    if dest.exists() and dest.read_text() != text:
        raise ValueError('Refusing to replace frozen training export')
    dest.write_text(text)
    draft['training_export_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    write_immutable(ROOT / 'data/sft-pilot-v1.json', draft)
    manifest = {k: v for k, v in draft.items() if k != 'problems'}
    manifest['selection'] = 'Audited IDs first, then SHA256([42, sft-pilot-v1, id]); development first; unique source families and identifier-normalized implementations; all references pass twice; no Qwen scores used for selection.'
    manifest['selected'] = [{k: p[k] for k in ('id', 'split', 'family_id', 'implementation_key', 'difficulty', 'prompt_tokens', 'target_tokens')} for p in draft['problems']]
    manifest['training_tokens'] = sum(r['target_tokens'] for r in records)
    write_immutable(ROOT / 'manifests/sft-pilot-v1.json', manifest)
    print('Frozen 64 SFT / 32 development:', draft['sample_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'freeze'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else freeze()
