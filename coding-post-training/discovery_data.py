"""Freeze family pools before sampling 100 reference-validated discovery tasks."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

from data import ROOT, DATASET, DATA_REVISION, MODEL, MODEL_REVISION, digest, tokenize_prompt, parse_problem
from evaluator import evaluate_with_runner
from local_runner import image_identity, run_docker


def family_assignments(rows, reserved_ids, manual_links=()):
    """Transitive union across ALL raw rows, including ineligible bridge rows."""
    parent = {r['question_id']: r['question_id'] for r in rows}
    owners = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for row in rows:
        rid = row['question_id']
        seeds = ast.literal_eval(row['metadata'].get('seed_ids', '[]'))
        if not isinstance(seeds, list) or not all(type(s) in (str, int) for s in seeds):
            raise ValueError('Invalid family metadata')
        try:
            solution = ast.dump(ast.parse(row['solution']), include_attributes=False)
        except SyntaxError:
            solution = ' '.join(row['solution'].split())
        keys = ['seed:' + str(s) for s in seeds] + [
            'prompt:' + digest(' '.join(row['question'].split())), 'solution:' + digest(solution)]
        for key in keys:
            if key in owners:
                a, b = find(rid), find(owners[key])
                parent[max(a, b)] = min(a, b)
            owners[key] = rid
    for left, right in manual_links:
        a, b = find(left), find(right)
        parent[max(a, b)] = min(a, b)
    reserved_families = {find(rid) for rid in reserved_ids}
    result = {}
    for rid in sorted(parent):
        family = find(rid)
        bucket = int(digest([42, 'split-v1', family])[:16], 16) % 10000
        split = ('sft_train' if bucket < 5000 else 'alignment_train' if bucket < 7500
                 else 'development' if bucket < 8500 else 'final_test')
        if family in reserved_families:
            split = 'development'
        result[rid] = {'family_id': family, 'split': split}
    return result


def write_immutable(path, value):
    text = json.dumps(value, indent=2) + '\n'
    if path.exists() and path.read_text() != text:
        raise ValueError(f'Refusing to overwrite frozen artifact: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def prepare():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    source = ROOT / 'data/source/data/train-00000-of-00001.parquet'
    rows = pq.read_table(source).to_pylist()
    smoke = json.loads((ROOT / 'manifests/development-smoke.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != smoke['source_sha256']:
        raise ValueError('Source differs from pinned baseline')
    reserved_ids = {p['id'] for p in smoke['reserved_development']}
    audit = json.loads((ROOT / 'manifests/discovery-audit.json').read_text())
    assignments = family_assignments(rows, reserved_ids, audit['family_links'])
    split_manifest = {'dataset': DATASET, 'revision': DATA_REVISION,
        'source_sha256': smoke['source_sha256'], 'seed': 42, 'manual_family_links': audit['family_links'],
        'policy': 'Transitive shared seed (numeric and string IDs normalized conservatively across subsets) plus manual links / whitespace-normalized prompt / solution AST; 50/25/10/15 percent family hash pools. Smoke families reserved for development. Pools are not reference-validated training subsets.',
        'assignments': assignments}
    write_immutable(ROOT / 'manifests/family-pools-v1.json', split_manifest)
    reserved_families = {assignments[rid]['family_id'] for rid in reserved_ids}
    exclusions = json.loads((ROOT / 'manifests/audit-exclusions.json').read_text())
    exclusions.update(audit['exclusions'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REVISION)
    eligible, rejections = [], Counter()
    for row in rows:
        try:
            if row['question_id'] in exclusions:
                raise ValueError('Manual audit exclusion')
            p = parse_problem(row)
            p.update(assignments[p['id']])
            p['difficulty'] = row['gpt_difficulty']
            tokens = tokenize_prompt(p, tokenizer)["input_ids"]
            if len(tokens) > 1024:
                raise ValueError('Prompt exceeds 1024 tokens')
            p['prompt_tokens'] = len(tokens)
            eligible.append(p)
        except (ValueError, TypeError, KeyError, SyntaxError) as exc:
            rejections[str(exc)] += 1
    candidates = [p for p in eligible if p['split'] != 'final_test' and p['family_id'] not in reserved_families]
    candidates.sort(key=lambda p: digest([42, 'discovery-v1', p['id']]))
    identity = image_identity()
    runner = lambda req: run_docker(req, identity['id'])
    selected, selected_families, validation = [], set(), []
    for stratum in ('broad', 'medium_hard'):
        count = 0
        for p in candidates:
            if p['family_id'] in selected_families or (stratum == 'medium_hard' and p['difficulty'] not in ('medium', 'hard')):
                continue
            grade = evaluate_with_runner(p, p['reference_solution'], runner)
            validation.append({'id': p['id'], 'status': grade['status']})
            if not grade['reward']:
                print('Rejected reference:', p['id'], grade['status'], flush=True)
                continue
            selected.append({**p, 'stratum': stratum})
            selected_families.add(p['family_id'])
            count += 1
            print(f'{stratum} {count}/50: {p["id"]}', flush=True)
            if count == 50:
                break
        if count != 50:
            raise RuntimeError('Insufficient eligible validated families')
    artifact = {'dataset': DATASET, 'revision': DATA_REVISION,
        'split_manifest_sha256': digest(split_manifest), 'sample_sha256': digest(selected), 'problems': selected}
    write_immutable(ROOT / 'data/discovery-v1.json', artifact)
    manifest = {k: v for k, v in artifact.items() if k != 'problems'}
    manifest.update({'selection': '50 broad then 50 medium/hard; SHA256([42, discovery-v1, id]) order; one per family; reference failures excluded; final-test and smoke families excluded',
        'eligible_pool_counts': dict(Counter(p['split'] for p in eligible)),
        'eligible_difficulty_counts': dict(Counter(p['difficulty'] for p in eligible)),
        'rejections': dict(rejections), 'reference_validation': validation,
        'reference_runtime': identity, 'audit_sha256': digest(audit),
        'selected': [{k: p[k] for k in ('id', 'family_id', 'split', 'stratum', 'difficulty', 'prompt_tokens')} for p in selected]})
    write_immutable(ROOT / 'manifests/discovery-v1.json', manifest)
    print('Prepared', artifact['sample_sha256'], dict(Counter(p['difficulty'] for p in selected)))


if __name__ == '__main__':
    prepare()
