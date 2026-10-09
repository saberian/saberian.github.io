"""Bounded 100 x 4 sampling, with local isolated grading and durable raw answers."""
from collections import Counter
from datetime import datetime, timezone
import json
import subprocess
import time

from data import ROOT, DATA_REVISION, MODEL, MODEL_REVISION, digest
from evaluator import EVALUATOR_VERSION, evaluate_with_runner
from local_runner import image_identity, run_docker

BUDGET_USD = 3.0
SETUP_RESERVE_USD = 0.25
RESOURCE_USD_PER_SECOND = 0.000694 + 2 * 0.0000131 + 16 * 0.00000222
# Includes the next call's maximum startup, execution, and idle tail.
CALL_RESERVE_SECONDS = 300 + 240 + 2


def can_start_batch(charged_seconds, budget_usd=BUDGET_USD):
    return SETUP_RESERVE_USD + (charged_seconds + CALL_RESERVE_SECONDS) * RESOURCE_USD_PER_SECOND <= budget_usd


def validate_artifact(artifact, manifest, pools):
    problems = artifact['problems']
    if len(problems) != 100 or artifact['revision'] != DATA_REVISION:
        raise ValueError('Expected 100 problems from the pinned dataset')
    if digest(problems) != artifact['sample_sha256'] or artifact['sample_sha256'] != manifest['sample_sha256']:
        raise ValueError('Data differs from frozen manifest')
    if digest(pools) != artifact['split_manifest_sha256']:
        raise ValueError('Family pool manifest changed')
    if len({p['id'] for p in problems}) != 100 or len({p['family_id'] for p in problems}) != 100:
        raise ValueError('Duplicate problem/family')
    if Counter(p['stratum'] for p in problems) != {'broad': 50, 'medium_hard': 50}:
        raise ValueError('Unexpected sample strata')
    for p in problems:
        if p['split'] not in ('sft_train', 'alignment_train', 'development') or pools['assignments'][p['id']] != {k: p[k] for k in ('family_id', 'split')}:
            raise ValueError('Held-out problem or invalid family assignment')


def validate_outputs(batch, outputs):
    expected = {(p['id'], i) for p in batch for i in range(4)}
    actual = [(o['id'], o['sample_index']) for o in outputs]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Missing/duplicate/unexpected generated response')


def summarize(results, problems):
    rows = []
    for p in problems:
        samples = [r for r in results if r['id'] == p['id']]
        if len(samples) != 4:
            continue
        rows.append({k: p[k] for k in ('id', 'split', 'stratum', 'difficulty')} | {
            'passed': sum(r['reward'] for r in samples),
            'statuses': dict(Counter(r['status'] for r in samples)),
            'unique_responses': len({r['code'] for r in samples})})
    strata = {}
    for stratum in ('broad', 'medium_hard'):
        subset = [r for r in rows if r['stratum'] == stratum]
        strata[stratum] = {'problems': len(subset),
            'pass_histogram': {str(k): sum(r['passed'] == k for r in subset) for k in range(5)}}
    return {'problems': rows, 'strata': strata,
        'complete_problems': len(rows), 'responses': len(results),
        'sample_accuracy': sum(r['reward'] for r in results) / len(results) if results else None,
        'raw_format_compliance': sum(r.get('raw_format_compliant', False) for r in results) / len(results) if results else None,
        'pass_histogram': {str(k): sum(r['passed'] == k for r in rows) for k in range(5)},
        'generated_tokens': sum(r['generated_tokens'] for r in results)}


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit and push before a paid run')
    if revision != subprocess.check_output(['git', 'rev-parse', '@{upstream}'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Runner must match pushed upstream')
    artifact = json.loads((ROOT / 'data/discovery-v1.json').read_text())
    manifest = json.loads((ROOT / 'manifests/discovery-v1.json').read_text())
    pools = json.loads((ROOT / 'manifests/family-pools-v1.json').read_text())
    validate_artifact(artifact, manifest, pools)
    problems = artifact['problems']
    audit = json.loads((ROOT / 'manifests/discovery-audit.json').read_text())
    if digest(audit) != manifest['audit_sha256'] or any(p['id'] not in audit['reviewed_ids'] or p['id'] in audit['exclusions'] for p in problems):
        raise RuntimeError('Every selected prompt must match the recorded audit')
    identity = image_identity()
    if identity != manifest['reference_runtime']:
        raise RuntimeError('Docker reference-validation runtime changed')
    runner = lambda req: run_docker(req, identity['id'])
    run_id = datetime.now(timezone.utc).strftime('discovery-%Y%m%dT%H%M%SZ-') + revision[:8]
    destination = ROOT / 'runs' / (run_id + '.json')
    destination.parent.mkdir(exist_ok=True)
    report = {'run_id': run_id, 'revision': revision, 'status': 'running',
        'model': MODEL, 'model_revision': MODEL_REVISION, 'dataset_revision': DATA_REVISION,
        'sample_sha256': artifact['sample_sha256'], 'split_manifest_sha256': artifact['split_manifest_sha256'],
        'evaluator_version': EVALUATOR_VERSION, 'runtime': identity,
        'sampling': {'samples': 4, 'temperature': 1.0, 'top_p': 1.0, 'top_k': 0, 'max_new_tokens': 512,
            'seed': 'first 32 bits of SHA256(JSON([42, problem_id]))', 'batching': 'four independent sequences per prompt'},
        'budget': {'allowance_usd': BUDGET_USD, 'setup_reserve_usd': SETUP_RESERVE_USD,
            'resource_usd_per_second': RESOURCE_USD_PER_SECOND, 'pricing_url': 'https://modal.com/pricing',
            'note': 'Conservative application estimate, not a provider billing cap; remote wall time plus 2s idle per call; billed total unknown'},
        'gpu_batches': [], 'raw_outputs': [], 'results': []}
    def save():
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2) + '\n')
        temporary.replace(destination)
    def grade(outputs):
        by_id = {p['id']: p for p in problems}
        for output in outputs:
            p = by_id[output['id']]
            if not output['terminated']:
                result = {'status': 'truncated', 'reward': 0, 'passed_cases': 0, 'total_cases': len(p['cases']), 'raw_format_compliant': False}
            else:
                result = evaluate_with_runner(p, output['code'], runner)
            report['results'].append({**output, **result})
            save()
    charged_seconds = 0.0
    try:
        # Local execution canary before any cloud work.
        probe = {'entry_point': 'f', 'cases': [{'args': [7], 'expected': 7}]}
        if evaluate_with_runner(probe, 'def f(x): return x', runner)['reward'] != 1:
            raise RuntimeError('Local evaluation canary failed')
        import modal
        from modal_app import app, discover_generate
        with modal.enable_output(), app.run():
            report['app_id'] = app.app_id
            save()
            batches = [problems[:1]] + [problems[i:i + 9] for i in range(1, 100, 9)]
            for index, batch in enumerate(batches):
                if not can_start_batch(charged_seconds):
                    report['status'] = 'budget_stopped'
                    break
                prompts = [{k: p[k] for k in ('id', 'question', 'signature')} for p in batch]
                started = time.monotonic()
                generated = discover_generate.remote(prompts, run_id, index)
                charged_seconds += time.monotonic() - started + 2
                report['budget']['remote_wall_seconds_plus_idle'] = charged_seconds
                report['budget']['estimated_usd_with_setup_reserve'] = SETUP_RESERVE_USD + charged_seconds * RESOURCE_USD_PER_SECOND
                report['gpu_batches'].append({k: v for k, v in generated.items() if k != 'outputs'})
                report['raw_outputs'].extend(generated['outputs'])
                save()  # Keep raw responses even if later validation/grading fails.
                validate_outputs(batch, generated['outputs'])
                if index == 0:
                    grade(generated['outputs'])
                    # Incorrect answers do not fail the infrastructure canary.
                    print('Canary generated, retrieved, graded, saved, and containers cleaned.', flush=True)
                print(f'Saved {len(report["raw_outputs"])}/400 responses; estimate including reserve ${report["budget"]["estimated_usd_with_setup_reserve"]:.2f}', flush=True)
        # End the GPU app before CPU-only evaluation.
        grade(report['raw_outputs'][4:])
        report['summary'] = summarize(report['results'], problems)
        if report['status'] == 'running':
            report['status'] = 'completed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(json.dumps(report['summary'], indent=2), flush=True)
    print(f'Report: {destination}', flush=True)


if __name__ == '__main__':
    main()
