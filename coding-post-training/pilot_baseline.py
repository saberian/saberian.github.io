"""Greedy pre-SFT evaluation on the frozen 32-problem development set."""
from collections import Counter
from datetime import datetime, timezone
import json
import subprocess
import time

from data import ROOT, MODEL, MODEL_REVISION, digest
from discovery import can_start_batch, RESOURCE_USD_PER_SECOND, SETUP_RESERVE_USD
from evaluator import EVALUATOR_VERSION, evaluate_with_runner
from local_runner import image_identity, run_docker
from pilot_data import validate_splits


def main():
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip() or revision != subprocess.check_output(['git', 'rev-parse', '@{upstream}'], cwd=ROOT, text=True).strip():
        raise RuntimeError('Commit and push the exact runner before paid inference')
    artifact = json.loads((ROOT / 'data/sft-pilot-v1.json').read_text())
    manifest = json.loads((ROOT / 'manifests/sft-pilot-v1.json').read_text())
    pools = json.loads((ROOT / 'manifests/family-pools-v1.json').read_text())
    audit = json.loads((ROOT / 'manifests/sft-pilot-audit.json').read_text())
    validate_splits(artifact['problems'], pools)
    if digest(artifact['problems']) != manifest['sample_sha256'] or artifact['sample_sha256'] != manifest['sample_sha256'] or digest(pools) != manifest['split_manifest_sha256'] or digest(audit) != manifest['audit_sha256']:
        raise ValueError('Frozen data, audit, or family hashes do not match')
    problems = [p for p in artifact['problems'] if p['split'] == 'development']
    identity = image_identity()
    if identity != manifest['reference_runtime']:
        raise ValueError('Reference-validation runtime changed')
    runner = lambda req: run_docker(req, identity['id'])
    run_id = datetime.now(timezone.utc).strftime('sft-pilot-baseline-%Y%m%dT%H%M%SZ-') + revision[:8]
    destination = ROOT / 'runs' / (run_id + '.json')
    destination.parent.mkdir(parents=True, exist_ok=True)
    report = {'run_id': run_id, 'revision': revision, 'model': MODEL, 'model_revision': MODEL_REVISION,
        'status': 'running', 'sample_sha256': artifact['sample_sha256'],
        'development_sha256': digest(problems), 'evaluator_version': EVALUATOR_VERSION,
        'reference_runtime': identity, 'decoding': {'do_sample': False, 'max_new_tokens': 512},
        'budget_usd': 1.5, 'resource_usd_per_second': RESOURCE_USD_PER_SECOND,
        'setup_reserve_usd': SETUP_RESERVE_USD, 'raw_outputs': [], 'results': [], 'gpu_batches': []}
    def save():
        tmp = destination.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n'); tmp.replace(destination)
    def grade(outputs):
        by_id = {p['id']: p for p in problems}
        for out in outputs:
            p = by_id[out['id']]
            result = (evaluate_with_runner(p, out['code'], runner) if out['terminated'] else
                {'status': 'truncated', 'reward': 0, 'passed_cases': 0, 'total_cases': len(p['cases']), 'raw_format_compliant': False})
            report['results'].append({**out, **result}); save()
    charged_seconds = 0
    try:
        probe = {'entry_point': 'f', 'cases': [{'args': [7], 'expected': 7}]}
        if evaluate_with_runner(probe, 'def f(x): return x', runner)['reward'] != 1:
            raise RuntimeError('Local evaluator canary failed')
        import modal
        from modal_app import app, pilot_generate
        with modal.enable_output(), app.run():
            report['app_id'] = app.app_id; save()
            batches = [problems[:1]] + [problems[i:i+9] for i in range(1, 32, 9)]
            for index, batch in enumerate(batches):
                if not can_start_batch(charged_seconds, budget_usd=report['budget_usd']):
                    report['status'] = 'budget_stopped'; break
                started = time.monotonic()
                generated = pilot_generate.remote([{k: p[k] for k in ('id', 'question', 'signature')} for p in batch], run_id, index)
                charged_seconds += time.monotonic() - started + 2
                report['estimated_usd_with_setup_reserve'] = SETUP_RESERVE_USD + charged_seconds * RESOURCE_USD_PER_SECOND
                report['remote_wall_seconds_plus_idle'] = charged_seconds
                report['gpu_batches'].append({k: v for k, v in generated.items() if k != 'outputs'})
                report['raw_outputs'].extend(generated['outputs']); save()
                if [o['id'] for o in generated['outputs']] != [p['id'] for p in batch] or any(o['sample_index'] != 0 for o in generated['outputs']):
                    raise RuntimeError('Incomplete, reordered, or duplicate inference results')
                if index == 0:
                    grade(generated['outputs'])
                    print('Canary generated, saved, graded, and cleaned.', flush=True)
                print(f'Saved {len(report["raw_outputs"])}/32 greedy answers.', flush=True)
        grade(report['raw_outputs'][1:])
        n = len(report['results'])
        report['summary'] = {'problems': n, 'passed': sum(r['reward'] for r in report['results']),
            'greedy_accuracy': sum(r['reward'] for r in report['results']) / n if n else None,
            'status_counts': dict(Counter(r['status'] for r in report['results'])),
            'passed_cases': sum(r['passed_cases'] for r in report['results']),
            'total_cases': sum(r['total_cases'] for r in report['results']),
            'raw_format_compliance': sum(r['raw_format_compliant'] for r in report['results']) / n if n else None,
            'generated_tokens': sum(r['generated_tokens'] for r in report['results'])}
        if report['status'] == 'running':
            report['status'] = 'completed'
        print(json.dumps(report['summary']), flush=True)
    except Exception as exc:
        report['status'] = 'failed'; report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        save()
    print(destination, flush=True)


if __name__ == '__main__':
    main()
