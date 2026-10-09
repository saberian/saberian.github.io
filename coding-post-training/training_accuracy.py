"""Measure generated-code accuracy on all 64 frozen SFT training problems, without training."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import subprocess
import time

from data import ROOT,digest,evaluation_batches
from discovery import can_start_batch,RESOURCE_USD_PER_SECOND
from evaluator import EVALUATOR_VERSION,evaluate_with_runner
from local_runner import run_docker
from sft_core import file_manifest
from sft_run import load_data


def training_problems(artifact,records):
    problems=[p for p in artifact['problems'] if p['split']=='sft_train']
    if len(problems)!=64 or len({p['id'] for p in problems})!=64 or {p['id'] for p in problems}!={r['id'] for r in records}:
        raise ValueError('Expected exactly the 64 frozen SFT training problems')
    return problems


def resume_report(previous, expected, problems):
    for key in ('policy','split','model_revision','source_sft_run_id','evaluation_problems_sha256',
                'training_export_sha256','adapter_files','evaluator_version','decoding','reference_runtime','budget_usd'):
        if previous[key]!=expected[key]:raise ValueError(f'Resume provenance mismatch: {key}')
    if previous['status']!='failed':raise ValueError('Only a failed run may resume')
    batches=evaluation_batches(problems)
    completed=len(previous['gpu_batches'])
    ids=[p['id'] for batch in batches[:completed] for p in batch]
    if [o['id'] for o in previous['raw_outputs']]!=ids or any(o['sample_index']!=0 for o in previous['raw_outputs']):
        raise ValueError('Resume requires complete, ordered saved batches')
    if [r['id'] for r in previous['results']]!=ids[:len(previous['results'])]:
        raise ValueError('Invalid graded prefix')
    previous.setdefault('attempts',[]).append({k:previous[k] for k in ('app_id','revision','error')})
    previous['revision']=expected['revision'];previous['status']='running';previous.pop('error')
    return previous


def recover_saved_batches(report, problems, read_batch):
    """Recover committed remote results when response delivery failed after generation."""
    batches=evaluation_batches(problems)
    for index in range(len(report['gpu_batches']),len(batches)):
        saved=read_batch(index)
        if saved is None:
            break
        outputs=saved['outputs']
        if [o['id'] for o in outputs]!=[p['id'] for p in batches[index]] or any(o['sample_index']!=0 for o in outputs):
            raise ValueError('Remote checkpoint is incomplete or mismatched; do not regenerate blindly')
        report['raw_outputs'].extend(outputs)
        report['gpu_batches'].append({'recovered_from_volume':True,'batch_index':index})


def read_saved_batch(volume, run_id, index):
    try:
        return json.loads(b''.join(volume.read_file(f'{run_id}/batch-{index}.json')))
    except FileNotFoundError:
        # Volume.read_file maps a missing remote file to the Python filesystem exception.
        return None


def main(policy="sft", split="train", resume=None, checkpoint="pilot"):
    if policy not in ("base","sft"):
        raise ValueError("Unknown policy")
    if checkpoint not in ('pilot','overfit') or (checkpoint=='overfit' and (policy!='sft' or split!='validation')):
        raise ValueError('The overfit checkpoint is evaluated only on validation')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or revision!=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit and push the exact runner before paid inference')
    records,development,manifest,baseline=load_data()
    source=json.loads((ROOT/'reports/sft-pilot-2026-10-09.json').read_text())
    if source['sample_sha256']!=manifest['sample_sha256'] or source['training_export_sha256']!=manifest['training_export_sha256'] or source['status']!='completed' or source['model_revision']!=baseline['model_revision']:
        raise ValueError('SFT checkpoint/data provenance mismatch')
    if checkpoint=='overfit':
        source=json.loads((ROOT/'reports/sft-overfit-15-2026-10-09.json').read_text())
        subset=json.loads((ROOT/'manifests/overfit-15-v1.json').read_text())
        if source['status']!='completed' or source['manifest_sha256']!=digest(subset) or source['model_revision']!=baseline['model_revision'] or source['decoding']!=baseline['decoding'] or source['evaluator_version']!=EVALUATOR_VERSION:
            raise ValueError('Overfit checkpoint provenance mismatch')
    if policy == 'sft' and file_manifest(ROOT/source['training']['local_adapter_path'])!=source['training']['adapter_files']:
        raise ValueError('Local adapter files differ from recorded checkpoint')
    problems=training_problems(json.loads((ROOT/'data/sft-pilot-v1.json').read_text()),records)
    if split == 'test':
        from test_data import validate_test
        artifact=json.loads((ROOT/'data/final-test-v1.json').read_text())
        test_manifest=json.loads((ROOT/'manifests/final-test-v1.json').read_text())
        pools=json.loads((ROOT/'manifests/family-pools-v1.json').read_text())
        pilot=json.loads((ROOT/'data/sft-pilot-v1.json').read_text())['problems']
        problems=artifact['problems']
        validate_test(problems,pools,pilot)
        if digest(problems)!=test_manifest['sample_sha256'] or digest(pools)!=test_manifest['split_manifest_sha256'] or artifact['reference_runtime']!=baseline['reference_runtime']:
            raise ValueError('Frozen test provenance mismatch')
    elif split=='validation':
        problems=development
        if len(problems)!=32 or digest(problems)!=baseline['development_sha256']:
            raise ValueError('Expected the frozen 32 validation problems')
    elif split != 'train':raise ValueError('Unknown split')
    n=len(problems)
    budget=8.0 if split=='test' else 2.0
    run_id=datetime.now(timezone.utc).strftime(f'{split}-eval-%Y%m%dT%H%M%SZ-')+revision[:8]
    destination=ROOT/'runs'/f'{run_id}.json'
    report={'run_id':run_id,'revision':revision,'status':'running','policy':policy,'split':split,'source_sft_run_id':source['run_id'] if policy=='sft' else None,
        'model':source['model'],'model_revision':source['model_revision'],
        'sample_sha256':manifest['sample_sha256'],'evaluation_problems_sha256':digest(problems),
        'training_export_sha256':source.get('training_export_sha256'),
        'checkpoint':checkpoint,'checkpoint_records_sha256':source.get('records_sha256'),
        'adapter_files':source['training']['adapter_files'] if policy=='sft' else None,'evaluator_version':EVALUATOR_VERSION,
        'decoding':source['decoding'],'reference_runtime':baseline['reference_runtime'],
        'budget_usd':budget,'setup_reserve_usd':0.25,'resource_usd_per_second':RESOURCE_USD_PER_SECOND,
        'charged_wall_seconds':0,'raw_outputs':[],'results':[],'gpu_batches':[]}
    if resume:
        import re
        if not re.fullmatch(fr'{split}-eval-[0-9TZ]+-[0-9a-f]{{8}}',resume):raise ValueError('Invalid resume run ID')
        destination=ROOT/'runs'/f'{resume}.json'
        report=resume_report(json.loads(destination.read_text()),report,problems)
        run_id=resume
    runner=lambda req:run_docker(req,baseline['reference_runtime']['id'])
    def save():
        tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(destination)
    by_id={p['id']:p for p in problems}
    def grade(outputs):
        for out in outputs:
            p=by_id[out['id']]
            result=evaluate_with_runner(p,out['code'],runner) if out['terminated'] else {
                'status':'truncated','reward':0,'passed_cases':0,'total_cases':len(p['cases']),'raw_format_compliant':False}
            report['results'].append({**out,**result});save()
    try:
        probe={'entry_point':'f','cases':[{'args':[7],'expected':7}]}
        if evaluate_with_runner(probe,'def f(x): return x',runner)['reward']!=1:
            raise RuntimeError('Local evaluator canary failed')
        import modal
        from modal_app import app,sft_training_generate,base_training_generate,results_volume
        if resume:
            recover_saved_batches(report,problems,lambda index:read_saved_batch(results_volume,run_id,index))
            save()
        with modal.enable_output(),app.run():
            report['app_id']=app.app_id;save()
            batches=evaluation_batches(problems)
            completed=len(report['gpu_batches'])
            for i,batch in enumerate(batches):
                if i<completed:continue
                if not can_start_batch(report['charged_wall_seconds'],budget_usd=budget):
                    raise RuntimeError('Budget cannot reserve the next complete call')
                start=time.monotonic()
                try:
                    prompts=[{k:p[k] for k in ('id','question','signature')} for p in batch]
                    generated=(base_training_generate.remote(prompts,run_id,i) if policy=='base' else
                        sft_training_generate.remote(prompts,source['run_id'],run_id,i,source['training']['adapter_files'],
                            'overfit' if checkpoint=='overfit' else 'train'))
                finally:
                    report['charged_wall_seconds']+=time.monotonic()-start+2
                    report['estimated_usd_with_setup_reserve']=0.25+report['charged_wall_seconds']*RESOURCE_USD_PER_SECOND
                    save()
                report['raw_outputs'].extend(generated['outputs'])
                report['gpu_batches'].append({k:v for k,v in generated.items() if k!='outputs'});save()
                if [o['id'] for o in generated['outputs']]!=[p['id'] for p in batch] or any(o['sample_index']!=0 for o in generated['outputs']):
                    raise ValueError('Unexpected inference IDs/sample count')
                if i==0:grade(generated['outputs'])
                print(f'Saved {len(report["raw_outputs"])}/{n} {split}-set answers.',flush=True)
        grade(report['raw_outputs'][len(report['results']):])
        if len(report['results'])!=n:raise ValueError('Incomplete evaluation')
        report['summary']={'problems':n,'passed':sum(r['reward'] for r in report['results']),
            'accuracy':sum(r['reward'] for r in report['results'])/n,
            'status_counts':dict(Counter(r['status'] for r in report['results'])),
            'raw_format_passed':sum(r['raw_format_compliant'] for r in report['results']),
            'generated_tokens':sum(r['generated_tokens'] for r in report['results']),
            'passed_cases':sum(r['passed_cases'] for r in report['results']),
            'total_cases':sum(r['total_cases'] for r in report['results'])}
        report['status']='completed';save();print(json.dumps(report['summary']),flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(destination,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=['base','sft'],default='sft')
    parser.add_argument('--split',choices=['train','test','validation'],default='train')
    parser.add_argument('--checkpoint',choices=['pilot','overfit'],default='pilot')
    parser.add_argument('--resume')
    args=parser.parse_args()
    main(args.policy,args.split,args.resume,args.checkpoint)
