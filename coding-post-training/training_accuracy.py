"""Measure generated-code accuracy on all 64 frozen SFT training problems, without training."""
from collections import Counter
from datetime import datetime,timezone
import json
import subprocess
import time

from data import ROOT,digest
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


def main():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or revision!=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit and push the exact runner before paid inference')
    records,_,manifest,baseline=load_data()
    source=json.loads((ROOT/'reports/sft-pilot-2026-10-09.json').read_text())
    if source['sample_sha256']!=manifest['sample_sha256'] or source['training_export_sha256']!=manifest['training_export_sha256'] or source['status']!='completed' or source['model_revision']!=baseline['model_revision']:
        raise ValueError('SFT checkpoint/data provenance mismatch')
    if file_manifest(ROOT/source['training']['local_adapter_path'])!=source['training']['adapter_files']:
        raise ValueError('Local adapter files differ from recorded checkpoint')
    problems=training_problems(json.loads((ROOT/'data/sft-pilot-v1.json').read_text()),records)
    run_id=datetime.now(timezone.utc).strftime('train-eval-%Y%m%dT%H%M%SZ-')+revision[:8]
    destination=ROOT/'runs'/f'{run_id}.json'
    report={'run_id':run_id,'revision':revision,'status':'running','source_sft_run_id':source['run_id'],
        'model':source['model'],'model_revision':source['model_revision'],
        'sample_sha256':manifest['sample_sha256'],'training_problems_sha256':digest(problems),
        'training_export_sha256':manifest['training_export_sha256'],
        'adapter_files':source['training']['adapter_files'],'evaluator_version':EVALUATOR_VERSION,
        'decoding':source['decoding'],'reference_runtime':baseline['reference_runtime'],
        'budget_usd':2.0,'setup_reserve_usd':0.25,'resource_usd_per_second':RESOURCE_USD_PER_SECOND,
        'charged_wall_seconds':0,'raw_outputs':[],'results':[],'gpu_batches':[]}
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
        from modal_app import app,sft_training_generate
        with modal.enable_output(),app.run():
            report['app_id']=app.app_id;save()
            batches=[problems[:1]]+[problems[i:i+9] for i in range(1,64,9)]
            for i,batch in enumerate(batches):
                if not can_start_batch(report['charged_wall_seconds'],budget_usd=2.0):
                    raise RuntimeError('Budget cannot reserve the next complete call')
                start=time.monotonic()
                try:
                    generated=sft_training_generate.remote([{k:p[k] for k in ('id','question','signature')} for p in batch],source['run_id'],run_id,i,source['training']['adapter_files'])
                finally:
                    report['charged_wall_seconds']+=time.monotonic()-start+2
                    report['estimated_usd_with_setup_reserve']=0.25+report['charged_wall_seconds']*RESOURCE_USD_PER_SECOND
                    save()
                report['raw_outputs'].extend(generated['outputs'])
                report['gpu_batches'].append({k:v for k,v in generated.items() if k!='outputs'});save()
                if [o['id'] for o in generated['outputs']]!=[p['id'] for p in batch] or any(o['sample_index']!=0 for o in generated['outputs']):
                    raise ValueError('Unexpected inference IDs/sample count')
                if i==0:grade(generated['outputs'])
                print(f'Saved {len(report["raw_outputs"])}/64 training-set answers.',flush=True)
        grade(report['raw_outputs'][1:])
        if len(report['results'])!=64:raise ValueError('Incomplete evaluation')
        report['summary']={'problems':64,'passed':sum(r['reward'] for r in report['results']),
            'accuracy':sum(r['reward'] for r in report['results'])/64,
            'status_counts':dict(Counter(r['status'] for r in report['results'])),
            'raw_format_passed':sum(r['raw_format_compliant'] for r in report['results']),
            'generated_tokens':sum(r['generated_tokens'] for r in report['results']),
            'passed_cases':sum(r['passed_cases'] for r in report['results']),
            'total_cases':sum(r['total_cases'] for r in report['results'])}
        report['status']='completed';save();print(json.dumps(report['summary']),flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(destination,flush=True)


if __name__=='__main__':main()
