"""Run the frozen 15-example memorization diagnostic with a fresh adapter."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import subprocess
import time
from data import ROOT,digest,MODEL,MODEL_REVISION
from discovery import RESOURCE_USD_PER_SECOND
from evaluator import evaluate_with_runner,EVALUATOR_VERSION
from local_runner import image_identity,run_docker
from overfit_data import select
from sft_core import OVERFIT_SETTINGS
from sft_run import load_data,download_adapter


def load_overfit():
    records,_,pilot,baseline=load_data()
    a=json.loads((ROOT/'data/overfit-15-v1.json').read_text())
    m=json.loads((ROOT/'manifests/overfit-15-v1.json').read_text())
    source=json.loads((ROOT/'reports/base-training-accuracy-2026-10-09.json').read_text())
    if source['model_revision']!=MODEL_REVISION or source['decoding']!=m['decoding'] or source['run_id']!=m['baseline_run_id']:
        raise ValueError('Original-model baseline provenance mismatch')
    all_problems=json.loads((ROOT/'data/sft-pilot-v1.json').read_text())['problems']
    expected=select(all_problems,source['results'])
    byid={r['id']:r for r in records}
    if a['selected']!=expected or a['records']!=[byid[p['id']] for p in expected]:raise ValueError('Subset changed')
    if digest(a['selected'])!=m['sample_sha256'] or digest(a['records'])!=m['records_sha256'] or digest(pilot)!=m['source_manifest_sha256']:raise ValueError('Subset provenance mismatch')
    if image_identity()!=m['reference_runtime'] or EVALUATOR_VERSION!=m['evaluator_version'] or baseline['decoding']!=m['decoding']:raise ValueError('Evaluation contract changed')
    return a,m,source


def main():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or revision!=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit and push before paid execution')
    a,m,baseline=load_overfit()
    runner=lambda req:run_docker(req,m['reference_runtime']['id'])
    if evaluate_with_runner({'entry_point':'f','cases':[{'args':[7],'expected':7}]},'def f(x): return x',runner)['reward']!=1:raise RuntimeError('Evaluator canary failed')
    run_id=datetime.now(timezone.utc).strftime('sft-%Y%m%dT%H%M%SZ-')+revision[:8]
    dest=ROOT/'runs'/f'{run_id}.json'
    report={'run_id':run_id,'revision':revision,'experiment':'overfit-15-v1','status':'running',
        'model':MODEL,'model_revision':MODEL_REVISION,'settings':OVERFIT_SETTINGS,
        'manifest_sha256':digest(m),'sample_sha256':m['sample_sha256'],'records_sha256':m['records_sha256'],
        'baseline_run_id':baseline['run_id'],'decoding':m['decoding'],'evaluator_version':EVALUATOR_VERSION,
        'reference_runtime':m['reference_runtime'],'budget_usd':4.0,'setup_reserve_usd':0.25,
        'charged_wall_seconds':0,'results':[]}
    def save():
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(dest)
    def call(fn,args,timeout):
        if .25+(report['charged_wall_seconds']+300+timeout+2)*RESOURCE_USD_PER_SECOND>4:raise RuntimeError('Cannot reserve next full call within budget')
        started=time.monotonic()
        try:return fn.remote(*args)
        finally:
            report['charged_wall_seconds']+=time.monotonic()-started+2
            report['estimated_usd_with_setup_reserve']=.25+report['charged_wall_seconds']*RESOURCE_USD_PER_SECOND
            save()
    try:
        import modal
        from modal_app import app,results_volume,sft_canary,sft_overfit
        with modal.enable_output(),app.run():
            report['app_id']=app.app_id;save()
            report['canary']=call(sft_canary,(a['records'],run_id),300)
            report['canary']['local_adapter_path']=download_adapter(results_volume,run_id,'canary',report['canary']);save()
            print('Canary passed; starting fresh 15-example training.',flush=True)
            prompts=[{k:p[k] for k in ('id','question','signature')} for p in a['selected']]
            report['training']=call(sft_overfit,(a['records'],prompts,run_id),1200);save()
            if report['training']['optimizer_steps']!=60:raise ValueError('Expected 60 optimizer steps')
            report['training']['local_adapter_path']=download_adapter(results_volume,run_id,'overfit',report['training']);save()
        outputs=report['training']['evaluation']['outputs']
        if [o['id'] for o in outputs]!=[p['id'] for p in a['selected']]:raise ValueError('Evaluation IDs mismatch')
        byid={p['id']:p for p in a['selected']}
        def grade(o):
            p=byid[o['id']]
            g=evaluate_with_runner(p,o['code'],runner) if o['terminated'] else {'status':'truncated','reward':0,'passed_cases':0,'total_cases':len(p['cases']),'raw_format_compliant':False}
            return o|g
        with ThreadPoolExecutor(max_workers=4) as executor:report['results']=list(executor.map(grade,outputs))
        before={r['id']:r for r in baseline['results']}
        report['summary']={'problems':15,'before_passed':8,'after_passed':sum(r['reward'] for r in report['results']),
            'status_counts':dict(Counter(r['status'] for r in report['results'])),
            'improved_ids':[r['id'] for r in report['results'] if r['reward']>before[r['id']]['reward']],
            'regressed_ids':[r['id'] for r in report['results'] if r['reward']<before[r['id']]['reward']],
            'raw_format_passed':sum(r['raw_format_compliant'] for r in report['results'])}
        report['status']='completed';save();print(json.dumps(report['summary']),flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(dest,flush=True)

if __name__=='__main__':main()
