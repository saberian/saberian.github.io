"""Budgeted online GRPO: local isolated rewards, original-Qwen reference, validation only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import re
import subprocess
import time
from data import ROOT,MODEL,MODEL_REVISION,digest,evaluation_batches
from discovery import RESOURCE_USD_PER_SECOND
from evaluator import EVALUATOR_VERSION,evaluate_with_runner
from local_runner import run_docker,image_identity
from grpo_core import SETTINGS
from grpo_data import load_training
from sft_run import load_data,download_adapter


def prepare_data():
    train,source=load_training();m=json.loads((ROOT/'manifests/grpo-pilot-v1.json').read_text())
    _,dev,pilot,baseline=load_data()
    if digest(train)!=m['sample_sha256'] or digest(source)!=m['discovery_manifest_sha256'] or m['model_revision']!=MODEL_REVISION or m['evaluator_version']!=EVALUATOR_VERSION or m['reference_runtime']!=image_identity():raise ValueError('GRPO provenance mismatch')
    if {p['family_id'] for p in train}&{p['family_id'] for p in dev}:raise ValueError('Training/validation overlap')
    return train,dev,m,baseline


def main(resume=None):
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or revision!=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip():raise RuntimeError('Commit and push exact runner before paid execution')
    train,dev,manifest,baseline=prepare_data()
    run_id=resume or datetime.now(timezone.utc).strftime('grpo-%Y%m%dT%H%M%SZ-')+revision[:8]
    if not re.fullmatch(r'grpo-[0-9TZ]+-[0-9a-f]{8}',run_id):raise ValueError('Invalid run ID')
    path=ROOT/'runs'/f'{run_id}.json'
    report={'run_id':run_id,'revision':revision,'status':'running','model':MODEL,'model_revision':MODEL_REVISION,
        'settings':SETTINGS,'manifest_sha256':digest(manifest),'training_sha256':digest(train),'development_sha256':digest(dev),
        'evaluator_version':EVALUATOR_VERSION,'reference_runtime':manifest['reference_runtime'],
        'decoding':baseline['decoding'],'baseline_validation_run_id':baseline['run_id'],
        'budget_usd':4.0,'setup_reserve_usd':.25,'charged_wall_seconds':0,'rounds':[],'evaluations':{},'app_ids':[]}
    if resume:
        old=json.loads(path.read_text())
        for key in ('model_revision','settings','manifest_sha256','development_sha256','evaluator_version','reference_runtime'):
            if old[key]!=report[key]:raise ValueError('Resume provenance changed: '+key)
        if old['status']=='completed':raise ValueError('Run already completed')
        old.setdefault('resume_revisions',[]).append(revision);report=old;report['status']='running'
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
    def call(fn,*args):
        import modal
        for attempt in range(2):
            # Reserve a complete startup + call + idle period, including checkpoint recovery.
            if .25+(report['charged_wall_seconds']+960)*RESOURCE_USD_PER_SECOND>4:raise RuntimeError('Cannot reserve next full call within $4 allowance')
            start=time.monotonic()
            try:return fn.remote(*args)
            except (modal.exception.ConnectionError,ConnectionError,OSError) as exc:
                report.setdefault('connection_events',[]).append({'function':str(fn),'error':str(exc),'retry':attempt});save()
                if attempt:raise
            finally:
                report['charged_wall_seconds']+=time.monotonic()-start+60
                report['estimated_usd_with_setup_reserve']=.25+report['charged_wall_seconds']*RESOURCE_USD_PER_SECOND
                save()
    all_problems={p['id']:p for p in train+dev}
    runner=lambda req:run_docker(req,manifest['reference_runtime']['id'])
    def grade_one(o):
        p=all_problems[o['id']]
        result=evaluate_with_runner(p,o['code'],runner) if o['terminated'] else {'status':'truncated','reward':0,'passed_cases':0,'total_cases':len(p['cases']),'raw_format_compliant':False}
        return o|result
    def grade(outputs):
        with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(grade_one,outputs))
    def prompts(problems):return [{k:p[k] for k in ('id','question','signature')} for p in problems]
    try:
        if evaluate_with_runner({'entry_point':'f','cases':[{'args':[7],'expected':7}]},'def f(x): return x',runner)['reward']!=1:raise RuntimeError('Local evaluator canary failed')
        import modal
        from modal_app import app,GRPOWorker,results_volume
        with modal.enable_output(),app.run():
            report['app_ids'].append(app.app_id);save();worker=GRPOWorker(run_id=run_id)
            def evaluate(label,ps,samples):
                if label not in report['evaluations']:
                    generated=call(worker.evaluate,prompts(ps),samples,label)
                    if [(o['id'],o['sample_index']) for o in generated['outputs']]!=[(p['id'],i) for p in ps for i in range(samples)]:raise ValueError('Unexpected evaluation IDs')
                    report['evaluations'][label]=generated;save()
                if 'results' not in report['evaluations'][label]:
                    report['evaluations'][label]['results']=grade(report['evaluations'][label]['outputs']);save()
            evaluate('train-before',train,1)
            for index in range(SETTINGS['rounds']):
                if len(report['rounds'])<=index:
                    result=call(worker.rollout,index,prompts(train))
                    if [(o['id'],o['sample_index']) for o in result['outputs']]!=[(p['id'],i) for p in train for i in range(4)] or digest(result['outputs'])!=result['digest']:raise ValueError('Rollout order/hash mismatch')
                    report['rounds'].append({'rollout':result});save()
                entry=report['rounds'][index]
                if 'results' not in entry:entry['results']=grade(entry['rollout']['outputs']);save()
                if 'update' not in entry:
                    entry['update']=call(worker.update,index,entry['rollout']['digest'],[r['reward'] for r in entry['results']]);save()
                if index==0 and 'canary_reload' not in report:
                    report['canary_reload']=call(worker.verify_reload);save()
                    print('Canary passed real rewards, two updates, persistence and exact-logit reload.',flush=True)
                print(f'Round {index+1}/10: reward {entry["update"]["mean_reward"]:.3f}, mixed groups {entry["update"]["mixed_groups"]}/2',flush=True)
            if 'training' not in report:report['training']=call(worker.finish);save()
            if 'local_adapter_path' not in report['training']:
                report['training']['local_adapter_path']=download_adapter(results_volume,run_id,'grpo',report['training']);save()
            evaluate('train-after',train,1);evaluate('sampled-train-after',train,4)
            for i,batch in enumerate(evaluation_batches(dev)):evaluate(f'validation-{i}',batch,1)
        before=report['evaluations']['train-before']['results'];after=report['evaluations']['train-after']['results']
        val=[r for k,v in report['evaluations'].items() if k.startswith('validation-') for r in v['results']]
        if len(val)!=32:raise ValueError('Incomplete validation')
        report['summary']={'training_problems':2,'training_greedy_before':sum(r['reward'] for r in before),'training_greedy_after':sum(r['reward'] for r in after),
            'sampled_training_before':sum(r['reward'] for r in report['rounds'][0]['results']),
            'sampled_training_after':sum(r['reward'] for r in report['evaluations']['sampled-train-after']['results']),'sampled_training_denominator':8,
            'validation_problems':32,'validation_before':baseline['summary']['passed'],'validation_after':sum(r['reward'] for r in val),
            'optimizer_steps':report['training']['optimizer_steps'],'rollouts':80,
            'mixed_groups':sum(x['update']['mixed_groups'] for x in report['rounds']),
            'total_groups':20}
        report['status']='completed';save();print(json.dumps(report['summary']),flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(path,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume')
    main(parser.parse_args().resume)
