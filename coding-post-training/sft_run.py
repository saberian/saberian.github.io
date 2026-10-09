"""Run a bounded SFT canary, fresh one-epoch pilot, and paired development evaluation."""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

from data import ROOT,MODEL,MODEL_REVISION,digest
from discovery import RESOURCE_USD_PER_SECOND
from evaluator import EVALUATOR_VERSION,evaluate_with_runner
from local_runner import image_identity,run_docker
from pilot_data import validate_splits,training_record
from sft_core import SETTINGS,validate_records

BUDGET_USD=4.0
SETUP_RESERVE_USD=0.25


def load_data():
    from transformers import AutoTokenizer
    manifest=json.loads((ROOT/'manifests/sft-pilot-v1.json').read_text())
    artifact=json.loads((ROOT/'data/sft-pilot-v1.json').read_text())
    pools=json.loads((ROOT/'manifests/family-pools-v1.json').read_text())
    audit=json.loads((ROOT/'manifests/sft-pilot-audit.json').read_text())
    baseline=json.loads((ROOT/'reports/sft-pilot-baseline-2026-10-09.json').read_text())
    raw=(ROOT/'data/sft-pilot-v1-train.jsonl').read_bytes()
    validate_splits(artifact['problems'],pools)
    if digest(artifact['problems'])!=manifest['sample_sha256'] or digest(pools)!=manifest['split_manifest_sha256'] or digest(audit)!=manifest['audit_sha256'] or hashlib.sha256(raw).hexdigest()!=manifest['training_export_sha256']:
        raise ValueError('Frozen artifact hash mismatch')
    records=[json.loads(line) for line in raw.splitlines()]
    train=[p for p in artifact['problems'] if p['split']=='sft_train']
    dev=[p for p in artifact['problems'] if p['split']=='development']
    tok=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION)
    validate_records(records,[p['id'] for p in train],tok.eos_token_id)
    if records != [training_record(p,tok) for p in train]:
        raise ValueError('Frozen export no longer matches canonical tokenizer/prompt contract')
    if digest(dev)!=baseline['development_sha256'] or EVALUATOR_VERSION!=baseline['evaluator_version']:
        raise ValueError('Before/after evaluation contract differs')
    if MODEL_REVISION!=baseline['model_revision'] or image_identity()!=baseline['reference_runtime']:
        raise ValueError('Model or evaluation runtime differs from baseline')
    return records,dev,manifest,baseline


def download_adapter(volume,run_id,stage,report):
    dest=ROOT/'checkpoints'/run_id/stage/'adapter'
    dest.mkdir(parents=True,exist_ok=True)
    for name,info in report['adapter_files'].items():
        if Path(name).name!=name:
            raise ValueError('Invalid adapter filename')
        data=b''.join(volume.read_file(report['volume_adapter_path']+'/'+name))
        if len(data)!=info['bytes'] or hashlib.sha256(data).hexdigest()!=info['sha256']:
            raise ValueError('Downloaded adapter hash mismatch')
        (dest/name).write_bytes(data)
    return str(dest)


def main():
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip() or revision!=subprocess.check_output(['git','rev-parse','@{upstream}'],cwd=ROOT,text=True).strip():
        raise RuntimeError('Commit and push the exact runner before paid execution')
    records,dev,manifest,baseline=load_data()
    runner=lambda req:run_docker(req,baseline['reference_runtime']['id'])
    if evaluate_with_runner({'entry_point':'f','cases':[{'args':[7],'expected':7}]},'def f(x): return x',runner)['reward']!=1:
        raise RuntimeError('Local evaluator canary failed')
    run_id=datetime.now(timezone.utc).strftime('sft-%Y%m%dT%H%M%SZ-')+revision[:8]
    dest=ROOT/'runs'/f'{run_id}.json';dest.parent.mkdir(exist_ok=True)
    report={'run_id':run_id,'revision':revision,'status':'running','model':MODEL,'model_revision':MODEL_REVISION,
        'settings':SETTINGS,'training_export_sha256':manifest['training_export_sha256'],
        'sample_sha256':manifest['sample_sha256'],'development_sha256':digest(dev),
        'baseline_run_id':baseline['run_id'],'evaluator_version':EVALUATOR_VERSION,
        'decoding':baseline['decoding'],'reference_runtime':baseline['reference_runtime'],
        'budget_usd':BUDGET_USD,'setup_reserve_usd':SETUP_RESERVE_USD,
        'resource_usd_per_second':RESOURCE_USD_PER_SECOND,'charged_wall_seconds':0,
        'raw_outputs':[],'results':[],'gpu_batches':[]}
    def save():
        temp=dest.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(dest)
    def call(fn,args,timeout):
        maximum=300+timeout+2
        if SETUP_RESERVE_USD+(report['charged_wall_seconds']+maximum)*RESOURCE_USD_PER_SECOND>BUDGET_USD:
            raise RuntimeError('Stage budget cannot reserve the next complete call')
        started=time.monotonic()
        try:return fn.remote(*args)
        finally:
            report['charged_wall_seconds']+=time.monotonic()-started+2
            report['estimated_usd_with_setup_reserve']=SETUP_RESERVE_USD+report['charged_wall_seconds']*RESOURCE_USD_PER_SECOND
            save()
    by_id={p['id']:p for p in dev}
    def grade(outputs):
        for out in outputs:
            p=by_id[out['id']]
            result=evaluate_with_runner(p,out['code'],runner) if out['terminated'] else {
                'status':'truncated','reward':0,'passed_cases':0,'total_cases':len(p['cases']),'raw_format_compliant':False}
            report['results'].append({**out,**result});save()
    try:
        import modal
        from modal_app import app,results_volume,sft_canary,sft_train,sft_generate
        with modal.enable_output(),app.run():
            report['app_id']=app.app_id;save()
            # Canary learns a single training row, saves/reloads it, and is discarded.
            report['canary']=call(sft_canary,(records,run_id),300);save()
            report['canary']['local_adapter_path']=download_adapter(results_volume,run_id,'canary',report['canary']);save()
            print('Canary passed loss decrease, adapter reload, artifact download and hash checks.',flush=True)
            # Fresh base and fresh optimizer; canary updates never enter the pilot.
            report['training']=call(sft_train,(records,run_id),900);save()
            if report['training']['optimizer_steps']!=4:
                raise ValueError('Expected exactly four full effective-batch updates')
            report['training']['local_adapter_path']=download_adapter(results_volume,run_id,'train',report['training']);save()
            batches=[dev[:1]]+[dev[i:i+9] for i in range(1,32,9)]
            for i,batch in enumerate(batches):
                prompts=[{k:p[k] for k in ('id','question','signature')} for p in batch]
                result=call(sft_generate,(prompts,run_id,i),240)
                report['gpu_batches'].append({k:v for k,v in result.items() if k!='outputs'})
                report['raw_outputs'].extend(result['outputs']);save()
                if [o['id'] for o in result['outputs']]!=[p['id'] for p in batch] or any(o['sample_index']!=0 for o in result['outputs']):
                    raise ValueError('Unexpected inference IDs or sample count')
                if i==0:grade(result['outputs'])
                print(f'Saved {len(report["raw_outputs"])}/32 SFT development answers.',flush=True)
        grade(report['raw_outputs'][1:])
        before={r['id']:r for r in baseline['results']}
        report['summary']={'problems':32,'passed':sum(r['reward'] for r in report['results']),
            'status_counts':dict(Counter(r['status'] for r in report['results'])),
            'raw_format_passed':sum(r['raw_format_compliant'] for r in report['results']),
            'generated_tokens':sum(r['generated_tokens'] for r in report['results']),
            'improved_ids':[r['id'] for r in report['results'] if r['reward']>before[r['id']]['reward']],
            'regressed_ids':[r['id'] for r in report['results'] if r['reward']<before[r['id']]['reward']]}
        report['status']='completed';save();print(json.dumps(report['summary']),flush=True)
    except Exception as exc:
        report['status']='failed';report['error']=f'{type(exc).__name__}: {exc}';save();raise
    print(dest,flush=True)


if __name__=='__main__':main()
