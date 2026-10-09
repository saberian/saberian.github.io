"""Select a diagnostic 15-example training subset: all 7 baseline failures + 8 passes."""
import json
from data import ROOT,digest
from discovery_data import write_immutable
from sft_run import load_data


def select(problems, results):
    train={p['id']:p for p in problems if p['split']=='sft_train'}
    scores={r['id']:r for r in results}
    if len(scores)!=64 or scores.keys()!=train.keys():raise ValueError('Expected the paired 64-example training baseline')
    failed=sorted((i for i in train if scores[i]['reward']==0))
    passed=sorted((i for i in train if scores[i]['reward']==1),key=lambda i:digest([42,'overfit-15-v1',i]))[:8]
    if len(failed)!=7 or len(passed)!=8:raise ValueError('Expected seven failures and eight successes')
    return [train[i] for i in failed+passed]


def prepare():
    records,_,pilot,_=load_data()
    artifact=json.loads((ROOT/'data/sft-pilot-v1.json').read_text())
    baseline=json.loads((ROOT/'reports/base-training-accuracy-2026-10-09.json').read_text())
    train=[p for p in artifact['problems'] if p['split']=='sft_train']
    if baseline['evaluation_problems_sha256']!=digest(train):raise ValueError('Baseline data mismatch')
    selected=select(artifact['problems'],baseline['results'])
    byid={r['id']:r for r in records};export=[byid[p['id']] for p in selected]
    scores={r['id']:r for r in baseline['results']}
    meta={'source_manifest_sha256':digest(pilot),'baseline_run_id':baseline['run_id'],
        'selection':'All seven original-model failures from the frozen 64 training examples, plus eight successes ordered by SHA256([42, overfit-15-v1, id]). No validation/test examples.',
        'purpose':'Deliberately selected in-sample memorization diagnostic; not an unbiased evaluation or evidence of generalization.',
        'sample_sha256':digest(selected),'records_sha256':digest(export),
        'before_passed':8,'problems':15,'reference_runtime':baseline['reference_runtime'],
        'decoding':baseline['decoding'],'evaluator_version':baseline['evaluator_version'],
        'supervised_tokens_per_epoch':sum(r['target_tokens'] for r in export)}
    write_immutable(ROOT/'data/overfit-15-v1.json',meta|{'selected':selected,'records':export})
    write_immutable(ROOT/'manifests/overfit-15-v1.json',meta|{'selected':[
        {k:p[k] for k in ('id','family_id','split','prompt_tokens','target_tokens')}|{'baseline_status':scores[p['id']]['status'],'baseline_reward':scores[p['id']]['reward']} for p in selected]})
    for p in selected:print(p['id'],scores[p['id']]['status'],p['signature'])

if __name__=='__main__':prepare()
