"""Freeze 300 held-out, reference-validated tasks before either model is scored."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from data import ROOT, DATASET, DATA_REVISION, digest, parse_problem, tokenize_prompt, MODEL, MODEL_REVISION
from discovery_data import write_immutable
from pilot_data import implementation_key
from evaluator import evaluate_with_runner
from local_runner import image_identity, run_docker


def validate_test(problems, pools, pilot):
    if len(problems)!=300 or len({p['id'] for p in problems})!=300:
        raise ValueError('Expected 300 unique test problems')
    families={p['family_id'] for p in pilot}
    implementations={p['implementation_key'] for p in pilot}
    for p in problems:
        if pools['assignments'][p['id']]!={'split':'final_test','family_id':p['family_id']}:
            raise ValueError('Test pool assignment mismatch')
        if p['family_id'] in families or p['implementation_key'] in implementations:
            raise ValueError('Duplicate family or implementation')
        families.add(p['family_id']);implementations.add(p['implementation_key'])


def prepare():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    pools=json.loads((ROOT/'manifests/family-pools-v1.json').read_text())
    pilot=json.loads((ROOT/'data/sft-pilot-v1.json').read_text())['problems']
    source=ROOT/'data/source/data/train-00000-of-00001.parquet'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=pools['source_sha256']:
        raise ValueError('Source hash mismatch')
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=MODEL_REVISION)
    exclusions=json.loads((ROOT/'manifests/audit-exclusions.json').read_text())
    for name in ('discovery-audit','sft-pilot-audit'):
        exclusions.update(json.loads((ROOT/f'manifests/{name}.json').read_text())['exclusions'])
    candidates=[];rejections=Counter()
    for row in pq.read_table(source).to_pylist():
        assignment=pools['assignments'][row['question_id']]
        if assignment['split']!='final_test':continue
        try:
            if row['question_id'] in exclusions:raise ValueError('Existing audit exclusion')
            p=parse_problem(row)|assignment
            p['implementation_key']=implementation_key(p['reference_solution'])
            p['prompt_tokens']=len(tokenize_prompt(p,tokenizer)['input_ids'])
            if p['prompt_tokens']>1024:raise ValueError('Prompt too long')
            candidates.append(p)
        except (ValueError,TypeError,SyntaxError,KeyError) as exc:rejections[str(exc)]+=1
    candidates.sort(key=lambda p:digest([42,'final-test-v1',p['id']]))
    identity=image_identity();checks=[];selected=[]
    families={p['family_id'] for p in pilot};implementations={p['implementation_key'] for p in pilot}
    def check(p):
        return [evaluate_with_runner(p,p['reference_solution'],lambda req:run_docker(req,identity['id']))['status'] for _ in range(2)]
    remaining=iter(candidates)
    with ThreadPoolExecutor(max_workers=4) as executor:
        while len(selected)<300:
            batch=[]
            while len(batch)<min(4,300-len(selected)):
                p=next(remaining,None)
                if p is None:break
                if p['family_id'] in families or p['implementation_key'] in implementations:continue
                families.add(p['family_id']);implementations.add(p['implementation_key']);batch.append(p)
            if not batch:raise RuntimeError('Insufficient eligible test families')
            for p,statuses in zip(batch,executor.map(check,batch)):
                checks.append({'id':p['id'],'statuses':statuses})
                if statuses==['passed','passed']:selected.append(p)
            print(f'Reference validated {len(selected)}/300',flush=True)
    validate_test(selected,pools,pilot)
    metadata={'dataset':DATASET,'revision':DATA_REVISION,'sample_sha256':digest(selected),
        'split_manifest_sha256':digest(pools),'reference_runtime':identity,'reference_checks':checks,
        'selection':'SHA256([42, final-test-v1, id]); final_test pool only; unique families and normalized implementations, disjoint from pilot; references pass twice; no model scores used.',
        'limitations':'Automatic reference agreement is not a complete manual semantic audit. Public dataset pretraining contamination is unknown.',
        'rejections':dict(rejections)}
    write_immutable(ROOT/'data/final-test-v1.json',metadata|{'problems':selected})
    write_immutable(ROOT/'manifests/final-test-v1.json',metadata|{'selected':[{k:p[k] for k in ('id','family_id','implementation_key','prompt_tokens')} for p in selected]})

if __name__=='__main__':prepare()
