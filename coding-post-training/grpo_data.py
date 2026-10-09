"""Freeze the two independently checked, mixed-outcome alignment tasks."""
import json
from data import ROOT,digest,MODEL_REVISION
from discovery_data import write_immutable
from evaluator import EVALUATOR_VERSION,evaluate_with_runner
from local_runner import image_identity,run_docker

from grpo_core import TRAINING_IDS as IDS

def load_training():
    d=json.loads((ROOT/'data/discovery-v1.json').read_text())
    m=json.loads((ROOT/'manifests/discovery-v1.json').read_text())
    pools=json.loads((ROOT/'manifests/family-pools-v1.json').read_text())
    if digest(d['problems'])!=m['sample_sha256'] or digest(pools)!=m['split_manifest_sha256']:raise ValueError('Discovery provenance changed')
    byid={p['id']:p for p in d['problems']};ps=[byid[i] for i in IDS]
    if any(p['split']!='alignment_train' or pools['assignments'][p['id']]!={k:p[k] for k in ('split','family_id')} for p in ps):raise ValueError('Alignment pool violation')
    return ps,m

def prepare():
    ps,source=load_training();runtime=image_identity()
    if runtime!=source['reference_runtime']:raise ValueError('Evaluator runtime changed')
    checks=[]
    for p in ps:
        result=evaluate_with_runner(p,p['reference_solution'],lambda req:run_docker(req,runtime['id']))
        if result['reward']!=1:raise ValueError('Reference failed')
        checks.append({'id':p['id'],'status':result['status'],'cases':len(p['cases'])})
    m={'selected_ids':IDS,'sample_sha256':digest(ps),'discovery_manifest_sha256':digest(source),
        'model_revision':MODEL_REVISION,'reference_runtime':runtime,'reference_checks':checks,
        'evaluator_version':EVALUATOR_VERSION,'selection':'The two previously independently audited mixed-outcome alignment groups; chosen from training-only discovery results, never validation/test scores.',
        'limitations':'Only two training problems; diagnostic of learning mechanics, not broad generalization. Preserve original supplied-test rewards and prompt ambiguities, including a singleton palindrome case outside the stated minimum length.'}
    write_immutable(ROOT/'manifests/grpo-pilot-v1.json',m)
    write_immutable(ROOT/'data/grpo-pilot-v1.json',m|{'problems':ps})
    print(json.dumps(m,indent=2))

if __name__=='__main__':prepare()
