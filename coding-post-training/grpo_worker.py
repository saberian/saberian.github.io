"""Stateful GRPO engine. Only trusted weights/state execute here; code grading stays local."""
import gc
import json
import re
import time
from pathlib import Path
import torch
from data import MODEL,MODEL_REVISION,digest,tokenize_prompt
from grpo_core import TRAINING_IDS,SETTINGS,group_advantages,completion_logps,apply_update,response_objective
from sft_core import lora_config,file_manifest


class Engine:
    def __init__(self,run_id,volume):
        if not re.fullmatch(r'grpo-[0-9TZ]+-[0-9a-f]{8}',run_id):raise ValueError('Invalid GRPO run ID')
        from peft import get_peft_model,set_peft_model_state_dict
        from transformers import AutoModelForCausalLM,AutoTokenizer,set_seed
        set_seed(42)
        self.volume=volume;volume.reload();self.root=Path('/results')/run_id;self.root.mkdir(parents=True,exist_ok=True)
        self.tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True)
        base=AutoModelForCausalLM.from_pretrained('/model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        self.model=get_peft_model(base,lora_config()).eval()
        if any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in self.model.modules()):raise ValueError('Rollout and update require zero dropout')
        parameters=[p for n,p in self.model.named_parameters() if p.requires_grad]
        if any('lora_' not in n for n,p in self.model.named_parameters() if p.requires_grad):raise ValueError('Backbone must remain frozen')
        self.optimizer=torch.optim.AdamW(parameters,lr=SETTINGS['learning_rate'],weight_decay=0)
        self.version=0;self.pending=None;self.history=[]
        if (self.root/'state.pt').exists():
            s=torch.load(self.root/'state.pt',map_location='cuda',weights_only=True)
            if s['settings']!=SETTINGS or s['model_revision']!=MODEL_REVISION:raise ValueError('Checkpoint configuration mismatch')
            set_peft_model_state_dict(self.model,s['adapter']);self.optimizer.load_state_dict(s['optimizer'])
            self.version=s['version'];self.pending=s['pending'];self.history=s['history']
        torch.cuda.reset_peak_memory_stats()

    def save(self):
        from peft import get_peft_model_state_dict
        state={'adapter':get_peft_model_state_dict(self.model),'optimizer':self.optimizer.state_dict(),
            'version':self.version,'pending':self.pending,'history':self.history,
            'settings':SETTINGS,'model_revision':MODEL_REVISION}
        tmp=self.root/'state.tmp';torch.save(state,tmp);tmp.replace(self.root/'state.pt')
        self.volume.commit()

    def generate(self,problems,samples,round_index=0,with_logps=False):
        from transformers import GenerationConfig
        if not 1<=len(problems)<=9 or samples not in (1,4):raise ValueError('Invalid inference batch')
        if any(set(p)!={'id','question','signature'} for p in problems):raise ValueError('Prompt-only inference required')
        outputs=[];trajectories=[]
        self.model.eval()
        for p in problems:
            inputs=tokenize_prompt(p,self.tokenizer,return_tensors='pt').to('cuda')
            prompt=inputs['input_ids'][0].tolist()
            if not 0<len(prompt)<=1024:raise ValueError('Prompt too long')
            seed=42 if samples==1 else int(digest([42,p['id']] if round_index==0 else [42,p['id'],round_index])[:8],16)
            torch.manual_seed(seed)
            config=GenerationConfig(max_new_tokens=512,do_sample=samples==4,num_return_sequences=samples,
                temperature=1.0 if samples==4 else None,top_p=1.0 if samples==4 else None,top_k=0 if samples==4 else None,
                eos_token_id=self.model.generation_config.eos_token_id,pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.model.generation_config.bos_token_id)
            start=time.monotonic()
            with torch.inference_mode():sequences=self.model.generate(**inputs,generation_config=config)
            elapsed=time.monotonic()-start
            eos=config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
            for i,sequence in enumerate(sequences):
                tokens=sequence[len(prompt):].tolist()
                end=next((j+1 for j,t in enumerate(tokens) if t in eos),None)
                tokens=tokens[:end] if end else tokens
                o={'id':p['id'],'sample_index':i,'seed':seed,'code':self.tokenizer.decode(tokens,skip_special_tokens=True),
                    'prompt_tokens':len(prompt),'generated_tokens':len(tokens),'terminated':end is not None,'generation_seconds':elapsed/samples}
                outputs.append(o)
                if with_logps:
                    with torch.no_grad():
                        old=completion_logps(self.model,prompt,tokens).cpu().tolist()
                        with self.model.disable_adapter():ref=completion_logps(self.model,prompt,tokens).cpu().tolist()
                    trajectories.append({'id':p['id'],'sample_index':i,'prompt_ids':prompt,'completion_ids':tokens,'old_logps':old,'reference_logps':ref})
            print(f'Generated {samples} answers for {p["id"]}, policy version {self.version}',flush=True)
        return outputs,trajectories

    def rollout(self,round_index,problems):
        if [p['id'] for p in problems]!=TRAINING_IDS or not 0<=round_index<SETTINGS['rounds']:raise ValueError('Frozen training cohort/round mismatch')
        if round_index!=self.version:raise ValueError('Stale or skipped rollout version')
        if self.pending is None:
            outputs,trajectories=self.generate(problems,4,round_index,True)
            self.pending={'round':round_index,'outputs':outputs,'trajectories':trajectories,'digest':digest(outputs)}
            self.save()
        return {k:self.pending[k] for k in ('round','outputs','digest')}

    def objective(self,trajectories,rewards):
        a=group_advantages(rewards);total=0
        with torch.no_grad():
            for t,adv in zip(trajectories,a):
                current=completion_logps(self.model,t['prompt_ids'],t['completion_ids'])
                loss,_=response_objective(current,torch.tensor(t['old_logps'],device='cuda'),torch.tensor(t['reference_logps'],device='cuda'),adv.to('cuda'))
                total+=float(loss)/len(trajectories)
        return total

    def update(self,round_index,outputs_digest,rewards):
        if self.version==round_index+1:
            last=self.history[-1]
            if last['outputs_digest']!=outputs_digest or last['rewards']!=rewards:raise ValueError('Conflicting duplicate update')
            return last
        if self.pending is None or self.pending['round']!=round_index or self.pending['digest']!=outputs_digest or len(rewards)!=8 or any(type(x) not in (int,float) or x not in (0,1) for x in rewards):raise ValueError('Rewards do not match pending rollout')
        trajectories=self.pending['trajectories'];before=self.objective(trajectories,rewards)
        started=time.monotonic()
        metrics=apply_update(self.model,self.optimizer,trajectories,rewards,epochs=2)
        after=self.objective(trajectories,rewards)
        groups=[rewards[i:i+4] for i in (0,4)]
        result={'round':round_index,'outputs_digest':outputs_digest,'rewards':rewards,'mean_reward':sum(rewards)/8,
            'mixed_groups':sum(len(set(g))>1 for g in groups),'zero_variance_groups':sum(len(set(g))==1 for g in groups),
            'objective_before':before,'objective_after':after,'updates':metrics,'update_seconds':time.monotonic()-started,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
        if round_index==0 and (result['mixed_groups']==0 or not after<before or not any(x['gradient_norm']>0 for x in metrics)):
            raise ValueError('Canary did not demonstrate mixed rewards and objective improvement')
        self.history.append(result);self.version+=1;self.pending=None;self.save()
        (self.root/f'round-{round_index}.json').write_text(json.dumps(result,indent=2));self.volume.commit()
        return result

    def verify_reload(self):
        from peft import get_peft_model,set_peft_model_state_dict,get_peft_model_state_dict
        from transformers import AutoModelForCausalLM
        ids=torch.tensor([[100,200,300,400]],device='cuda')
        with torch.no_grad():expected=self.model(input_ids=ids,use_cache=False).logits.float().cpu()
        base=AutoModelForCausalLM.from_pretrained('/model',local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
        loaded=get_peft_model(base,lora_config()).eval()
        s=torch.load(self.root/'state.pt',map_location='cuda',weights_only=True)
        set_peft_model_state_dict(loaded,s['adapter'])
        with torch.no_grad():actual=loaded(input_ids=ids,use_cache=False).logits.float().cpu()
        error=float((actual-expected).abs().max())
        if not torch.allclose(actual,expected,atol=1e-5,rtol=1e-5):raise ValueError('Checkpoint reload changed logits')
        del loaded,base;gc.collect();torch.cuda.empty_cache()
        return {'max_logit_error':error,'policy_version':self.version}

    def evaluate(self,problems,samples,label):
        if not re.fullmatch(r'[a-z0-9-]+',label):raise ValueError('Invalid evaluation label')
        dest=self.root/f'{label}.json'
        if dest.exists():
            result=json.loads(dest.read_text())
            if result['policy_version']!=self.version or result['prompt_sha256']!=digest(problems) or result['samples']!=samples:raise ValueError('Conflicting evaluation label')
            return result
        outputs,_=self.generate(problems,samples)
        result={'outputs':outputs,'policy_version':self.version,'prompt_sha256':digest(problems),'samples':samples}
        dest.write_text(json.dumps(result));self.volume.commit()
        return result

    def finish(self):
        if self.version!=SETTINGS['rounds'] or self.pending is not None:raise ValueError('Training is incomplete')
        adapter=self.root/'adapter'
        self.model.peft_config['default'].base_model_name_or_path=MODEL
        self.model.peft_config['default'].revision=MODEL_REVISION
        self.model.save_pretrained(adapter);self.volume.commit()
        return {'adapter_files':file_manifest(adapter),'volume_adapter_path':str(adapter.relative_to('/results')),
            'rounds':self.version,'optimizer_steps':2*self.version,
            'trainable_parameters':sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'history':self.history,'reload':self.verify_reload()}
