import copy
import math
import unittest
import torch
from peft import get_peft_model
from transformers import Qwen3Config,Qwen3ForCausalLM
from sft_core import lora_config
from grpo_core import group_advantages,response_objective,completion_logps,apply_update

class GRPOContracts(unittest.TestCase):
    def test_group_relative_advantages_and_constant_groups(self):
        a=group_advantages([0,1,0,1,1,1,1,1])
        self.assertTrue(torch.allclose(a[:4],torch.tensor([-1,1,-1,1])/1.0002))
        self.assertTrue(torch.equal(a[4:],torch.zeros(4)))
        with self.assertRaises(ValueError):group_advantages([0,1,0])
        with self.assertRaises(ValueError):group_advantages([0,1,0,float('nan')])

    def test_clipping_both_advantage_signs_kl_and_length_reduction(self):
        z=torch.zeros(2)
        loss,_=response_objective(torch.tensor([math.log(1.4),0]),z,z,1.,beta=0)
        self.assertAlmostEqual(float(loss),-1.1,places=6)
        loss,_=response_objective(torch.tensor([math.log(.5),0]),z,z,-1.,beta=0)
        self.assertAlmostEqual(float(loss),.9,places=6)
        loss,metrics=response_objective(z,z,torch.ones(2),0.,beta=.02)
        self.assertAlmostEqual(float(loss),.02*(math.e-2),places=6)
        for n in (1,3,17):
            x=torch.zeros(n,requires_grad=True)
            loss,_=response_objective(x,torch.zeros(n),torch.zeros(n),1.)
            self.assertEqual(float(loss.detach()),-1.)
            loss.backward();self.assertTrue(torch.allclose(x.grad,torch.full((n,),-1/n)))

    def test_real_model_completion_alignment_and_reference_freezing(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=128,attention_dropout=0.0)
        model=get_peft_model(Qwen3ForCausalLM(config),lora_config()).eval()
        prompt=[3,4];completions=[[5,2],[6,7,2],[8,2],[9,10,11,2]]
        frozen={n:p.clone().detach() for n,p in model.named_parameters() if not p.requires_grad}
        ts=[]
        with torch.no_grad():
            for c in completions:
                old=completion_logps(model,prompt,c)
                logits=model(torch.tensor([prompt+c]),use_cache=False).logits
                expected=torch.log_softmax(logits[0,1:-1].float(),-1).gather(1,torch.tensor(c)[:,None]).squeeze(1)
                self.assertTrue(torch.equal(old,expected));self.assertEqual(len(old),len(c))
                with model.disable_adapter():ref=completion_logps(model,prompt,c)
                ts.append({'prompt_ids':prompt,'completion_ids':c,'old_logps':old.tolist(),'reference_logps':ref.tolist()})
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=0)
        metrics=apply_update(model,opt,ts,[0,1,0,1])
        self.assertEqual(len(metrics),2);self.assertGreater(metrics[0]['gradient_norm'],0)
        self.assertLess(metrics[1]['loss'],metrics[0]['loss'])
        for n,p in model.named_parameters():
            if n in frozen:self.assertTrue(torch.equal(p,frozen[n]),n)
        with torch.no_grad(),model.disable_adapter():
            for t in ts:self.assertTrue(torch.equal(completion_logps(model,t['prompt_ids'],t['completion_ids']),torch.tensor(t['reference_logps'])))

    def test_update_idempotency_rejects_conflicting_reward_delivery(self):
        from grpo_worker import Engine
        e=Engine.__new__(Engine);e.version=1;e.history=[{'outputs_digest':'a','rewards':[0,1]*4}]
        self.assertEqual(e.update(0,'a',[0,1]*4),e.history[0])
        with self.assertRaises(ValueError):e.update(0,'b',[0,1]*4)
        with self.assertRaises(ValueError):e.update(0,'a',[1]*8)

    def test_optimizer_checkpoint_roundtrip_preserves_next_update(self):
        import tempfile
        from pathlib import Path
        from peft import get_peft_model_state_dict,set_peft_model_state_dict
        torch.set_num_threads(1);torch.manual_seed(7)
        cfg=Qwen3Config(vocab_size=16,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,head_dim=8,max_position_embeddings=64,attention_dropout=0.)
        first=get_peft_model(Qwen3ForCausalLM(cfg),lora_config()).eval();second=copy.deepcopy(first)
        oa=torch.optim.AdamW([p for p in first.parameters() if p.requires_grad],lr=.001,weight_decay=0)
        ob=torch.optim.AdamW([p for p in second.parameters() if p.requires_grad],lr=.001,weight_decay=0)
        def trajectories(m):
            out=[]
            with torch.no_grad():
                for c in ([5,2],[6,2],[7,2],[8,2]):
                    old=completion_logps(m,[3,4],list(c)).tolist()
                    with m.disable_adapter():ref=completion_logps(m,[3,4],list(c)).tolist()
                    out.append({'prompt_ids':[3,4],'completion_ids':list(c),'old_logps':old,'reference_logps':ref})
            return out
        apply_update(first,oa,trajectories(first),[0,1,0,1])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';torch.save({'adapter':get_peft_model_state_dict(first),'optimizer':oa.state_dict()},path)
            saved=torch.load(path,weights_only=True)
            set_peft_model_state_dict(second,saved['adapter']);ob.load_state_dict(saved['optimizer'])
        ts=trajectories(first)
        apply_update(first,oa,ts,[0,1,0,1]);apply_update(second,ob,ts,[0,1,0,1])
        for (n,p),(name,q) in zip(first.named_parameters(),second.named_parameters()):
            self.assertEqual(n,name);self.assertTrue(torch.equal(p,q),n)
