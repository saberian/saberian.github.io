"""Real Transformers/PEFT checks on a tiny randomly initialized Qwen, entirely on CPU."""
import copy
import tempfile
import unittest

import torch
from peft import PeftModel,get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast,Qwen3Config,Qwen3ForCausalLM
from sft_core import collator,lora_config,make_trainer,mean_nll,validate_records

class SFTContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(42)
        self.tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'<pad>':0,'<unk>':1,'<eos>':2,**{str(i):i for i in range(3,32)}},unk_token='<unk>')),
            eos_token='<eos>',pad_token='<pad>',unk_token='<unk>')
        self.config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,
            num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=128,attention_dropout=0.0)
        self.records=[{'id':'a','input_ids':[3,4,5,6,2],'labels':[-100,-100,5,6,2],
            'attention_mask':[1]*5,'prompt_tokens':2,'target_tokens':3},
            {'id':'b','input_ids':[7,8,9,2],'labels':[-100,-100,9,2],
            'attention_mask':[1]*4,'prompt_tokens':2,'target_tokens':2}]
    def model(self):
        return get_peft_model(Qwen3ForCausalLM(self.config),lora_config())
    def test_export_guards_and_real_collator(self):
        validate_records(self.records,['a','b'],2)
        batch=collator(self.tokenizer)(self.records)
        self.assertEqual(batch['labels'].tolist(),[[-100,-100,5,6,2],[-100,-100,9,2,-100]])
        self.assertEqual(batch['attention_mask'][1].tolist(),[1,1,1,1,0])
        for bad in ('labels','id','attention_mask'):
            rs=copy.deepcopy(self.records)
            rs[0][bad]=([3,4,5,6,2] if bad=='labels' else 'x' if bad=='id' else [1]*4)
            with self.assertRaises(ValueError):validate_records(rs,['a','b'],2)
    def test_trainer_loss_matches_shifted_cross_entropy_and_accumulation(self):
        model=self.model().eval()
        with tempfile.TemporaryDirectory() as d:
            trainer=make_trainer(model,self.tokenizer,self.records,d,cpu=True)
            batch=collator(self.tokenizer)(self.records)
            with torch.no_grad():
                logits=model(input_ids=batch['input_ids'],attention_mask=batch['attention_mask']).logits
                expected=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,32),batch['labels'][:,1:].reshape(-1),ignore_index=-100)
            full=trainer.compute_loss(model,dict(batch),num_items_in_batch=torch.tensor(5))
            self.assertAlmostEqual(float(full.detach()),float(expected),places=5)
            full.backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad()
            for r in self.records:
                trainer.compute_loss(model,collator(self.tokenizer)([r]),num_items_in_batch=torch.tensor(5)).backward()
            for n,p in model.named_parameters():
                if n in grads:self.assertTrue(torch.allclose(grads[n],p.grad,atol=1e-6,rtol=1e-4),n)
    def test_optimizer_update_matches_full_batch_with_unequal_answer_lengths(self):
        first=self.model();second=copy.deepcopy(first)
        with tempfile.TemporaryDirectory() as d:
            a=make_trainer(first,self.tokenizer,self.records,d+'/a',cpu=True)
            b=make_trainer(second,self.tokenizer,self.records,d+'/b',cpu=True)
            a.args.max_steps=b.args.max_steps=1
            a.args.gradient_accumulation_steps=2
            b.args.gradient_accumulation_steps=1
            b.args.per_device_train_batch_size=2
            # Trainer caches batch size during construction; rebuild with the reference settings.
            b=type(b)(model=second,args=b.args,train_dataset=b.train_dataset,
                processing_class=self.tokenizer,data_collator=b.data_collator,compute_loss_func=b.compute_loss_func)
            a.train();b.train()
            for (n,p),(name,q) in zip(first.named_parameters(),second.named_parameters()):
                self.assertEqual(n,name)
                self.assertTrue(torch.allclose(p,q,atol=1e-6,rtol=1e-4),n)

    def test_real_training_freezes_backbone_and_reload_preserves_logits(self):
        model=self.model()
        original={n:p.detach().clone() for n,p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as d:
            trainer=make_trainer(model,self.tokenizer,self.records,d,canary=True,cpu=True)
            trainer.train()
            self.assertEqual(trainer.state.global_step,8)
            self.assertTrue(any(not torch.equal(original[n],p) for n,p in model.named_parameters() if p.requires_grad))
            for n,p in model.named_parameters():
                if not p.requires_grad:self.assertTrue(torch.equal(original[n],p),n)
            model.eval();batch=collator(self.tokenizer)(self.records)
            with torch.no_grad():expected=model(**batch).logits
            model.save_pretrained(d+'/adapter')
            base=Qwen3ForCausalLM(self.config)
            # Restore the identical random backbone, then load only saved adapter weights.
            base.load_state_dict({n.removeprefix('base_model.model.').replace('.base_layer.','.'):p
                for n,p in original.items() if 'lora_' not in n})
            loaded=PeftModel.from_pretrained(base,d+'/adapter').eval()
            with torch.no_grad():actual=loaded(**batch).logits
            self.assertTrue(torch.allclose(expected,actual,atol=1e-6,rtol=1e-6))
            self.assertTrue(torch.isfinite(torch.tensor(mean_nll(loaded,self.tokenizer,self.records))))
