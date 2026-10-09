"""Shared, tested Transformers/PEFT training contract for the first SFT pilot."""
import hashlib
import json
from pathlib import Path

TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
SETTINGS = dict(rank=16, alpha=32, dropout=0.0, learning_rate=1e-4,
    microbatch=1, accumulation=16, epochs=1, seed=42, max_grad_norm=1.0)


def validate_records(records, selected_ids, eos):
    if len(records) != len(selected_ids) or {r['id'] for r in records} != set(selected_ids):
        raise ValueError('Training IDs differ from the frozen SFT split')
    for r in records:
        ids, labels, n = r['input_ids'], r['labels'], r['prompt_tokens']
        if not 0 < n <= 1024 or not 1 < len(ids) - n <= 512:
            raise ValueError('Invalid prompt/target length')
        if len(ids) != n + r['target_tokens'] or r['attention_mask'] != [1]*len(ids):
            raise ValueError('Invalid length/attention metadata')
        if labels != [-100]*n + ids[n:] or ids[-1] != eos:
            raise ValueError('Answer-only labels or EOS changed')


def lora_config():
    from peft import LoraConfig
    return LoraConfig(r=SETTINGS['rank'], lora_alpha=SETTINGS['alpha'],
        lora_dropout=0.0, target_modules=TARGETS, bias='none', task_type='CAUSAL_LM')


def collator(tokenizer):
    from functools import partial
    return partial(pad_records, pad_id=tokenizer.eos_token_id)


def pad_records(records, pad_id):
    import torch
    from torch.nn.utils.rnn import pad_sequence
    ids = [torch.tensor(r['input_ids'],dtype=torch.long) for r in records]
    labels = [torch.tensor(r['labels'],dtype=torch.long) for r in records]
    return {'input_ids':pad_sequence(ids,batch_first=True,padding_value=pad_id),
        'labels':pad_sequence(labels,batch_first=True,padding_value=-100),
        'attention_mask':pad_sequence([torch.ones_like(x) for x in ids],batch_first=True,padding_value=0)}


def answer_loss(outputs, labels, num_items_in_batch=None):
    import torch
    targets = labels[:,1:].contiguous()
    logits = outputs.logits[:,:-1].float().contiguous()
    total = torch.nn.functional.cross_entropy(logits.view(-1,logits.shape[-1]),
        targets.view(-1),ignore_index=-100,reduction='sum')
    count = targets.ne(-100).sum() if num_items_in_batch is None else num_items_in_batch
    return total / torch.as_tensor(count,device=total.device).clamp(min=1)


def make_trainer(model, tokenizer, records, output_dir, *, canary=False, cpu=False):
    from datasets import Dataset
    from transformers import TrainingArguments, Trainer
    args = TrainingArguments(output_dir=str(output_dir), num_train_epochs=1,
        max_steps=8 if canary else -1, learning_rate=SETTINGS['learning_rate'],
        per_device_train_batch_size=1, gradient_accumulation_steps=1 if canary else 16,
        lr_scheduler_type='constant', warmup_steps=0, optim='adamw_torch', weight_decay=0.0,
        max_grad_norm=1.0, bf16=not cpu, fp16=False, use_cpu=cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant':False},
        seed=42, data_seed=42, report_to='none',
        logging_steps=1, logging_nan_inf_filter=False, save_strategy='no',
        eval_strategy='no', disable_tqdm=True, dataloader_pin_memory=not cpu)
    dataset = Dataset.from_list([{k:r[k] for k in ('input_ids','labels')} for r in records])
    model.config.use_cache = False
    return Trainer(model=model, args=args, train_dataset=dataset,
        processing_class=tokenizer, data_collator=collator(tokenizer), compute_loss_func=answer_loss)


def inference_model(trainer):
    """Remove training-only autocast hooks before measuring/saving inference behavior."""
    model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    model.gradient_checkpointing_disable()
    return model.eval()


def mean_nll(model, tokenizer, records):
    """Token-weighted reference-answer NLL, measured consistently before/after."""
    import torch
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for r in records:
            batch = {k:v.to(model.device) for k,v in collator(tokenizer)([r]).items()}
            n = int(batch['labels'][:,1:].ne(-100).sum())
            labels = batch.pop('labels')
            out = model(**batch, use_cache=False)
            total += float(answer_loss(out, labels)) * n
            count += n
    return total/count


def file_manifest(directory):
    return {p.name: {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes':p.stat().st_size}
        for p in Path(directory).iterdir() if p.is_file()}


def train_adapter(records, run_id, stage, volume):
    import gc
    import importlib.metadata
    import time
    import torch
    from peft import PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from data import MODEL, MODEL_REVISION
    started = time.monotonic()
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained('/model',local_files_only=True)
    validate_records(records, [r['id'] for r in records], tokenizer.eos_token_id)
    chosen = records[:1] if stage == 'canary' else records
    dest = Path('/results') / run_id / stage
    if dest.exists():
        raise ValueError('Refusing to overwrite an existing training run')
    dest.mkdir(parents=True)
    def load_base():
        return AutoModelForCausalLM.from_pretrained('/model',local_files_only=True,
            dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
    base = load_base()
    model = get_peft_model(base, lora_config())
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    if not trainable or any('lora_' not in n for n in trainable):
        raise ValueError('Only LoRA parameters should be trainable')
    torch.cuda.reset_peak_memory_stats()
    before = mean_nll(model,tokenizer,chosen)
    trainer = make_trainer(model,tokenizer,chosen,dest/'trainer',canary=stage=='canary')
    output = trainer.train()
    model = inference_model(trainer)
    after = mean_nll(model,tokenizer,chosen)
    if not torch.isfinite(torch.tensor([before,after])).all():
        raise ValueError('Nonfinite reference loss')
    if stage == 'canary' and not after < before * 0.95:
        raise ValueError('Tiny overfit check did not reduce loss by at least 5%')
    adapter = dest/'adapter'
    model.peft_config['default'].base_model_name_or_path = MODEL
    model.peft_config['default'].revision = MODEL_REVISION
    model.save_pretrained(adapter)
    # Compare real logits before and after reloading from the saved artifact.
    probe = torch.tensor([chosen[0]['input_ids'][:32]],device='cuda')
    model.eval()
    with torch.no_grad():
        expected = model(input_ids=probe,use_cache=False).logits.float().cpu()
    report = {'stage':stage,'settings':SETTINGS | ({'accumulation':1,'epochs':8} if stage=='canary' else {}),'examples':len(chosen),
        'optimizer_steps':trainer.state.global_step,'reference_nll_before':before,'reference_nll_after':after,
        'training_metrics':output.metrics,'log_history':trainer.state.log_history,
        'trainable_parameters':sum(p.numel() for p in trainable.values()),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        'supervised_tokens_per_epoch':sum(r['target_tokens'] for r in chosen),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','datasets','accelerate')},
        'gpu':torch.cuda.get_device_name()}
    del trainer,model,base,trainable
    gc.collect(); torch.cuda.empty_cache()
    reloaded = PeftModel.from_pretrained(load_base(),adapter,is_trainable=False).eval()
    with torch.no_grad():
        actual = reloaded(input_ids=probe,use_cache=False).logits.float().cpu()
    max_error = float((expected-actual).abs().max())
    report['adapter_reload_max_logit_error'] = max_error
    if not torch.allclose(expected,actual,atol=1e-5,rtol=1e-5):
        raise ValueError(f'Adapter reload changed logits: {max_error}')
    report['adapter_files'] = file_manifest(adapter)
    report['volume_adapter_path'] = f'{run_id}/{stage}/adapter'
    report['function_seconds'] = time.monotonic()-started
    (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    volume.commit()
    print(json.dumps(report),flush=True)
    return report
