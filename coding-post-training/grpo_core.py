"""Explicit original-style GRPO objective; shared by CPU tests and the GPU worker."""
import torch

TRAINING_IDS=['Prefill_19551_I','Prefill_35466_I']

SETTINGS={'rounds':10,'group_size':4,'update_epochs':2,'learning_rate':1e-5,
    'beta':0.02,'clip_epsilon':0.2,'advantage_epsilon':1e-4,'seed':42,
    'temperature':1.0,'top_p':1.0,'top_k':0,'max_new_tokens':512,
    'std_correction':0,'loss_reduction':'mean of per-response token means',
    'max_grad_norm':1.0,'reference':'original pinned model with adapters disabled'}


def group_advantages(rewards, group_size=4):
    r=torch.as_tensor(rewards,dtype=torch.float32)
    if r.ndim!=1 or len(r)%group_size or not torch.isfinite(r).all():
        raise ValueError('Invalid complete reward groups')
    r=r.reshape(-1,group_size)
    std=r.std(dim=1,correction=0,keepdim=True)
    return ((r-r.mean(dim=1,keepdim=True))/(std+SETTINGS['advantage_epsilon'])).flatten()


def response_objective(current, old, reference, advantage, *, beta=.02, epsilon=.2):
    if current.ndim!=1 or current.shape!=old.shape or current.shape!=reference.shape or not current.numel():
        raise ValueError('Log probabilities must align over completion tokens only')
    ratio=torch.exp(current-old.detach())
    delta=reference.detach()-current
    kl=torch.expm1(delta)-delta
    surrogate=torch.minimum(ratio*advantage,ratio.clamp(1-epsilon,1+epsilon)*advantage)
    loss=(-surrogate+beta*kl).mean()
    if not torch.isfinite(loss):raise ValueError('Nonfinite GRPO objective')
    return loss,{'kl':float(kl.detach().mean()),
        'clip_fraction':float(((ratio.detach()<1-epsilon)|(ratio.detach()>1+epsilon)).float().mean())}


def completion_logps(model, prompt_ids, completion_ids):
    if not prompt_ids or not completion_ids:raise ValueError('Empty prompt or completion')
    ids=torch.tensor([prompt_ids+completion_ids],device=model.device)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits
    # Logit at final prompt token predicts first completion token; include the sampled EOS.
    selected=logits[0,len(prompt_ids)-1:-1].float()
    targets=ids[0,len(prompt_ids):]
    return torch.log_softmax(selected,dim=-1).gather(-1,targets[:,None]).squeeze(-1)


def apply_update(model, optimizer, trajectories, rewards, *, epochs=2):
    if len(trajectories)!=len(rewards):raise ValueError('Reward/trajectory mismatch')
    advantages=group_advantages(rewards)
    stats=[]
    for _ in range(epochs):
        model.eval() # no dropout; gradients remain enabled
        optimizer.zero_grad(set_to_none=True)
        losses=[];kls=[];clips=[]
        for t,a in zip(trajectories,advantages):
            current=completion_logps(model,t['prompt_ids'],t['completion_ids'])
            old=torch.tensor(t['old_logps'],device=current.device)
            reference=torch.tensor(t['reference_logps'],device=current.device)
            loss,metrics=response_objective(current,old,reference,a.to(current.device),
                beta=SETTINGS['beta'],epsilon=SETTINGS['clip_epsilon'])
            (loss/len(trajectories)).backward()
            losses.append(float(loss.detach()));kls.append(metrics['kl']);clips.append(metrics['clip_fraction'])
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],SETTINGS['max_grad_norm'])
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradient')
        optimizer.step()
        stats.append({'loss':sum(losses)/len(losses),'reference_kl_estimate':sum(kls)/len(kls),
            'clip_fraction':sum(clips)/len(clips),'gradient_norm':float(norm)})
    return stats
