---
layout: post
title: "Understanding LLM post-training with PPO, DPO, GRPO and OPO"
description: "A learning guide to the motivation, equations and limitations of four methods for training language models from feedback."
math: true
---

Pretraining gives a language model broad capabilities by learning from large amounts of data. But predicting text is a different objective from answering a question helpfully, writing correct code, or completing a task. **Post-training** adapts a pretrained model toward those behaviors.

One approach is supervised fine-tuning, or SFT: show the model examples of the responses we want and train it to imitate them. Another is to learn from feedback. We might compare two responses and choose the better one, or let the model attempt a task and score the outcome. The InstructGPT pipeline combined demonstrations, human preferences, a learned reward model, and reinforcement learning. [InstructGPT paper](https://arxiv.org/abs/2203.02155)

This post examines four ways to learn from feedback: **PPO, DPO, GRPO, and OPO**. They are useful entry points into current post-training, but they are not an exhaustive list or four successive replacements. PPO supplies foundational ideas; DPO learns directly from preference pairs; GRPO replaces a learned critic with comparisons among attempts; OPO investigates fresh sampling and a different reward baseline. Current training tools support DPO and GRPO, including variants of their original objectives. [TRL documentation](https://huggingface.co/docs/trl/index)

To connect the algorithms to practice, the [libraries and hosted services](#libraries-and-hosted-services) comparison at the end explains which tools implement them and how much of the training workflow each service manages.

The examples here concern language models. The same questions about feedback, exploration, and optimization also matter for other foundation models, but the equations below assume an autoregressive model with computable token probabilities. Applying them to diffusion models or continuous actions requires additional choices.

Before the four methods, we need a shared vocabulary.

Imagine a coding agent fixing a bug. It reads a file, edits a function, runs tests, sees a failure, and tries again. This sequence of observations, actions, and rewards is a **trajectory**. Generating an ordinary text answer is also a sequence of decisions: at each step, the model chooses another token.

The **policy**, written $$\pi_\theta$$, is the model's distribution over those choices; $$\theta$$ denotes its trainable parameters. For a prompt $$x$$ and a response $$y$$ of length $$T$$:

$$
\log \pi_\theta(y\mid x)
= \sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t}).
$$

Here, $$y_{<t}$$ means the tokens before position $$t$$. The log probability of a response is the sum of its token log probabilities. In a tool-using trajectory, tool observations enter the context; the policy is trained on its own generated actions.

A **reward** $$R(x,y)$$ scores an outcome. For example, a coding task might return 1 when a patch passes a trusted evaluation and 0 otherwise. A learned reward model can instead score qualities such as helpfulness. The basic RL objective is:

$$
J(\theta)=\mathbb{E}_{x,\,y\sim\pi_\theta}[R(x,y)].
$$

The expectation means an average over prompts and sampled attempts. Training aims to improve that average, rather than one particular answer. A **baseline** supplies a comparison point: was an outcome better or worse than expected? The resulting difference is an **advantage**. It helps turn rewards into a learning signal, though it does not establish which action caused success.

During training, we compare the **policy being updated**, $$\pi_\theta$$, with two other versions of the model. These versions serve different purposes, even if all three begin with identical weights.

The **old policy**, $$\pi_{\mathrm{old}}$$, is the version that generated the current batch of attempts. Imagine that our coding agent produces 100 attempted bug fixes. We then use those attempts for several optimization steps. The trainable policy changes after each step, but the old policy remains the comparison point for that batch. It tells us how likely each sampled action was when we collected it. PPO compares that probability with the action's probability under the policy being updated. A large difference means we are moving away from the behavior that produced our training data. PPO's clipping discourages pushing that change too far. When we collect the next batch, the updated model becomes the new old policy. We can store the original action probabilities rather than keep a separate full copy of the old model. [PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

The **reference policy**, $$\pi_{\mathrm{ref}}$$, is a longer-term comparison point. In a typical RLHF run, it is the model after supervised fine-tuning, held fixed throughout RL training. That starting model may already have useful behavior, such as following instructions and producing readable answers. A penalty for deviating from its output distribution makes large departures costly while we optimize the reward. It does not guarantee that every capability is preserved, but it gives training a reason to retain the starting behavior. [InstructGPT paper](https://arxiv.org/abs/2203.02155)

For example, suppose training starts from model version 0. The first batch is generated by version 0, so both the old and reference policies are version 0. After several updates, we reach version 1. For the next batch, the old policy becomes version 1, while the reference remains version 0. The policy being trained then continues changing toward version 2.

These comparisons address two different scales of change. **The old policy helps control movement while learning from one batch. The reference policy helps control accumulated movement over the whole run.** Many small updates can eventually produce a large departure from the starting model, so the first comparison does not replace the second. Not every method uses both: standard DPO uses a reference policy without collecting fresh RL attempts, while the OPO recipe below omits a reference-policy penalty.

## PPO

**Motivation and problem.** Proximal Policy Optimization addresses a basic difficulty in RL: an update that looks promising on a batch of trajectories can change the policy so much that its actual performance deteriorates. PPO makes it practical to reuse a batch for several optimization steps while discouraging excessive changes. It predates LLM post-training. [PPO paper](https://arxiv.org/abs/1707.06347)

In PPO-based RLHF, a reward model scores generated answers, while a **critic** predicts future return from each partial response. These have different jobs: the reward model evaluates behavior; the critic estimates what reward to expect. A reference-policy penalty is often added to the rewards to discourage drift from the SFT model. [InstructGPT paper](https://arxiv.org/abs/2203.02155)

**Formulation.** Let $$s_t$$ be the context before token $$a_t$$. A simple advantage estimate is:

$$
\hat A_t = G_t - V_\phi(s_t),
$$

where $$G_t$$ is the observed future return and $$V_\phi$$ is the critic. For example, if expected return was 0.3 and the attempt earns 1, the advantage is 0.7. PPO implementations commonly use generalized advantage estimation, which combines predictions and rewards across steps, instead of this simple estimate.

Define the probability ratio:

$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

For compactness, define the clipped contribution:

$$
\begin{aligned}
C_\epsilon(\rho,A)=\min\bigl(&\rho A,\\
&\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A\bigr).
\end{aligned}
$$

PPO maximizes this policy objective over the collected data:

$$
J_{\mathrm{PPO}}(\theta)=\mathbb{E}_t[C_\epsilon(\rho_t(\theta),\hat A_t)].
$$

Positive advantages encourage sampled actions; negative advantages discourage them. The critic is trained separately to predict returns. [PPO paper](https://arxiv.org/abs/1707.06347)

Suppose an action has positive advantage and its probability rises from 0.10 to 0.12. Its ratio becomes 1.2. With $$\epsilon=0.2$$, this term offers no further improvement for increasing that probability. For negative advantages, the corresponding limit concerns decreasing the probability. Clipping limits the incentive, not the actual probability change: shared parameters and other examples can still move it farther. [PPO explanation](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

**Shortcomings.** The critic adds training cost and another source of estimation error. Collecting trajectories remains expensive, and clipping cannot guarantee stable updates. The policy can also lose exploration as it becomes concentrated on rewarded behavior. [PPO implementation notes](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

A practical limitation follows from the objective: improving reward only helps if the reward measures what we care about. For our coding example, a weak test suite can reward an incorrect patch. Clipping does not repair that evaluation problem.

## DPO

**Motivation and problem.** Direct Preference Optimization simplifies learning from preferences. A conventional RLHF pipeline first fits a reward model to comparisons, then trains a policy against that model. DPO expresses the preference loss directly in terms of the policy, removing the separate reward model and RL sampling loop. [DPO paper](https://arxiv.org/abs/2305.18290)

Each training example contains a prompt $$x$$, a preferred response $$y^+$$, and a rejected response $$y^-$$. For the coding agent, these could be two proposed patches, ranked by correctness and readability. Standard offline DPO trains on existing pairs; it does not execute fresh attempts at every update.

**Formulation.** First measure how much the model has changed each response's log probability relative to the reference:

$$
u_\theta(x,y)=\log\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

Then take the preferred response's margin over the rejected one:

$$
\Delta_\theta=u_\theta(x,y^+)-u_\theta(x,y^-).
$$

DPO minimizes:

$$
\mathcal L_{\mathrm{DPO}}(\theta)
=-\mathbb{E}_{(x,y^+,y^-)\sim D}
\left[\log\sigma(\beta\Delta_\theta)\right].
$$

Here, $$D$$ is the preference dataset, $$\sigma(z)=1/(1+e^{-z})$$ is the sigmoid, and $$\beta>0$$ controls the scale of the margin. In the derivation, $$\beta$$ comes from the strength of the KL penalty in the underlying reward-maximization problem. KL divergence measures how much one probability distribution differs from another. The derivation connects that objective to preference fitting under a probabilistic model of pairwise choices. [DPO paper](https://arxiv.org/abs/2305.18290)

For intuition, suppose the reference gives the two responses equal probability. If training makes the preferred response twice as likely as the rejected response, then $$\Delta_\theta=\log 2$$. Increasing this margin lowers the loss. This is a relative preference: the preferred answer's absolute probability need not increase if the rejected answer's probability falls more. [DPO trainer documentation](https://huggingface.co/docs/trl/dpo_trainer)

**Shortcomings.** The formulation reveals an exploration limitation: a fixed dataset cannot provide feedback on newly discovered attempts. Generalization is possible, but the training loop does not test those new behaviors. Poor or inconsistent preference labels can also teach the wrong behavior; preference for an answer is not necessarily evidence of correctness.

DPO is sensitive to the choice of reference, regularization, and dataset. Sequence log probabilities also make response length relevant. Current implementations expose alternatives that address noisy labels, overfitting, and length effects; “DPO training” may therefore refer to more than the original sigmoid loss. [DPO trainer documentation](https://huggingface.co/docs/trl/dpo_trainer)

## GRPO

**Motivation and problem.** Group Relative Policy Optimization asks whether we can estimate advantages without training PPO's critic. Its answer is to generate several responses to the same prompt and compare their rewards. DeepSeekMath introduced this approach to reduce the memory cost of RL training. [DeepSeekMath paper](https://arxiv.org/abs/2402.03300)

**Formulation.** Sample $$K$$ responses from $$\pi_{\mathrm{old}}$$. Let $$R_i$$ be response $$i$$'s reward, $$\bar R$$ the group mean, and $$s_R$$ the group standard deviation. With a small numerical stabilizer $$\delta$$:

$$
\hat A_i=\frac{R_i-\bar R}{s_R+\delta}.
$$

In the outcome-reward version, every generated token in response $$i$$ receives this same advantage. A compact expression of the original objective is:

$$
J_{\mathrm{GRPO}}=\mathbb{E}\left[
\frac{1}{K}\sum_{i=1}^{K}\frac{1}{T_i}
\sum_{t=1}^{T_i}\ell_{i,t}\right],
$$

$$
\ell_{i,t}=C_\epsilon(\rho_{i,t},\hat A_i)-\beta d_{i,t}.
$$

Here, $$T_i$$ is the response length, $$\rho_{i,t}$$ is PPO's token probability ratio, and $$d_{i,t}$$ estimates the KL divergence from the reference policy at that token. The expectation averages over prompts and sampled groups. The original method combines group advantages, clipping, and a reference-policy penalty. [DeepSeekMath paper](https://arxiv.org/abs/2402.03300)

For example, consider four patches with rewards $$[0,0,1,1]$$. Their mean is 0.5 and their population standard deviation is 0.5, giving advantages approximately $$[-1,-1,1,1]$$. The comparison comes from competing attempts at the same problem.

**Shortcomings.** That example also shows the limits. If all four patches fail, every advantage is zero; if all succeed, the same happens. The reward term has no relative signal in either case. More sampling costs more compute, and a single outcome advantage gives little information about which tokens mattered.

Normalization changes how examples are weighted. Dividing by each group's standard deviation can introduce difficulty-related weighting, while averaging each response by its own length can introduce length bias. Modern implementations distinguish the original formula from alternatives such as Dr. GRPO and DAPO, and may disable the KL penalty. [GRPO trainer documentation](https://huggingface.co/docs/trl/grpo_trainer)

This is an active area of development. DAPO changes sampling, clipping, and loss aggregation; GSPO uses sequence-level importance ratios and clipping and reports use in Qwen3 training. These are reasons to inspect the actual training objective when a system is described as “GRPO-based.” [DAPO paper](https://arxiv.org/abs/2503.14476), [GSPO paper](https://arxiv.org/abs/2507.18071)

## OPO

**Motivation and problem.** Here, OPO means **On-Policy RL with Optimal Reward Baseline**, the method proposed by Hao and colleagues. It studies two sources of instability: reusing samples after the policy changes, and noisy gradient estimates. It combines one update per freshly sampled batch with a baseline designed to reduce gradient variance. [OPO paper](https://arxiv.org/html/2505.23585v2)

**Formulation.** For a fixed prompt, define the response's score gradient:

$$
g(y)=\nabla_\theta\log\pi_\theta(y\mid x).
$$

The theoretical scalar baseline that minimizes the total variance of the corresponding policy-gradient estimator is:

$$
b^*(x)=\frac{\mathbb E[\|g(y)\|^2R(x,y)]}
{\mathbb E[\|g(y)\|^2]}.
$$

Those expectations use responses from the current policy. The formula weights each reward by the squared size of its response's gradient. Computing a separate gradient norm for every response is expensive, so OPO uses response length as a proxy. Two assumptions motivate this substitution. [OPO paper, Section 3.2](https://arxiv.org/html/2505.23585v2#S3.SS2)

**What is a token gradient?** For each generated token, define:
{: #opo-gradient-assumptions}

$$
g_t=\nabla_\theta\log\pi_\theta(y_t\mid x,y_{<t}).
$$

This is a vector with one component per trainable parameter. It describes which changes to the model's weights would increase the probability of the token it generated. It is not the token's embedding, and it has not yet been multiplied by a reward or advantage.

Because a response's log probability is the sum of its token log probabilities, its gradient is also a sum:

$$
g(y)=\sum_{t=1}^{T}g_t.
$$

The question is how large this sum becomes as the response gets longer.

**Assumption 1: token gradients are approximately orthogonal.** Orthogonal vectors point at right angles, so their dot product is zero. Here, the approximation is $$g_t^\top g_u\approx 0$$ for different token positions $$t$$ and $$u$$. Intuitively, the parameter changes that increase one sampled token's probability neither strongly reinforce nor strongly oppose those for another.

This matters because the squared norm of a sum expands as:

$$
\begin{aligned}
\left\|\sum_{t=1}^{T}g_t\right\|^2
&=\sum_{t=1}^{T}\|g_t\|^2\\
&\quad+2\sum_{t<u}g_t^\top g_u.
\end{aligned}
$$

The first term adds the individual squared sizes. The second captures how the vectors reinforce or cancel each other. If the combined cross terms are small, we can approximate the squared size of the sum by the first term alone.

For a two-dimensional example, consider two vectors, each of length 1:

- **Perpendicular:** $$(1,0)+(0,1)=(1,1)$$. The sum's squared norm is 2.
- **Aligned:** $$(1,0)+(1,0)=(2,0)$$. The sum's squared norm is 4.
- **Opposing:** $$(1,0)+(-1,0)=(0,0)$$. The sum's squared norm is 0.

The orthogonality assumption gives the first kind of behavior. It does not mean the words are independent or that they affect separate parameters. Many individually small cross terms can also add up, so their aggregate must be small for this approximation to work.

**Assumption 2: token gradients have comparable expected squared norms.** A squared norm measures a vector's size by squaring its components and adding them. The assumption is that each token contributes roughly the same amount on average:

$$
\mathbb E[\|g_t\|^2]\approx c.
$$

Here, $$c$$ is a common average contribution. Individual tokens can differ. To use length as a proxy across responses, we also need that typical contribution to stay roughly comparable across response lengths; longer responses should not systematically have much larger or smaller per-token gradients.

Combining these assumptions motivates:

$$
\mathbb E[\|g(y)\|^2\mid T]\approx Tc.
$$

A 200-token response would then have approximately twice the expected **squared gradient norm** of a 100-token response. Its root-mean-square gradient norm would be about $$\sqrt{2}$$ times as large, not twice as large.

**From gradient size to length weighting.** Substituting $$\|g(y)\|^2\approx cT$$ into the theoretical baseline cancels the common factor:

$$
b^*(x)\approx\frac{\mathbb E[cTR]}{\mathbb E[cT]}
=\frac{\mathbb E[TR]}{\mathbb E[T]}.
$$

This explains the length-weighted reward average. It also reveals a limit: average scaling with length alone does not prove that the substitution preserves the optimal baseline. Among responses of equal length, gradient magnitude might still correlate with reward. Length weighting ignores that relationship. The practical baseline is therefore an approximation to the variance-minimizing one, rather than a guarantee of optimality for every model or task.

For freshly sampled responses of lengths $$T_i$$, compute:

$$
\hat b(x)=\frac{\sum_{i=1}^{K}T_iR_i}{\sum_{i=1}^{K}T_i},
\qquad \hat A_i=R_i-\hat b(x).
$$

Then take an ascent step using this sample objective:

$$
\hat J_{\mathrm{OPO}}(\theta)=\frac{1}{K}\sum_{i=1}^{K}
\operatorname{sg}(\hat A_i)\log\pi_\theta(y_i\mid x).
$$

The notation $$\operatorname{sg}$$ means “stop gradient”: treat the advantage as a fixed number during the update. OPO then collects fresh responses. Its proposed recipe needs no learned critic, reference-policy penalty, or reward-standard-deviation normalization. [OPO implementation](https://verl.readthedocs.io/en/latest/algo/opo.html)

For example, suppose a failed response has 100 tokens and a successful response has 300. Their rewards are 0 and 1. The length-weighted baseline is 0.75, giving advantages of −0.75 and 0.25. This weighting changes the comparison point; it does not declare that longer answers deserve higher rewards.

**Shortcomings.** Fresh sampling for every update sacrifices repeated optimization on a collected batch. The practical baseline depends on assumptions about token gradients and estimates an expectation from a finite group. “Optimal” refers to a theoretical variance criterion, not a guarantee of the best policy or training result. The paper's mathematical-reasoning experiments do not establish universal superiority across post-training tasks. [OPO paper](https://arxiv.org/html/2505.23585v2)

There is also a finite-sample subtlety: the estimated baseline includes the very response being updated. It is therefore not independent of that response, so the usual unbiasedness argument for an action-independent baseline does not transfer unchanged. This is a consequence of the estimator, separate from the population-level derivation.

As with GRPO, equal rewards provide no relative reward signal. And removing a reference-policy penalty removes an explicit mechanism for discouraging drift. OPO is useful to study as a concrete proposal about sampling and variance reduction; adopting it still requires measuring task performance, reward reliability, and retained capabilities.

## Libraries and hosted services

An algorithm specifies how to update the model. A training library implements that update and connects it to generation, rewards, and distributed computation. A hosted service takes responsibility for some or all of the infrastructure. These are separate choices: using a service does not necessarily mean giving up control over the algorithm.

The tables below are a selected overview, with documentation checked on **October 8, 2026**. Algorithm support can differ by release, model, and training backend. Listed methods are relevant examples, not exhaustive support matrices.

**Libraries and frameworks.** These provide code you can inspect and run on your own or rented GPUs. Some emphasize accessible experiments; others emphasize distributed training or agent environments.

| Library | Relevant functionality | What distinguishes it |
| --- | --- | --- |
| **Hugging Face TRL** | SFT, DPO, GRPO, reward modeling, and related trainers. | Integrates with Transformers and provides trainer APIs and examples. A useful starting point for following the data-to-loss path. [Documentation](https://huggingface.co/docs/trl/index) |
| **verl** | PPO, GRPO, and variants such as DAPO and GSPO; a documented OPO recipe. | A framework for distributed RL experiments, including rollout and training orchestration. Of the libraries listed here, it provides a direct route to trying this post's OPO formulation. [Framework](https://github.com/verl-project/verl), [OPO recipe](https://verl.readthedocs.io/en/latest/algo/opo.html) |
| **OpenRLHF** | PPO, GRPO, RLOO, REINFORCE++, plus SFT, reward modeling, and DPO. | Uses Ray, vLLM, and DeepSpeed; supports single-turn and multi-turn agent training with synchronous or asynchronous execution. [Documentation](https://openrlhf.readthedocs.io/en/latest/) |
| **NVIDIA NeMo RL** | PPO, DPO, GRPO, DAPO, SFT, and distillation. | Supports distributed training through DTensor and Megatron Core, with recipes for large models and multimodal workloads. [Documentation](https://docs.nvidia.com/nemo/rl/latest/index.html) |
| **Axolotl** | SFT, DPO and other preference losses, GRPO, and reward modeling. | Configuration-driven training with support for full fine-tuning and adapters. Its GRPO implementation builds on TRL. [Support matrix](https://docs.axolotl.ai/docs/support-matrix.html), [GRPO guide](https://docs.axolotl.ai/docs/grpo.html) |
| **Unsloth** | Efficient SFT, DPO and other preference methods, and GRPO. | Optimized training implementations and guided notebooks for working within limited GPU memory. Feasibility still depends on model size, context length, and rollout count. [RL guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [preference training](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/preference-dpo-orpo-and-kto) |
| **OpenPipe ART** | GRPO-based training for multi-turn agents. | Organizes training around agent trajectories and rewards. Provides both local and managed backends, connecting the framework and service categories. [Overview](https://art.openpipe.ai/getting-started/about/), [backends](https://art.openpipe.ai/fundamentals/art-backend/) |
{: aria-label="Post-training libraries and frameworks" tabindex="0"}

**Hosted post-training.** There are two common interfaces. With a *managed job*, you submit data or a grader and the provider runs the training loop. With a *training API*, your code controls sampling, rewards, and updates while the provider executes the GPU work. This difference matters when the goal is to experiment with an objective such as OPO.

| Provider | What it offers | Control and availability |
| --- | --- | --- |
| **Thinking Machines Lab — Tinker** | A training API with DPO and RL cookbook recipes, including group-based advantages. | You write the loop and call sampling, forward/backward, and optimizer operations; Tinker manages the infrastructure. [Training API](https://thinkingmachines.ai/tinker/), [preferences](https://tinker-docs.thinkingmachines.ai/cookbook/preferences/), [RL](https://tinker-docs.thinkingmachines.ai/cookbook/rl/) |
| **Fireworks AI** | Managed SFT and preference training, plus a Training API with DPO and GRPO-style recipes. | Offers both provider-run jobs and user-controlled loops on hosted trainers and samplers. The choice of interface determines how much you can customize. [Training options](https://fireworks.ai/training) |
| **Together AI** | Managed SFT and DPO for supported models, with LoRA or full fine-tuning. | Handles data upload, training, and deployment. Its documented fine-tuning interface lists SFT and DPO; this is different from bringing an arbitrary RL loop to rented GPUs. [Fine-tuning overview](https://docs.together.ai/docs/fine-tuning/overview) |
| **Amazon Web Services — Bedrock** | Managed reinforcement fine-tuning using GRPO. | You provide prompts and a reward function, such as a Lambda grader or model judge. Bedrock runs the loop; model and region support are restricted to the documented catalog. [RFT documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/reinforcement-fine-tuning.html) |
| **Google Cloud — Gemini tuning** | Managed supervised and preference tuning for supported Gemini models. | Preference tuning takes chosen/rejected response pairs. It is a managed model-customization interface, rather than a general API for choosing PPO, GRPO, or OPO. [Preference tuning documentation](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/tuning/preference-tuning) |
| **Microsoft Foundry — Azure OpenAI** | Managed SFT, DPO, and reinforcement fine-tuning, depending on the model. | RFT uses supplied graders; access and regional support vary. The documentation lists GPT-5 RFT as invitation-only. [Fine-tuning overview](https://learn.microsoft.com/en-us/azure/foundry-classic/concepts/fine-tuning-overview), [RFT availability](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/reinforcement-fine-tuning) |
| **OpenPipe ART managed backend** | Hosted training and inference for ART agents. | The serverless backend runs on managed GPUs, stores checkpoints as W&B Artifacts, and deploys them through W&B Inference. You supply the agent's interaction and reward logic. [Backend documentation](https://art.openpipe.ai/fundamentals/art-backend/) |
{: aria-label="Hosted post-training providers" tabindex="0"}

**OpenAI availability note.** OpenAI has documented DPO and grader-based reinforcement fine-tuning, but its current documentation says the fine-tuning platform is winding down and is no longer open to new users. Existing users can create jobs during the remaining transition period; trained models remain available for inference until their base models are deprecated. This makes availability a deciding factor for a new project. [OpenAI fine-tuning documentation](https://developers.openai.com/api/docs/guides/reinforcement-fine-tuning)

For learning, I would start with a small DPO or GRPO example in TRL or Unsloth, where the inputs, rewards, and loss are easy to inspect. To reproduce OPO, I would start with verl's recipe. A training API such as Tinker or Fireworks is useful when I want to change the loop without operating a GPU cluster. A managed job fits a different goal: adapting a supported model with an established method and less infrastructure work. These are workflow recommendations based on the interfaces above, not a ranking of model quality.

Regardless of the interface, the data, reward function, and held-out evaluation remain part of the experiment. A service can run optimization for us; we still have to establish whether the resulting model is better at the intended task.
