# Learning LLM post-training through Python code generation

Project specification · October 8, 2026 · **Status: baseline and local evaluator-v2 replay completed; no training has run.**

This project teaches SFT, DPO, PPO, and GRPO by adapting a 4B-parameter language model to solve short Python programming problems. The model receives a problem and function signature, writes an implementation, and receives a reward from executing that implementation against tests.

The goal is to understand both the learning algorithms and the engineering behind them: data quality, token masking, reference policies, critics, rollout generation, GPU memory, evaluation, and cost. We will build a small, reproducible experiment suitable for discussing in an RL/LLM interview. It is not a claim to have reproduced frontier-scale training.

The initial resource limit is **$30 in Modal credits and no local GPU**. Treat this as a budget for smoke tests and short learning runs, not a promise that four well-tuned experiments will converge. This directory is excluded from the Jekyll website. The companion explanation is [the post-training blog post](../_posts/2026-10-08-understanding-llm-post-training.md).

### How we will work through this project

The user has confirmed the non-thinking model and wants a guided interview-preparation exercise. Work in small steps rather than implementing the whole pipeline at once. For each step, explain the concept and its purpose, work through one concrete example, implement the smallest useful piece, inspect its behavior, and discuss an interview question or tradeoff. Introduce algorithm details and hardware concepts when the current step needs them. Keep an explicit distinction between what has been planned, implemented, and measured.

The first learning step is to inspect three real dataset records and map each problem statement, reference solution, and test suite to its role. Define which information reaches the model and how success is scored. This inspection needs no GPU and does not execute downloaded code. Building the isolated evaluator follows after that contract is understood.

## 1. Decisions and scope

| Item | Initial decision |
| --- | --- |
| Task | Single-turn generation of short, deterministic Python functions |
| Starting model | `Qwen/Qwen3-4B-Instruct-2507` |
| Data source | `KodCode/KodCode-Light-RL-10K`, filtered and independently validated |
| Adaptation | BF16 frozen backbone with LoRA; full fine-tuning and quantization are later experiments |
| Experiments | Starting model → SFT; then independent DPO, PPO, and GRPO branches from the same SFT checkpoint |
| Reward | Binary correctness from a trusted test controller; no learned reward model initially |
| Compute | Modal; initially one A100 80GB, subject to a measured memory and cost check |
| Training software | Transformers Trainer + PEFT for SFT; TRL planned for DPO/GRPO; an explicitly tested PyTorch/Transformers PPO implementation |
| First deliverable | Reliable data and evaluator, followed by baseline evaluation and one SFT smoke run |
| Out of scope initially | Repository editing, multi-turn agents, web access, learned reward models, OPO, distributed training, leaderboard claims |

The model is already instruction-tuned. Our SFT stage is **additional task-specific SFT**, not the model's first post-training stage. Qwen describes this checkpoint as a non-thinking model; we will train code-only responses rather than import teacher reasoning traces. It uses an Apache-2.0 license. [Model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

### Running the first baseline

From this directory:

```sh
uv sync --python 3.12 --locked
uv run python data.py
uv run python -m unittest discover -s tests -v
# Commit and push the code and sample manifest before the paid command:
uv run modal run --detach modal_app.py
```

`data.py` downloads the pinned source to ignored `data/`, extracts only supported literal equality assertions, and prepares ten development-only problems. The committed manifest reserves their IDs and source families for development; future training/final-test preparation must exclude them. This is a deliberately narrow smoke sample, not a representative benchmark or the full split pipeline. Review exclusions live in `manifests/audit-exclusions.json`. Unsupported tests are rejected entirely, not partially scored. The system prompt deliberately requests code only even when source wording also asks for an explanation.

`modal_app.py` first runs evaluator probes and validates all ten reference solutions in isolated, network-blocked CPU sandboxes. The trusted controller holds expected outputs. Each candidate gets a 512 MiB sandbox with no secrets or shared mounts; its child process has a 2-second CPU limit, 10-second wall limit, 384 MiB address-space limit, and 64 KiB file/output limits. These measured-pilot settings replace the earlier proposed per-call limits. A sandbox is terminated in `finally`, with a 60-second platform lifetime as a fallback. Infrastructure failures get one retry, then fail the run as unscored.

Only prompts reach the GPU. The job generates one greedy completion per problem, first for one canary and then for the remaining nine. Incorrect model answers are legitimate baseline observations and do not block the remaining samples. Generated programs never execute directly on the GPU worker or the host computer: cloud evaluation uses Modal Sandboxes, and local replay uses restricted Docker containers. Model weights download during CPU image preparation. The GPU function uses one A100 80GB, max concurrency one, no automatic retries, two calls of at most 600 seconds each, and a two-second idle window. The coordinator has a 30-minute timeout; no serving endpoint is deployed. Allow up to $3 for initial image/setup and this baseline within the overall $30 budget; runtime limits bound work but are not a provider-enforced dollar cap. Check billed usage before another run.

The local launch manifest records the Modal app/call IDs. Intermediate and final reports persist in the `coding-post-training-results` Modal Volume, and a successful run also saves its report under ignored `runs/`. Reports include the code revision, model/data revisions, sample hash, sandbox IDs, per-case counts, generated code, token counts, latency, package versions, and peak GPU allocation/reservation. No optimizer or adapter is created: this measures the untouched model. Retrieve a report after a local disconnect with:

```sh
uv run modal volume get coding-post-training-results RUN_ID.json runs/RUN_ID.json
```

The baseline uses the same `uv.lock` for local checks and cloud images, with the `inference` extra enabled only where required. Dataset download and local contract checks are implemented; full data curation, robust held-out evaluation, and all training stages remain later milestones.

### First observed baseline — October 8, 2026

The [completed Modal run](https://modal.com/apps/cpbookbuilder/main/ap-FSh074g06OilNM6dM9r9qy) executed commit `62ae49b8` after a missing-module image packaging error was corrected and covered by an isolated import regression test. Both cloud apps are stopped. The [aggregate report](reports/baseline-2026-10-08.json) records the revisions, sample hash, metrics, and per-problem statuses; complete outputs remain in ignored `runs/` and the Modal results Volume.

All six evaluator probes and all ten reference programs passed. The model completed all ten responses within the token allowance, but **every response included Markdown code fences**. Under the predeclared raw-Python contract, every response therefore received `syntax_error` and reward 0. The code inside the fences was not executed. This result establishes a formatting failure, not that the model cannot solve these programming problems.

The run generated 1,375 tokens in about 42.1 seconds, with roughly 49.3 seconds inside GPU function calls and peak allocated GPU memory of 8.16 GB decimal (about 7.60 GiB). These are inference measurements, not training-memory estimates. The active GPU-function resource estimate is about $0.037; it excludes image builds, startup/idle time, evaluator sandboxes, storage, and the unsuccessful initial image build, so it is not the final bill.

This original report is preserved unchanged. The separately versioned replay below distinguishes output-format compliance from functional correctness; neither report is a post-training result.

### Local replay with evaluator v2

At the user's request, `python-output-v2` accepts raw Python unchanged or one complete, lowercase `python` Markdown code block spanning the entire response (surrounding whitespace allowed). It removes only the two fence lines, without rewriting code. Empty responses, other fence labels, unlabeled fences, multiple/nested blocks, surrounding explanations, malformed fences, and invalid Python are rejected before execution. The prompt still requests raw Python, and compliance with that preference is measured separately.

Re-evaluating the exact saved responses gave:

| Measure | Result |
| --- | ---: |
| Original strict end-to-end accuracy | 0/10 |
| Raw-format compliance | 0/10 |
| Functional accuracy after extraction | **10/10** |
| Individual cases passed after extraction | **66/66** |
| Reference solutions revalidated locally | 10/10 |
| Additional generations / Modal calls | 0 / 0 |

The [v2 replay report](reports/baseline-2026-10-08-replay-v2.json) records the original response-file hash, per-response hashes, evaluator source hashes, immutable Docker image identity, and case counts. The original responses, tests, model weights, and baseline report were not changed. These ten development problems are a pipeline smoke test, so 100% does not establish general coding performance. Their high success rate suggests we should inspect task difficulty before investing in RL; greedy success alone does not establish whether sampled GRPO groups would have reward variance.

To replay a saved run locally, with Docker running:

```sh
docker pull python:3.12.10-slim-bookworm
RUN_DOCKER_TESTS=1 uv run python -m unittest discover -s tests -v
uv run python reevaluate.py \
  --run runs/baseline-20261008T222157Z-62ae49b8.json \
  --output runs/replay-v2.json
```

The replay refuses to overwrite an existing report or a source artifact, checks that the responses match the prepared sample, revalidates references, and records partial progress if infrastructure fails. It does not import the Modal app, load a model, or make cloud calls. Each Docker container runs as a non-root user with no network, host mounts, or forwarded credentials; a read-only filesystem, capped temporary filesystem, and memory/process/CPU/time/output limits bound execution. Containers are removed after each candidate, including failures. The local and Modal paths share extraction, the execution driver/supervisor, and trusted result comparison; only the isolation backend differs. Future cloud runs also report evaluator version and raw-format compliance.

## 2. Problem definition

**Input:** an English specification, a Python function signature, and any public examples supplied with the problem.

**Output:** Python source defining the required function. Imports from an approved standard-library subset and helper functions are allowed. The prompt requests raw code without explanations or Markdown. Evaluator v2 additionally accepts one complete Python code block, while tracking raw-format compliance separately. Interactive input, file access, network access, and test-running code are outside the task.

**Success:** the response finishes within the generation limit, satisfies the output contract, and passes every private test case for that problem within execution limits. Passing finite tests is evidence of correctness, not a proof.

An illustrative problem, written for this specification rather than copied from the dataset:

```text
Implement longest_run(values: list[int]) -> int.
Return the largest number of consecutive equal values.
For an empty list, return 0.

Example: [2, 2, 3, 3, 3, 2] -> 3
Return only Python source defining longest_run.
```

Example private cases would include an empty list, one element, all equal elements, alternating elements, and a longest run at either end. They are inputs to the evaluator, not additional model context.

Initially accept only deterministic, self-contained functions with arguments and results representable by our versioned test protocol. Start with JSON-compatible values; reject tasks requiring custom classes, generators, filesystem state, randomness, external packages, interactive I/O, or unsupported fixtures. Avoid tasks whose central requirement, such as asymptotic complexity, cannot be judged by the chosen tests. These restrictions keep execution and scoring understandable while retaining real code-generation failures.

### The RL interpretation

- **State:** the problem prompt plus code tokens generated so far.
- **Action:** the next token.
- **Policy:** the language model's distribution over that next token.
- **Trajectory:** the prompt, generated tokens, stored token log-probabilities, termination reason, and execution outcome.
- **Environment:** generation termination followed by the isolated code evaluator.
- **Task reward:** 1 for a passing completed program, otherwise 0.

There is no repair turn in version 1. The model does not see failing assertions or test feedback in its next prompt. PPO and GRPO learn from the scalar outcome through their gradient updates.

## 3. Dataset, provenance, and splits

### Source snapshot

[KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K) contains 10,000 examples in a single `train` split. The relevant fields are `question`, `solution`, `test`, `question_id`, and `test_info`; metadata and teacher success/difficulty fields provide additional provenance. It also contains conversation/reasoning fields. We will use validated `solution` code as the SFT target, not those conversations. The dataset is licensed **CC BY-NC 4.0**: preserve attribution and provenance, and keep this learning project noncommercial. The model's different license does not erase the dataset's restrictions.

These immutable revisions were resolved while preparing this specification:

| Asset | Repository revision |
| --- | --- |
| Model and tokenizer | [`cdbee75f17c01a7cc42f958dc650907174af0554`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554) |
| Dataset | [`dcf78a8bbba9a613b596ce993c4921a38687dfcc`](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K/tree/dcf78a8bbba9a613b596ce993c4921a38687dfcc) |

Download by revision, retain the source license/attribution, and record downloaded file hashes. Do not commit the dataset or copied solutions to this website repository. Any later public release of derived data or weights needs its own license review.

### Preparation pipeline

1. Preserve raw records and source IDs. Validate schema and reject missing or conflicting signatures.
2. Apply the task restrictions above. Record a reason for every rejection; do not silently rewrite ambiguous problems.
3. Convert supported test assertions into a trusted, declarative case format. Reject unsupported pytest behavior rather than pretend to support arbitrary tests.
4. Execute the reference solution in isolation against the converted cases. A teacher correctness label is not sufficient. Quarantine failing, nondeterministic, or inconsistent records.
5. Deduplicate normalized prompts and solutions; inspect near-duplicate problem families. Use source seed IDs where available to keep related generated problems together. Report residual overlap risk: metadata and similarity checks are imperfect.
6. Assign whole problem families to splits with a fixed seed (`42`) and save immutable manifests. Do this before model-based difficulty filtering or preference generation.
7. Audit a sample manually for specification/test agreement and meaningful edge cases. Add independently checked cases to a small robustness subset before training.

Proposed **maximum first-study target**, contingent on enough eligible examples and measured cost:

| Partition | Target problems | Permitted use |
| --- | ---: | --- |
| SFT train | 1,000 | Demonstration code for supervised learning |
| Alignment train | 500 | Candidate generation for DPO and online rollouts for PPO/GRPO |
| Development | 200 | Debugging, difficulty diagnosis, checkpoint selection, limited tuning |
| Final test | 300 | Evaluation after configurations and checkpoints are fixed |

The four partitions must be disjoint by problem family. If eligibility leaves fewer examples, reduce these targets and report actual counts. Do not fill shortages with duplicate tasks.

The first smoke run uses nested subsets of **64 SFT problems, 32 alignment problems, and 32 development problems**. Its purpose is to validate the complete pipeline. Scale beyond these counts only after measuring throughput and remaining credits. Do not repeatedly inspect final-test scores during development. If final evaluation must be smaller, preselect a fixed random subset before training and report its size.

“Private tests” has two meanings that must stay separate:

- Alignment training tests are hidden from the prompt, but their rewards influence training. They are part of the training signal.
- Final-test problems and their cases are held out from both training and model selection.

A random split of a public dataset does not establish that the original Qwen model never saw those problems. Report this contamination limitation; use newly authored, independently checked variations as a small additional robustness check.

### Internal records

Keep three logical artifacts separate:

```text
Problem: id, source_revision, source_id, family_id, split,
         question, entry_point, prompt_version, token_counts
Target:  problem_id, validated_solution, validation_report
Cases:   problem_id, case_suite_version, cases, comparison_rules
```

The model receives only the rendered prompt. Training and evaluation orchestration use `problem_id` to find cases. Never concatenate reference solutions, teacher metadata, or private cases into prompts.

Derived SFT records contain `prompt` and `completion`. DPO records contain the same `prompt`, a sampled `chosen` completion, and a sampled `rejected` completion, plus provenance stored outside model input. PPO/GRPO records begin with prompts; completions and rewards are generated during training.

## 4. Evaluator and reward contract

The evaluator is part of the experiment, not an incidental helper. A flawed verifier teaches the model to exploit that flaw.

Proposed execution design:

1. Parse with the versioned `python-output-v2` rule above: accept raw Python or one complete Python Markdown block. Remove fence delimiters only, report raw-format compliance separately, and reject unsupported formats. Do not repair model code before scoring. The original baseline used the earlier raw-only rule and remains unchanged.
2. Run each candidate in a fresh isolated CPU sandbox with network disabled, no secrets, no dataset/checkpoint mounts, and bounded resources. Python `exec`, an AST check, or a subprocess alone is not the isolation boundary. [Modal Sandboxes](https://modal.com/docs/guide/sandboxes)
3. Keep private assertions, expected outputs, and scoring in a trusted controller outside the candidate's process. Send test inputs through a bounded JSON protocol and compare returned values in the controller. Do not deserialize candidate-controlled pickle objects.
4. Start with a proposed 1 CPU, 512 MiB memory, 2-second limit per call, 10-second total execution limit per candidate, and 64 KiB output cap. Measure sandbox startup separately. Calibrate these limits on reference solutions, then freeze them for all methods.
5. A valid completed candidate receives `1` only if all cases pass. Syntax errors, exceptions, wrong results, contract violations, resource-limit failures, and generation truncation receive `0`.
6. A platform outage or failed sandbox launch is an **infrastructure error**, not an incorrect answer. Retry at most once, then mark the sample unscored and stop/repair the affected batch instead of silently changing its reward.

The protocol must define type checking, floating-point tolerances where permitted, mutation behavior, and supported values. Initially reject problems needing semantics the protocol cannot preserve. Do not put pytest files in the same process as untrusted candidate code and call their contents secret.

Use identical task rewards for PPO and GRPO. DPO preferences are derived from the same verifier. Log fraction of cases passed for diagnosis, but do not use it as a reward initially: otherwise a method comparison also becomes a reward-design comparison. KL regularization is recorded separately from task correctness.

Before GPU training, verify the evaluator on known-correct code, wrong code, an infinite loop, excessive output, a crash, and attempted file/network access. Check that candidate output cannot directly set the reward. Any critical isolation or scoring failure blocks model-generated execution.

## 5. Experiment structure

```text
Qwen3-4B-Instruct-2507 ── baseline evaluation
          │
          └── task SFT ── SFT evaluation
                    │
                    ├── DPO  ── evaluation
                    ├── PPO  ── evaluation
                    └── GRPO ── evaluation
```

DPO, PPO, and GRPO each start independently from the **identical SFT checkpoint**. They are not a sequential SFT → DPO → PPO → GRPO pipeline.

For an uncomplicated reference-policy implementation, export the SFT adapter merged into BF16 backbone weights, verify logits against the unmerged model within a documented numerical tolerance, and freeze that export as the common starting point. Add a fresh, initially zero-effect LoRA adapter for each alignment run. This makes “disable the alignment adapter” refer to the SFT policy. Disabling an adapter on the original pretrained checkpoint would instead produce the wrong reference.

Use the same prompt template, tokenizer, EOS handling, actor adapter configuration, and evaluation settings across branches. Fix the seed for the pilot and retain all sampled outputs. One seed is exploratory evidence; multiple seeds are a later budget item.

### SFT: learn the task's response format and solution patterns

Train on validated reference implementations. Minimize next-token cross-entropy over **assistant completion tokens only**, including the intended EOS token. Mask system/user tokens and padding. Verify the mask on rendered examples rather than assume the chat template marks them correctly.

SFT gives us a controlled starting policy and teaches the model the desired output contract. It cannot distinguish two plausible solutions unless the demonstrations convey that distinction, and a lower training loss does not establish improved code correctness.

Report baseline versus SFT correctness, formatting failures, loss, and response lengths. Save one SFT checkpoint selected using development data and use it for all alignment methods.

### DPO: learn from sampled correct/incorrect pairs

Generate four candidates per alignment-training prompt from the frozen SFT model. A passing completion can be `chosen`; a failing completion can be `rejected`. Retain at most one pair per prompt initially, selected deterministically from eligible candidates. Skip all-pass and all-fail groups and report their frequencies. Do not invent incorrect programs to fill a quota.

The standard objective is:

$$
L_{\mathrm{DPO}}=-\mathbb{E}\log\sigma\left(\beta_D\left[
\log\frac{\pi_\theta(y^+\mid x)}{\pi_{\mathrm{ref}}(y^+\mid x)}-
\log\frac{\pi_\theta(y^-\mid x)}{\pi_{\mathrm{ref}}(y^-\mid x)}
\right]\right).
$$

Each response log-probability is a **sum over completion tokens**, with prompt/padding excluded. Start with ordinary sigmoid DPO, not an undocumented length-normalized variant. Cache reference log-probabilities only when model, tokenizer, template, tokenization, and masking hashes match.

DPO avoids online sampling during updates and needs no critic, but collecting pairs still costs inference and execution. Its usable prompts are biased toward mixed-success groups; report that coverage instead of claiming identical effective training data to PPO/GRPO. [DPO paper](https://arxiv.org/abs/2305.18290)

### PPO: learn from current rollouts with a value model

At each rollout batch, sample from a fixed behavior policy `pi_old`. Store its selected-token log-probabilities and old value predictions without gradients. Keep `pi_ref` frozen at SFT for the whole run; refresh `pi_old` only between rollout batches.

The **critic is the value model**: it predicts expected remaining return for each token prefix. It is not a reward model and does not run the tests. Our proposed critic has its own frozen copy of the SFT backbone, trainable LoRA adapter, and trainable scalar value head. Start with a short value-only warm-up on initial rollouts and include that compute in the PPO budget.

Use terminal correctness plus token-level KL shaping:

$$
r_t=\mathbf{1}[t=T]R(x,y)-\beta_{KL}
\left(\log\pi_{\mathrm{old}}(y_t\mid s_t)-\log\pi_{\mathrm{ref}}(y_t\mid s_t)\right).
$$

The sampled log-ratio is a KL estimator and can be negative for an individual token. With old value estimates, compute generalized advantage estimation (GAE):

$$
\delta_t=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t),
\qquad
\hat A_t=\sum_{l\geq0}(\gamma\lambda)^l\delta_{t+l}.
$$

Then minimize the negative clipped surrogate and a separate value regression loss:

$$
L_{\mathrm{policy}}=-\mathbb E\left[
\min\left(\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\right)
\right],\qquad
\rho_t=\frac{\pi_\theta(y_t\mid s_t)}{\pi_{\mathrm{old}}(y_t\mid s_t)}.
$$

Use detached targets and advantages. Set the terminal bootstrap value to zero; our task defines both EOS and exhausting the token allowance as terminal, with truncation a task failure. Include EOS in the response mask and exclude padding. Map the value before the first generated action to the final prompt position, avoiding a one-token shift.

Initially use `gamma=1`, `lambda=0.95`, clip width `0.2`, and two optimization passes per fresh rollout batch. Use masked advantage whitening across the rollout batch, and average valid token losses within a response before averaging responses. Record these choices because alternative reductions change weighting. Measure value loss, explained variance, clipping, and divergence in addition to task reward. [PPO paper](https://arxiv.org/abs/1707.06347)

### GRPO: compare answers to the same problem without a critic

Sample a group of four completions per prompt and compute:

$$
\hat A_i=\frac{R_i-\operatorname{mean}(R_1,\ldots,R_G)}
{\operatorname{std}(R_1,\ldots,R_G)+10^{-6}},\qquad G=4.
$$

Use a clipped token policy-ratio objective with these group advantages and explicit reference KL regularization. Start with the original per-response loss reduction: average token terms within each completion, then average completions. Explicitly configure that variant and verify it against a small reference calculation; library defaults can differ. Document the standard-deviation convention too. [GRPO paper](https://arxiv.org/abs/2402.03300), [TRL GRPO documentation](https://huggingface.co/docs/trl/grpo_trainer)

Groups with identical rewards have zero task advantage. Log the all-pass/all-fail fractions; those groups may still contribute KL regularization. Do not silently drop them or resample until success. If nearly all groups have zero advantage, investigate task difficulty using training/development data before spending more on updates.

GRPO saves critic memory but generates multiple answers per problem. Compare measured generation cost and useful learning signal, not just the number of model copies. Group standard-deviation scaling and response-length weighting are explicit candidates for later ablations.

## 6. Software and reproducibility

Proposed implementation uses Python, PyTorch, Transformers, PEFT, Datasets, TRL, Modal, and pytest for our own trusted tests. Use Transformers generation initially; add vLLM only after a throughput profile shows its value. Keep training dependencies in this subdirectory, separate from the blog's Ruby and Node dependencies.

**PPO needs an explicit implementation choice.** TRL removed `PPOTrainer` from its main branch in September 2026. OpenRLHF supports PPO, but its current Ray/vLLM path documents no LoRA support. For this single-GPU adapter project, implement a narrow synchronous PPO loop using supported model/adapter libraries, with independently checked loss, GAE, and masking tests. This costs engineering time and must be treated as an educational implementation, not an already validated production trainer. Do not install an obsolete TRL release just to make an old tutorial run. [TRL removal PR](https://github.com/huggingface/trl/pull/7020), [OpenRLHF RL guide](https://openrlhf.readthedocs.io/en/latest/agent_training.html)

Before training, pin a compatible dependency set and container digest after verifying model load, forward/backward, adapter save/reload, and generation in Modal. Exact package versions are **not yet validated**. Save the resolved configuration; do not rely on undocumented trainer defaults. OpenRLHF or another distributed framework is a later scaling exercise with its own memory and compatibility plan.

Proposed first settings, subject to the smoke test:

| Setting | Starting value / rule |
| --- | --- |
| Precision | BF16 backbone; stable loss/log-probability accumulation in FP32 |
| Actor adaptation | LoRA rank 16, alpha 32, dropout 0; explicitly named attention and MLP linear projections |
| Sequence budget | At most 1,024 prompt tokens and 512 completion tokens; filter overlong prompts rather than truncate specifications |
| Long SFT targets | Reject from the initial subset rather than cut off valid code |
| Microbatch | 1 response, gradient accumulation to fit; preserve full GRPO groups across accumulated work |
| Effective SFT/DPO batch | 16 examples/pairs initially |
| Online rollout batch | 8 prompts × 4 responses = 32 trajectories for both PPO and GRPO |
| Sampling | Temperature 1, top-p 1, no top-k cutoff or repetition penalty; sample from the logged policy distribution |
| Optimization | AdamW; actor LR SFT `1e-4`, DPO `5e-5`, PPO/GRPO `1e-5`; critic LR `1e-4`; gradient clipping at 1 |
| Regularization | DPO beta `0.1`; PPO/GRPO reference KL coefficient initially `0.02`; record the distinct meanings of these parameters |
| Online reuse | Two optimization passes per rollout batch; then regenerate under the updated policy |
| Memory controls | Gradient checkpointing; sequential generation/updates; no concurrent duplicate inference server initially |
| Pilot duration | First 1–2 optimizer steps, then at most 20 online rollout batches or the stage cost limit, whichever arrives first |

Sampling settings matter to PPO's denominator. If temperature or token filtering changes, the implementation must account for the actual behavior distribution rather than store incompatible raw-model log-probabilities. Disable dropout during rollout and log-probability recomputation. Verify model/tokenizer EOS and padding IDs; never assume their values.

Minimum algorithm correctness checks before a meaningful run:

- Hand-calculated DPO loss and preference direction, PPO clipping for positive/negative advantages, and GAE with explicit terminal masks.
- Ratio near 1 before the first update, frozen reference weights, and no gradients through old log-probabilities, rewards, advantages, or targets.
- Completion masks, shifted labels/value positions, EOS handling, padding invariance, and no prompt loss leakage.
- Constant-reward GRPO groups give zero task advantage; group membership survives batching; loss reduction matches the chosen formula.
- Save/reload preserves outputs within the chosen tolerance and restores optimizer, RNG, scheduler, and progress state for resume.

These are checks of mathematical and data contracts, not merely checks that a trainer command exits successfully.

## 7. Hardware and the $30 budget

Four billion BF16 parameters alone occupy roughly **8 GB decimal**. Gradients, activations, KV cache, optimizer state, temporary buffers, and extra model copies all add memory. A conventional full Adam training allocation can reach roughly 16 bytes per parameter before activations, depending on precision/state choices. LoRA reduces trainable state but does not remove the backbone or all activation memory.

With separate BF16 actor, reference, and critic backbones, PPO starts around 24 GB of parameter storage alone. Stored old log-probabilities avoid needing a fourth permanent `pi_old` model. Sharing a frozen SFT reference through adapter disabling can reduce copies once verified. Measure actual peak allocated/reserved memory and device usage; these estimates are not fit guarantees.

An A100 80GB is the initial low-complexity choice because it leaves headroom to inspect these costs. An L4/L40S can be a later cost experiment for simpler stages after profiling. Do not change model size below the agreed >1B scope to hide a memory problem.

Listed Modal GPU rates checked for this spec:

| GPU | GPU-only USD/hour |
| --- | ---: |
| L4 | 0.7992 |
| L40S | 1.9512 |
| A100 80GB | 2.4984 |

Modal also bills CPU and memory. At listed regular Function rates, an A100 80GB with 2 CPU cores and 16 GiB host memory is about **$2.72/hour**, before evaluator Sandboxes, storage, or other charges. Sandbox CPU/memory rates differ. This makes $30 roughly 11 hours of that one Function configuration before other costs, not 11 hours of useful optimizer work. Recheck rates, available credits, and the real resource configuration before launching. [Modal pricing](https://modal.com/pricing)

Proposed spending envelopes—not runtime forecasts:

| Stage | Maximum initial allocation |
| --- | ---: |
| Environment, memory, and evaluator smoke checks | $2 |
| Baseline and SFT pilot | $5 |
| Preference generation and DPO pilot | $4 |
| GRPO pilot | $4 |
| PPO pilot, including critic warm-up | $6 |
| Final evaluation and results | $4 |
| Uncommitted reserve | $5 |
| **Total** | **$30** |

Before each paid stage, project cost from observed seconds per batch, generation tokens, evaluator work, startup, and requested resources. Stop at the smaller of the stage allowance or remaining project allowance. Implement bounded invocations, concurrency limits, a spend ledger, and checkpoint-on-stop; compare estimates with Modal's usage dashboard. A local estimate is not a provider-enforced spending cap, and billing may lag.

Do local data inspection and loss tests first. Cache model/data downloads, build images before allocating GPUs where possible, terminate idle GPU workers, and do not leave an inference service running between sessions. Failed GPU starts and evaluator work count toward the budget. If all four useful pilots do not fit, retain partial results and report the cost constraint rather than present them as a completed comparison.

## 8. Evaluation and evidence

Evaluate five checkpoints: starting Qwen, SFT, SFT+DPO, SFT+PPO, and SFT+GRPO. Use the same final problems, prompt version, evaluator, generation cap, and decoding settings.

**Primary measure:** fraction of final-test problems solved by one greedy completion. Label it greedy single-attempt accuracy; do not silently equate it with a sampled pass@1 estimate.

**Optional, if affordable:** four independent samples per problem at the same fixed sampling settings. Report sampled pass@1 as the mean success fraction and empirical pass@4 as the fraction of problems with at least one passing candidate. Keep these separate from greedy results and count their additional generation cost.

Also report:

- Syntax/contract failures, exceptions, wrong answers, timeouts, generation truncations, and unscored infrastructure failures.
- Generated-token length distribution, EOS rate, reward versus length, and robustness-case accuracy.
- PPO/GRPO task reward, reference divergence estimator, clipping, gradient norms, and updates per rollout; PPO critic diagnostics; GRPO zero-variance groups.
- DPO pair count, mixed-group coverage, chosen/rejected lengths, and preference margin alongside execution correctness.
- Peak GPU memory, training and generation tokens/second measured separately, evaluator latency, end-to-end GPU time, and total dollars including data generation.

Show per-problem paired results and confidence intervals appropriate to the number of evaluated problems. Bootstrap at the problem level if using multiple completions per problem. A few percentage points on a tiny evaluation set may be noise. Single-seed pilot results cannot establish a general ranking of algorithms.

PPO/GRPO can visit different response distributions; DPO uses a fixed, filtered dataset. Report prompt exposure, sampled completions, trained tokens, and dollars rather than claim that equal optimizer steps mean equal compute. A later controlled study can match dollar or rollout-token budgets.

Do not set “all methods must improve” as a success condition. A reproducible negative result, with a diagnosed reason, is a valid learning outcome. The project succeeds when we can explain what ran, what changed, what it cost, and which conclusions the evidence supports.

## 9. Planned files and artifacts

The baseline currently uses `data.py`, `evaluator.py`, `modal_app.py`, contract tests, and a shared dependency lock. The following remains the proposed broader training layout:

```text
coding-post-training/
  README.md                 # This specification
  pyproject.toml + lockfile  # Reproducible training environment
  configs/                  # Explicit model/data/method settings
  src/                      # Data, prompts, evaluator client, training, evaluation
  tests/                    # Evaluator and algorithm contract checks
  modal_app.py              # Bounded GPU jobs and isolated CPU execution
  manifests/                # Split IDs, hashes, provenance; no copied solutions
  reports/                  # Small reviewed results and figures
  data/                     # Ignored source/processed records and rollouts
  checkpoints/              # Ignored weights and optimizer state
  runs/                     # Ignored detailed logs and local artifacts
```

Each run must retain: run ID; repository commit and dirty diff if any; source revisions; split hashes; container and package versions; GPU/resource settings; prompt and evaluator versions; random seeds; resolved trainer settings; trainable parameter counts; checkpoint hashes; token/sample/step counts; reward and failure logs; timings; and cost estimate versus billed usage. Keep large artifacts on a Modal Volume or other explicitly selected storage, with a small local manifest.

## 10. Implementation milestones

1. **Data and evaluator on CPU.** Inspect the source, implement eligibility and grouping, freeze splits, validate reference solutions, and verify isolation/scoring failure cases. Exit with an audit report and reproducible manifests.
2. **GPU compatibility and baseline.** Resolve dependencies, load the 4B model, verify generation and a backward pass, measure memory/throughput, and evaluate a development subset. Exit with a realistic spend forecast.
3. **SFT.** Validate masks, run a tiny overfit check, run the bounded SFT pilot, and verify the exported common checkpoint. Exit with baseline/SFT comparisons.
4. **DPO.** Collect real candidate pairs, report usable coverage, verify loss arithmetic, train the short branch, and evaluate on development data.
5. **GRPO.** Connect online generation to the same evaluator, verify grouping/normalization, inspect zero-advantage groups, and run the bounded pilot.
6. **PPO.** Validate GAE/clipping/value alignment, warm up the critic, run the bounded pilot, and compare critic cost and diagnostics with GRPO.
7. **Freeze and evaluate.** Select checkpoints using development data, run the untouched final evaluation within its budget, and write an evidence-based report with failures and limitations.

Keep smoke checks tiny. Increase sample counts or duration only when the previous stage's correctness, memory, and cost measurements justify it.

## 11. Questions the finished project should help us answer

- Why do we need both an old policy and an SFT reference in PPO? When can stored log-probabilities replace a model copy?
- What does the critic learn with terminal execution rewards, and how do GAE and value error affect the actor?
- Why can DPO improve preference margins without improving held-out code correctness?
- What happens to GRPO when every sample in a group fails, or every sample passes?
- How do EOS, padding masks, sampling temperature, and loss normalization alter the actual objective?
- How can a verifier be exploited, and what evidence distinguishes reward improvement from generalization?
- Where does GPU memory go, and when is generation or CPU evaluation the throughput bottleneck?
- Which change would help next: better tests, different task difficulty, a longer context, more samples, faster generation, or another GPU?

For a later scaling study, add vLLM weight synchronization, compare full fine-tuning with LoRA, and investigate DDP/FSDP/ZeRO and asynchronous rollout staleness. This pilot gives measurements to reason from; it does not supply hands-on multi-node experience by itself.

## 12. Items to resolve during implementation

The task, model family, branch structure, and budget are decided. The remaining empirical decisions are the eligible dataset size, supported case protocol, exact dependency lock, achievable microbatch/context size, critic warm-up duration, stopping thresholds, and the number of final evaluation problems affordable within the remaining credits. Record each resolution here and in configuration files before the corresponding experiment.

**Next learning checkpoint:** review the baseline's correctness, failure categories, token lengths, and GPU measurements together before designing the SFT run.

## Difficulty-discovery experiment: 100 problems × 4 answers

The ten-problem greedy smoke replay scored 10/10. Before training, we therefore
look for a measurable capability gap using `discovery_data.py` and `discovery.py`.
This experiment samples four answers per prompt at temperature 1, top-p 1, top-k
0, with a 512-token completion limit. It is a stochastic discovery experiment,
not a repeat of the greedy baseline and not final-test evaluation.

**“Supported by our test parser”** means all tests can be converted into explicit
JSON inputs and expected outputs without executing dataset test code. For example,
`assert add(2, 3) == 5` is supported. Loops, helper calls, fixtures, keyword
arguments, approximate comparisons, floats, and tuple-valued cases are currently
unsupported. We reject the entire problem if any test is unsupported. This is an
engineering limitation of our evaluator, not a statement about model ability.
Parser acceptance also does not establish that the expected answers are correct.

Preparation groups all 10,000 source records transitively by shared seed IDs,
normalized prompt text, normalized solution AST, and manually identified links.
Integer and text seed IDs are normalized together conservatively, which can
merge unrelated source families. Existing smoke families remain development-only.
Hash-assigned pools use nominal 50% SFT / 25% alignment / 10% development / 15%
final-test allocation by family, so example counts need not follow those ratios.
These are frozen source pools, not fully validated training datasets or a claim
that all semantic duplicates have been found. The earlier 1,000/500/200/300 study
sizes remain targets for later validated subsets.

From eligible training/development pools, select 50 broadly, then 50 additional
medium/hard examples in seeded hash order, with one example per family. Exclude
smoke families and final-test families. Dataset difficulty labels are teacher
metadata, not measured Qwen performance. Validate every selected reference in
restricted local Docker. Audit prompts against tests before inference; record
rejections and additional family links in `manifests/discovery-audit.json`.
Reference agreement alone cannot detect a shared mistake in the solution and tests.

```bash
cd coding-post-training
uv run --extra inference python discovery_data.py
RUN_DOCKER_TESTS=1 uv run python -m unittest discover -s tests -v
# Commit and push the prepared manifests and runner before this paid step:
uv run python discovery.py
```

Frozen manifests refuse silent overwrites. Revisions after publication need new
versioned manifests. Source data and full responses remain under ignored `data/`
and `runs/`. Only prompts reach the GPU; generated code runs exclusively in local
Docker. Generation shares the baseline's model loader, prompt, tokenizer, and EOS
handling, with four sequences batched per prompt. Each prompt gets a recorded
seed independent of batch boundaries. Different hardware/library versions can
still change sampled answers.

The runner first generates, retrieves, saves, and locally grades one four-answer
canary, including container cleanup. Then it submits batches of at most nine
prompts, with one A100-80GB worker and no automatic retries. Each cloud call has a
240-second execution timeout and a 300-second startup timeout. Raw completions
are checkpointed to the Modal Volume after every problem and saved locally after
each batch, before grading. The GPU app ends before grading the remaining answers.
A failed run never turns infrastructure errors into zero-reward examples.

The initial allowance is **$3**. The application reserves $0.25 for setup and uses
$0.00075572/second for A100-80GB + 2 CPU cores + 16 GiB RAM at
[Modal's published rates](https://modal.com/pricing), checked October 8, 2026.
Before each batch it reserves the next call's maximum startup/execution/idle time;
it stops with partial results if that would exceed the allowance. Accounted time
is remote wall time plus a two-second idle allowance per call. These are
conservative estimates, not a provider-enforced dollar cap or a billed total.

Report separate broad and medium/hard results. A prompt with 1–3 correct answers
out of four is a candidate for preference pairs and relative-reward learning;
4/4 offers no within-group binary reward contrast in this sample, and 0/4 needs
failure analysis. Neither four successes nor four failures proves an underlying
success probability of one or zero. Only alignment-pool examples may become
DPO/PPO/GRPO training data; development examples remain development-only. Audit
apparent failures before using their labels, and retain format and truncation
metrics separately from functional correctness.

### Completed discovery run — October 9, 2026

Run [`discovery-20261009T070510Z-3bb672d3`](https://modal.com/apps/cpbookbuilder/main/ap-zygxSP0zswo38PEDL1UOe8)
used committed revision `3bb672d3`, 100 audited problems, and 400 stochastic
completions. All 100 reference programs passed their 822 provided cases before
inference. The [aggregate report](reports/discovery-2026-10-09.json) preserves the
frozen evaluation result and provenance; full answers remain in ignored `runs/`.

| Passing answers per problem | Problems |
| --- | ---: |
| 4 / 4 | 83 |
| 1–3 / 4 | 8 |
| 0 / 4 | 9 |

Overall, **349/400 answers passed (87.25%)**. The broad sample scored 189/200
(94.5%); the additional medium/hard sample scored 160/200 (80%). These are
exploratory sampled-answer scores, not greedy or held-out final-test accuracy.
There were 39 wrong answers, four execution errors, and eight truncated answers.
All eight truncations came from two problems and count as failures under the
predeclared 512-token budget; they do not establish inability to solve those tasks.
Raw-source format compliance remained 0%; the evaluator accepts complete Python
fences without changing their enclosed code.

Two of the eight mixed groups belong to the alignment pool:

| Candidate | Passing answers | Observed failure |
| --- | ---: | --- |
| `Prefill_19551_I`: minimum palindrome deletions | 2 / 4 | Incorrect greedy logic instead of minimum-deletion dynamic programming |
| `Prefill_35466_I`: subarray sum that is a multiple of k | 1 / 4 | For k=0, looking only for a zero element misses multi-element zero-sum subarrays |

An [independent local check](reports/discovery-2026-10-09-independent-checks.json)
used eight additional cases per problem, with exhaustive small-input oracles.
The same two palindrome answers and one subarray answer passed all additional
cases. No new model calls were needed. The other mixed groups comprise five
SFT-pool problems and one development problem; preserve those assignments.

These give us real contrasting answers for DPO and nonconstant group rewards for
GRPO. For the palindrome group, rewards `[0, 1, 0, 1]` have mean 0.5: correct
answers are above the group mean and incorrect answers below it. This is evidence
of useful training signal, not yet evidence that a trained model will generalize.
The next learning step is to inspect the DP and greedy answers, then prepare the
small SFT pilot before branching into alignment runs. No training has run yet.

The run generated 55,291 tokens, used about 14.1 minutes inside GPU functions,
and peaked at 8.05 GiB allocated GPU memory. The application estimate including
its $0.25 setup reserve was **$0.93**. Modal's billing snapshot at
2026-10-09 07:25 UTC reported **$0.676345** for this app; reporting may lag final
metering. The app is stopped with zero tasks, and evaluation containers were
removed. Unit/contract checks and real Docker isolation checks passed (19 tests
across the validation runs).

Keep the original score unchanged while investigating data quality. In particular,
`Leetcode_41194_I` describes three-field operations but illustrates a two-field
deletion, and `Prefill_23088_I` does not explicitly state the replacement-case
convention. Those need specification review before becoming training feedback.
The two alignment candidates above have clearer, independently confirmed failures.

## Step 3: a frozen SFT pilot and its before-training score

The pilot uses **64 SFT demonstrations and 32 development problems** from the
existing family pools. Alignment and final-test pools remain reserved. This is a
small pipeline exercise, not enough data to establish broad coding improvements.
Development is for iteration; some of its tasks were already examined during
discovery, so it is not an untouched final test.

An SFT example is the same system instruction and problem/signature used at
inference, followed by the dataset's reference Python solution as the assistant
answer. We retain the reference text, including any comments and docstrings.
We do not train on sampled incorrect answers or include execution tests in the
training export. `data/sft-pilot-v1-train.jsonl` contains the 64 conversations,
token IDs, attention masks, and labels; `data/sft-pilot-v1.json` contains the
separate problems and grading cases. Both are local, ignored dataset artifacts.
The committed `manifests/sft-pilot-v1.json` records their provenance and hashes.

**What does the model learn here?** Each answer token is a supervised next-token
prediction. Prompt positions have label `-100`, which means ignore their loss;
answer tokens and the final end-of-turn token retain their token IDs as labels.
The model still reads the prompt. A future trainer must preserve these masks and
mask padding too; a causal language model performs the one-token label shift.
We verify the training prefix equals the actual inference prompt, allow at most
1,024 prompt and 512 target tokens, and reject oversized examples instead of
cutting off Python code. No optimizer or adapter training runs in this step.

Preparation checks original source-family separation, screens identifier-normalized
reference implementations, and includes manual checks for equivalent tasks with
different implementations. These checks reduce leakage without proving absence
of semantic duplicates. Every selected reference passes its supplied cases in two
fresh isolated Docker executions. Passing those cases alone is insufficient:
manual target review found several bugs, documented with independent countercases
in `reports/sft-pilot-reference-audit.json`. Exclusions and review IDs are recorded
in `manifests/sft-pilot-audit.json`; we exclude faulty or ambiguous demonstrations
rather than silently repair their labels. Some algorithm-specific tasks are also
excluded because our output grader cannot verify the requested implementation.
The remaining tests are finite and do not prove complete correctness or performance
at the largest input sizes.

From this directory:

```bash
uv run --extra inference python pilot_data.py prepare
# Review all IDs listed in needs_review; update the audit and prepare again if excluded.
uv run --extra inference python pilot_data.py freeze
RUN_DOCKER_TESTS=1 uv run --extra inference python -m unittest discover -s tests -v
# Commit and push the exact runner and frozen manifests before spending GPU credits.
uv run --extra inference python pilot_baseline.py
```

The baseline uses greedy decoding on development only, with the same 512-token
response cap and evaluator as the future after-SFT comparison. It generates and
grades one canary before continuing, checkpoints answers, ends the GPU app before
local bulk grading, and reserves the next call's maximum configured runtime before
starting it. Its $1.50 application allowance includes a $0.25 setup reserve; this
is a conservative estimate, not a provider-enforced billing cap. Do not rerun just
to regrade saved outputs. Freeze files refuse overwrite with changed content.

Before training, inspect one JSONL row and explain which tokens contribute to
loss. The next exercise will be a small adapter SFT run, followed by the **same**
development evaluation to check both improvements and regressions.

### Pre-SFT measurement (2026-10-09)

The [frozen baseline report](reports/sft-pilot-baseline-2026-10-09.json), run from
`5e64befe` on all 32 development problems, records **29/32 passing (90.625%)**:
29 passed, two wrong answers, and one execution error. The evaluator credited
231/240 cases; an execution error aborts a problem's batch and credits zero cases
for that problem, so this case count is not an independent case-by-case accuracy.
All responses completed within the token limit. All 32 used Markdown fences,
which the evaluator accepted; raw-format compliance remains a separate 0/32.

Output review matters as much as the aggregate score:

- `Filter_27332_I`: a missing regex boundary accepts the first five digits of a
  six-digit number. This is a clear implementation error.
- `Filter_39729_I`: Qwen lowercases characters; tests preserve their case. The
  question's instruction to ignore case sensitivity makes this ambiguous.
- `Filter_85258_I`: Qwen raises on an empty input list; the test expects `[]`, but
  the question does not specify behavior for invalid indices.

Keep the frozen score, but do not interpret the latter two failures as reliable
training opportunities. They remain development examples, never training rows.
Any future clarification needs an explicitly versioned prompt/test contract and
matching before/after evaluation; do not silently revise labels after seeing scores.
The earlier manual audit reduced problems but did not eliminate specification
ambiguity. This is a useful reason to inspect outputs, not just trust an accuracy
number.

The 64 SFT rows contain **7,452 supervised tokens**, with targets ranging from
37 to 237 tokens. The 96 problems have 754 supplied test cases. All 25 tests passed
in the validation run including real Docker isolation; actual pinned-tokenizer
checks also verified all 64 label masks, prompt prefixes, lengths, and end tokens.
The baseline generated 3,393 tokens, spent about 135.5 seconds inside GPU functions,
and peaked at 7.69 GiB allocated GPU memory. Modal reported **$0.09982899** at
07:57 UTC (billing may lag); the conservative estimate including its setup reserve
was **$0.37** against the $1.50 allowance. The app stopped with zero tasks and no
evaluation containers remained. No SFT optimization has run yet.


## Step 4: first LoRA SFT run

Run `uv run --extra inference --extra training python sft_run.py` after committing
and pushing the exact runner. This uses the frozen 64 examples and the same 32
problems, prompt template, 512-token greedy decoding, and evaluator as the baseline.
No development or final-test answers enter training.

The pilot uses Transformers `Trainer` with PEFT 0.21.2 and the pinned dependency
lock. TRL 1.15.0's SFT loss path requires Triton even for its CPU test path; the
standard Trainer lets us exercise the **same** masked cross-entropy and collator
on a tiny local CPU Qwen and the real CUDA model. No library monkey patches or
platform-specific loss substitutions are used. Future DPO/GRPO trainers are
separate decisions.

- **LoRA:** rank 16, alpha 32, dropout zero on `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. Base weights remain BF16 and
  frozen; PEFT manages trainable adapter precision. No quantization.
- **Optimization:** AdamW, learning rate `1e-4`, constant schedule, no warm-up,
  zero weight decay, gradient norm clipping at 1, seed 42.
- **Batching:** one example per microbatch; accumulate 16 examples per optimizer
  step. One epoch over 64 examples means **four updates**, not 64 updates.
  Gradient checkpointing trades extra forward computation for activation memory.
- **Loss:** shifted next-token cross-entropy in FP32, summed over answer tokens
  and divided by the total supervised-token count across the accumulated batch.
  This avoids giving a short microbatch disproportionate weight. Prompt and
  padding labels are `-100`; the answer's end token contributes to loss.
- **Canary:** eight updates on one training example, requiring at least a 5%
  reference-loss reduction. Save/reload logits must agree within `1e-5` absolute
  and relative tolerance, and downloaded adapter hashes must match. Then discard
  this adapter and start the full pilot from fresh base weights and optimizer.
- **Artifacts:** adapters persist on the results volume and are downloaded under
  ignored `checkpoints/<run-id>/`. Run records include losses, trainable parameter
  counts, memory peaks, paired evaluation changes, and dependency versions.

Local checks compare the loss to independently computed cross-entropy, compare
accumulated versus full-batch gradients and optimizer updates for unequal target
lengths, verify only LoRA parameters change, and reload a real saved tiny adapter.
The full-model GPU canary validates CUDA/BF16 execution before the pilot. Training
and evaluation use a single A100 80GB per function, no automatic retries, bounded
calls, durable outputs, and a $4 application allowance including a $0.25 setup
reserve. The allowance is not a provider-enforced billing cap.

Reference-answer NLL is measured over the same 64 training answers before and
after SFT. A decrease shows the model fits demonstrations better; correctness on
the separate development problems is the check for transfer. Keep all gains and
regressions, including the previously documented ambiguous cases. The final
one-epoch checkpoint is chosen in advance; this pilot does not sweep settings.

Implementation references: [PEFT LoRA](https://huggingface.co/docs/peft/en/package_reference/lora),
[Transformers gradient accumulation](https://huggingface.co/docs/transformers/grad_accumulation).

The first GPU canary exposed a precision-contract mismatch: Accelerate leaves a
training-time autocast wrapper on the model, while a freshly loaded adapter uses
ordinary inference precision. The runner now removes that wrapper through
`accelerator.unwrap_model(..., keep_fp32_wrapper=False)` and disables gradient
checkpointing before post-training NLL, export checks, and inference comparison.
A CPU regression test uses Accelerate's actual BF16 wrapper and verifies exact
restoration of the original inference outputs. This keeps the reload tolerance
strict rather than widening it to hide a mismatch. The $4 allowance also reserves
$0.25 for the initial terminated image build and rejected GPU canary; actual
provider billing is recorded separately when available.

### First SFT result (2026-10-09)

The [completed run report](reports/sft-pilot-2026-10-09.json) records a one-epoch
pilot from `8ad64017`, following a passing single-example overfit and adapter
reload canary. The final checkpoint was selected in advance; no development-based
hyperparameter sweep was performed.

| Measurement | Starting Qwen | After SFT |
| --- | ---: | ---: |
| Development problems passed | 29/32 (90.625%) | 27/32 (84.375%) |
| Raw-code format compliance | 0/32 | 32/32 |
| Training-reference token NLL | 0.9330 | 0.2986 |
| Generated development tokens | 3,393 | 2,562 |

All responses completed within the 512-token cap. The evaluator accepts the
baseline's fences, so its correctness score was not penalized for those fences.
The SFT run learned raw-code output and better fit the demonstrations, but **did
not improve functional accuracy**. Training loss measures how well the model
predicts reference answer tokens; it is not a code-execution reward.

The paired comparison found one improvement and three regressions:

- `Filter_85258_I` now uses ordinary slicing and passes the empty-list case.
  That case was already flagged as underspecified; do not treat it as strong
  evidence of a general coding improvement.
- `Algorithm_34333_I` now omits the import for its `Sequence` type annotation,
  so the otherwise plausible implementation fails in the pinned Python runtime.
- `Filter_51966_I` now crashes when parsing an empty string. The supplied test
  expects an empty dictionary, although the question does not specify empty-input
  behavior explicitly.
- `Leetcode_26807_I` remembers only the first occurrence of each prefix balance,
  rather than counting all previous matching balances. For `010101`, it counts
  five balanced substrings instead of nine.

The original zipcode-regex and case-preservation failures remain. Keep these
observations in development; do not move their answers into training. We have
only 32 problems and one seed, with known specification ambiguities. This is
useful pipeline evidence, not a reliable estimate of broad model quality or proof
that SFT in general harms performance. Preserve the pilot as an exploratory
checkpoint rather than promote it as a stronger coding model.

Only **33,030,144 LoRA parameters** were trained (about 0.81% of the adapted model's
parameters). The optimizer portion took **49.504 seconds** for four updates,
processing 7,452 supervised answer tokens. Including loading, before/after NLL,
saving, and reload verification, the training function took **88.207 seconds**.
Peak memory was **9.46 GiB allocated / 15.47 GiB reserved**; the latter includes
PyTorch's caching allocator. These measurements exclude any implication that a
GPU with exactly that advertised capacity is sufficient, or that longer contexts
and larger microbatches would have the same footprint.

The final adapter is 132,187,888 bytes (about 126 MiB), saved on Modal's results
volume at `sft-20261009T155203Z-8ad64017/train/adapter` and downloaded to
`checkpoints/sft-20261009T155203Z-8ad64017/train/adapter/`. Its SHA256 is
`e9d423228810165df3fdb2297269ac02ee00f66a976c9b423e3e61708f74e46a`.
The backbone is still the pinned Qwen revision; the adapter is not a standalone
model. Both the canary and final adapter reload checks had **zero** maximum logit
difference after correcting the inference precision contract.

Thirty local checks passed across validation runs, including real Docker
isolation and real CPU training. The conservative application estimate is
**$0.86**, including $0.25 setup and $0.25 prior-attempt reserves, against the $4
allowance. As of 16:00 UTC, Modal's billing report had no rows for these three apps;
actual charges are pending, not zero. All three apps are stopped with zero tasks,
and no evaluation containers remain. Full answers, adapters, and dataset records
remain local/ignored; the committed report records metrics, provenance, and the
failed setup/canary attempts transparently.

Next, review one regression and the training loss curve before changing the
recipe. Any subsequent run should change a stated hypothesis (for example, data
coverage or update strength) and repeat the same development comparison. The
final-test pool remains reserved.

### Training accuracy versus training loss

`uv run --extra inference --extra training python training_accuracy.py` evaluates
the saved SFT adapter on the exact 64 training prompts. It performs no optimization.
Only prompt fields reach the inference worker; reference solutions and cases stay
with the local evaluator. The adapter's content hashes are verified, evaluation
outputs use a separate run directory, and the same greedy decoding/512-token cap
and Docker evaluator are retained. One end-to-end canary precedes the remaining
63 answers. Commit/push before a paid run; the application allowance is $2.

This metric differs from reference correctness (all 64 reference solutions passed
514 supplied cases, twice) and from teacher-forced training NLL. NLL evaluates the
probability of reference tokens when earlier reference tokens are supplied.
Generated-code accuracy requires the model to produce the whole answer from the
problem alone, then pass the tests. The previous full-training-set NLL bars
(0.933 before / 0.299 after SFT) were never an accuracy measurement. Starting-model
generated accuracy on these 64 training prompts was subsequently measured: 57/64 both before and after SFT.

### SFT concepts behind this run

Both loss charts use average answer-token negative log likelihood (NLL), in
**nats** because the logarithm is natural. NLL equals cross-entropy against the
one-hot reference-token target. A single token assigned probability 0.5 incurs
about 0.693 nats; probability 0.9 incurs about 0.105. Neither number is an error
percentage. The four logged points are different accumulated training batches,
measured before their respective updates. The full-set bars measure all 64
reference answers before and after SFT. Even a program with mostly likely tokens
can fail because of one crucial missing import or incorrect condition.

LoRA leaves an existing matrix `W` frozen and learns a low-rank correction:

```text
Original layer: y = W x
With LoRA:      y = W x + (alpha / rank) B(Ax)
Trainable:      A and B
Frozen:         W
```

The low-rank constraint applies to the correction, not the original matrix or the
whole model. Here rank is 16 and alpha is 32, so the correction is scaled by 2.
This scale is distinct from the optimizer learning rate. In every one of the
36 transformer blocks, we adapt attention `q_proj`, `k_proj`, `v_proj`, `o_proj`
and feed-forward `gate_proj`, `up_proj`, `down_proj`. The original projections,
embeddings, normalization weights, and output head remain frozen. The saved
adapter has 504 tensors: 36 blocks × 7 projections × two matrices.

For a concrete attention query projection, the original matrix is 4096 × 2560
(10,485,760 weights). Its LoRA matrices are 16 × 2560 and 4096 × 16: 106,496
trainable weights. Across all adapted projections there are 33,030,144 trainable
weights. These added matrices are the **adapter weights**, not a standalone LLM;
the saved adapter requires the pinned base checkpoint.

QLoRA combines LoRA with a quantized frozen backbone, commonly stored in 4-bit
NF4. Adapter training remains in higher precision; it does not mean all forward
or backward arithmetic runs in four bits. Our run used ordinary LoRA with a BF16
backbone. Quantization primarily reduces backbone storage; activations and
optimizer state still consume memory. The original QLoRA method also introduced
double quantization and paged optimizers. [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/conceptual_guides/lora),
[QLoRA paper](https://arxiv.org/abs/2305.14314).

Teacher forcing supplies the complete prompt and reference answer as inputs.
A **causal attention mask** prevents each token position from seeing future
positions, so one forward pass can compute next-token distributions at all
positions in parallel. A separate **loss mask** excludes prompt and padding
positions. A backward pass then computes gradients. In our configuration:

```text
One example: forward → masked loss → backward → accumulate gradients
Repeat for 16 examples → clip accumulated gradient → one AdamW update → clear gradients
64 examples → four optimizer updates
```

The microbatch size is 1; the effective batch size is 16 on one GPU. Accumulation
lets each microbatch's activation graph be released after backward, rather than
retaining graphs for all 16 examples. Our loss is the sum of answer-token losses
divided by the total number of answer tokens across the effective batch. This
also scales raw gradients by that denominator. A 100-token answer contributes
ten times as many loss terms as a 10-token answer; this is not an equal-weight
average of per-response mean losses. Gradient clipping at norm 1 is a separate
operation, and AdamW further transforms the gradients into parameter updates.

**Intermediate activations** are temporary values produced inside the network,
such as a layer's input features and attention/MLP outputs. Backpropagation needs
some of these to calculate derivatives. For `y = W x`, the gradient with respect
to `W` depends on both the backward signal reaching `y` and the earlier input `x`.
For LoRA, the gradient for `B` needs its input `Ax`; the gradient for `A` needs `x`
and the backward signal through `B`. Freezing the backbone does not eliminate
this computation: gradients must still travel through frozen operations to reach
trainable adapters in earlier layers. Autograd saves required intermediates;
gradient checkpointing instead recomputes some of them during backward, trading
extra computation for memory. [PyTorch activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/).

### Generated training-set accuracy (2026-10-09)

The [inference-only report](reports/sft-training-accuracy-2026-10-09.json) evaluates
the saved adapter on all 64 SFT training problems: **57/64 pass (89.0625%)**,
with 496/514 individual cases passing. Seven programs return incorrect answers;
none fail execution or hit the token cap. All 64 comply with raw-code formatting.
The model generated 6,627 tokens. No weights were updated.

| Measurement | Result | Meaning |
| --- | ---: | --- |
| Reference solutions passing supplied cases | 64/64 | Dataset-target validation |
| After-SFT reference-token NLL | 0.2986 | Teacher-forced average token loss |
| After-SFT generated training accuracy | 57/64 (89.0625%) | Performance on problems used in training |
| After-SFT generated development accuracy | 27/32 (84.375%) | Performance on separate development problems |

These are different measurements. We have not measured starting-model generated
accuracy on these exact 64 problems, so this new result cannot establish an
improvement in training-set correctness. The train/development difference alone
also does not prove overfitting: the task sets and their difficulty mixes differ.

The failed training IDs are `Algorithm_20361_I`, `Apps_16553_I`, `Apps_17117_I`,
`Evol_9946_I`, `Prefill_10912_I`, `Prefill_38802_I`, and `Prefill_7450_I`. Preserve
the frozen evaluation instead of changing its labels after observing these answers.

Adapter hashes and exact training membership were verified locally and before
cloud inference. The new membership/output-directory checks and seven existing
inference contract checks passed. Modal reported **$0.28587341** at 18:09 UTC
(reporting may lag); the conservative estimate including setup reserve was
**$0.67**, within the $2 allowance. The app stopped with zero tasks and no evaluator
containers remained. Answers remain in the ignored local run file and results
volume; the aggregate report is committed.

### Complete SFT accuracy comparison protocol

The requested comparison reuses the saved 32-problem development evaluations and
64-problem after-SFT training evaluation. `training_accuracy.py --policy base`
measures the original pinned checkpoint on the same training prompts. No further
training takes place.

`test_data.py` freezes 300 problems from the previously reserved final-test family
pool in deterministic hash order. It excludes known audit problems and duplicate
families/normalized implementations, including matches to the 96 pilot problems.
Every selected reference passes the supplied cases twice in the same isolated
runtime. This is automated reference validation, not exhaustive human semantic
review; shared reference/test mistakes and unknown pretraining contamination remain
limitations. Selection never uses either checkpoint's scores.

After committing the manifest, run `training_accuracy.py --split test --policy base`
and `training_accuracy.py --split test --policy sft`. Both use identical greedy
512-token decoding, prompt construction, and Python-output-v2 grading. The budget
allowance is $8 per 300-problem run ($2 for the missing training baseline), with a
single-example end-to-end canary before subsequent batches. Raw generations are
saved before grading and kept ignored; aggregate reports and provenance are tracked.
Exposing this test set means it must not become a source of fixes or checkpoint
selection for later PPO/DPO/GRPO experiments.


### Completed paired accuracy evaluation (October 9)

| Set | Before SFT | After SFT |
|---|---:|---:|
| Training (64) | 57/64 (89.06%) | 57/64 (89.06%) |
| Development (32) | 29/32 (90.63%) | 27/32 (84.38%) |
| Held-out test (300) | 266/300 (88.67%) | 260/300 (86.67%) |

See `reports/sft-accuracy-comparison-2026-10-09.json`. Test evaluation is now complete;
earlier statements that the final-test pool remains reserved describe the earlier
pilot stage. No new training was done for this comparison. A batch-boundary bug
stopped the baseline at 298 answers; the remaining two were recovered without
regenerating those 298. The runner now derives bounds from the shared batching
function and supports provenance-checked recovery of complete saved batches.
The three new evaluations cost $5.18 as reported by Modal at 20:41 UTC (metering
may lag). All four apps, including the recovery attempt, stopped with zero tasks.
Serial generation and unmerged adapter overhead made this runner slow; no
performance improvement should be inferred from these experiments.

### Fifteen-example SFT memorization diagnostic

`overfit_data.py` freezes all seven baseline failures from the 64 audited training
problems plus eight baseline successes in deterministic hash order. The starting
score is **8/15 (53.33%)**. Five failures are wrong answers, one is a formatting
failure, and one is truncation; we preserve this distinction. Neither development
nor test examples participate in selection or training. This intentionally selected
subset is a diagnostic, not a representative benchmark.

`overfit_run.py` starts from the original pinned Qwen checkpoint and a fresh rank-16
LoRA adapter, using the same attention/MLP targets, answer-token loss, BF16 backbone,
and 1e-4 learning rate. Microbatch 1 × accumulation 5 gives effective batch 5;
60 optimizer updates cover 20 epochs of 15 examples. This tests whether repeated
supervision can change generated correctness, beyond the previous four-update
pilot. A discarded one-example canary validates loss decrease, serialization and
artifact download before fresh training. The final adapter is saved, reloaded,
logit-checked, and used to generate exactly one greedy answer per selected problem.
The fixed 512-token cap and existing evaluator remain unchanged. Results are saved
in two evaluation chunks, rather than committing the remote volume after every
answer. The complete experiment has a $4 allowance.

Success criterion: **15/15 generated solutions pass their supplied cases**. Report
actual accuracy even if this criterion is not met. Report before/after full-subset
reference NLL separately; lower teacher-forced loss is not proof of correct free
generation. A successful run demonstrates in-sample memorization only. It does not
establish generalization or promise that RL will help.

Selected examples:

| ID | Task | Baseline |
|---|---|---|
| `Algorithm_20361_I` | `extract_numbers` | wrong_answer |
| `Evol_9946_I` | `longest_repeated_substring` | wrong_answer |
| `Filter_26162_I` | `has_mirrored_pairs` | wrong_answer |
| `Filter_48511_I` | `is_prime` | format_error |
| `Filter_58003_I` | `is_sorted_and_unique` | truncated |
| `Prefill_31266_I` | `is_valid_phone_number` | wrong_answer |
| `Prefill_7450_I` | `swap_adjacent_characters` | wrong_answer |
| `Prefill_7890_I` | `is_sum_zero` | passed |
| `Prefill_28477_I` | `count_set_bits` | passed |
| `Apps_14662_I` | `find_unsorted_subarray` | passed |
| `Apps_14754_I` | `largest_rectangle_area` | passed |
| `Filter_113_I` | `longest_word` | passed |
| `Filter_68887_I` | `rotate_string` | passed |
| `Filter_3471_I` | `encrypt` | passed |
| `Filter_67993_I` | `convert_negatives_to_positives` | passed |


#### Diagnostic result: 8/15 → 15/15

The 60-update experiment completed successfully. All seven previously failing
problems now pass; all eight previous successes remain correct. All 15 outputs
are raw-format compliant. See `reports/sft-overfit-15-2026-10-09.json` for the
per-problem scores and provenance.

| Measure | Before | After |
|---|---:|---:|
| Generated-code accuracy on selected training problems | 8/15 (53.33%) | 15/15 (100%) |
| Full-subset reference-token NLL (nats) | 0.930229 | 0.000039932 |

The fresh rank-16 adapter received 60 updates at 1e-4, effective batch 5, across
20 epochs. Actual optimizer training took 158.7 seconds; the complete app, including
canary, model loading, adapter verification, answer generation and retrieval, ran
about seven minutes. Modal reported $0.30554 at 20:52 UTC (may lag); the conservative
estimate including setup reserve was $0.55 against a $4 allowance. The GPU app
stopped with zero tasks and no evaluator containers remained.

This establishes that the pipeline can memorize the selected training problems and
change greedy generated correctness. It does not establish a better general coding
model. A subsequent validation-only evaluation is recorded below; no new final-test
evaluation was performed for this deliberately overfit checkpoint. The earlier four-update pilot and this diagnostic differ in subset,
number of epochs and effective batch size, so this is not an isolated causal test
of update count alone. Keep this checkpoint separate from any model selected for RL.

At the user's request, evaluate this diagnostic checkpoint on **only the existing
32-problem development/validation set**, with no new 300-problem test evaluation:

```sh
uv run --extra inference --extra training python training_accuracy.py --policy sft --split validation --checkpoint overfit
```

The shared inference runner verifies the saved overfit adapter's hashes, keeps its
cache separate from the first SFT adapter, and requires the exact frozen validation
IDs. It retains greedy 512-token decoding and the existing evaluator, with a $2
allowance. No optimization takes place; compare to the saved original-model 29/32
baseline and the first pilot's 27/32. Remote persistence is once per completed
batch for this diagnostic adapter.


#### Validation-only follow-up

The same saved overfit adapter passed **26/32 validation problems (81.25%)**, versus
29/32 (90.63%) for the original checkpoint and 27/32 (84.38%) for the first SFT
pilot. It passed 203/240 individual cases and produced 32/32 raw-format compliant
answers. See `reports/sft-overfit-validation-2026-10-09.json`.

| Set | Original model | 15-example overfit adapter |
|---|---:|---:|
| Selected training subset | 8/15 (53.33%) | 15/15 (100%) |
| Existing validation set | 29/32 (90.63%) | 26/32 (81.25%) |

This demonstrates in-sample memorization without improved validation correctness.
No new training or 300-problem test evaluation took place. The validation run was
interrupted by a local DNS error: 10 answers were already local and nine more were
recovered from remote storage. Only the remaining 13 were generated afterward.
The runner now handles the Volume API's `FileNotFoundError` for an absent next
checkpoint, and validates recovered batch IDs before continuing. Nine local
inference/recovery contract tests and real present/missing Volume reads passed.
Both GPU apps stopped with zero tasks; no evaluator containers remained. Modal
reported **$0.23399** for validation, including the interrupted app, at the time
recorded in the report; provider metering may lag.

### GRPO diagnostic: original Qwen, verified execution rewards

The current outcomes and revised plan are summarized in [RESULTS.md](RESULTS.md).
All earlier results were committed before implementing this experiment. The two
training problems are `Prefill_19551_I` and `Prefill_35466_I`, the previously audited
mixed-success **alignment-pool** tasks. They are independent of the 32 development
problems. Their original supplied-case reward contracts remain unchanged. This is
a two-problem diagnostic, not a broad benchmark or an equal-data SFT comparison.

`grpo_core.py` implements the original-style clipped GRPO objective explicitly using
PyTorch/PEFT, with the same code exercised by real CPU model tests and on the GPU.
It does not claim to be TRL's current default objective: recent TRL defaults use a
different length reduction and omit KL unless requested. Here each response's token
losses are averaged, then the eight responses are averaged. Group advantages use
population standard deviation plus 1e-4; constant groups get zero task advantage
but may still receive a KL gradient. The sampled-token KL estimator is
`exp(logp_ref - logp) - (logp_ref - logp) - 1`.
[Original GRPO](https://arxiv.org/abs/2402.03300),
[TRL objective variants](https://huggingface.co/docs/trl/grpo_trainer).

Settings: two prompts × four sampled completions per rollout, ten fresh rollout
rounds, two optimizer passes per rollout (20 updates), AdamW at 1e-5 without weight
decay, gradient clipping 1, policy-ratio clipping 0.2, and reference KL coefficient
0.02. Rank-16 LoRA uses the existing attention/MLP target modules and a frozen BF16
backbone. The original checkpoint is the fixed reference, evaluated by disabling
the adapter; no value model or separate reward model is trained. Sampling uses
temperature 1, top-p 1, top-k 0, no dropout, and a 512-token cap. Prompt tokens and
post-EOS padding are excluded from the loss; sampled EOS is included. Truncated
responses get reward zero, retaining their sampled-token trajectory for learning.

A stateful A100 worker generates complete groups of four in one batched generation
call per prompt. The host grades code in the same isolated Docker evaluator and
returns numeric rewards only. The GPU receives neither expected test outputs nor
reference solutions. Behavior log probabilities are captured before any update;
reference probabilities remain fixed. Complete rollout records, adapter parameters,
and optimizer state are checkpointed. Duplicate rollout/update requests are checked
by version and output hash to avoid applying an update twice after a network loss.
The first round must have mixed rewards, improve its surrogate objective, and pass
an exact-logit checkpoint reload before further rounds proceed.

The local coordinator is `uv run --extra inference --extra training python grpo_run.py`.
The combined training and evaluation allowance is $4, including conservative
startup/call/idle reservations. It measures greedy accuracy on the two training
problems before/after, four fixed-seed samples per problem before/after, reward and
optimization diagnostics per round, and greedy accuracy on **only the existing 32
validation problems**. It reuses the original validation baseline, does not run the
300-problem test set, and does not reuse the overfit adapter. Eight sampled answers
are a very small, noisy diagnostic; identical seeds do not make them an unbiased
estimate of generalization.

#### Completed GRPO results (2026-10-09)

The ten-round run completed all 20 optimizer updates on original Qwen plus a fresh
LoRA adapter. Greedy training correctness improved **1/2 → 2/2**, and four fixed-seed
samples per training problem improved **3/8 → 7/8**. Existing validation remained
**29/32**: every problem retained its original pass/fail outcome. This shows learning
on the selected training tasks without a measured validation gain. It does not
establish superiority over the differently sized SFT experiments.

Of 20 prompt groups, 12 had mixed rewards and eight had zero reward variance.
Round 3 had zero task advantage for both prompts; KL gradients and optimizer
momentum can still change parameters. Optimizer work took 64.7 seconds; generation,
evaluation, checkpoint checks, and coordination brought app duration to about
30 minutes. Both adapter reload checks had zero logit error. Five GRPO tests and
nine existing inference/recovery tests passed before the paid run.

Modal reported $1.34315 at 22:38 UTC (billing may lag), below the $4 allowance; the
runner's separate conservative estimate was $2.93. The app stopped with zero tasks
and no evaluator containers remained. No 300-problem test run was performed.
See [RESULTS.md](RESULTS.md) and the
[full GRPO report](reports/grpo-pilot-2026-10-09.json) for provenance and diagnostics.
