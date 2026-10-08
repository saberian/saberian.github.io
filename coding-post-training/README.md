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
| Training software | TRL for SFT/DPO/GRPO; a small, explicitly tested PyTorch/Transformers PPO implementation |
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
