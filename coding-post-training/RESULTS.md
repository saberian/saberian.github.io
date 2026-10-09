# Experiment results

All runs use pinned **Qwen/Qwen3-4B-Instruct-2507**, a non-thinking model, and the
same greedy 512-token correctness evaluation (sampled GRPO metrics are labeled separately) with the Python-output-v2 evaluator. A problem
passes only when every supplied test passes. Raw formatting is tracked separately.
Generated code executes in isolated containers, never on the host.

| Checkpoint | Training evaluation | Validation (32) | Test (300) |
|---|---:|---:|---:|
| Original Qwen | 57/64 (89.06%) | 29/32 (90.63%) | 266/300 (88.67%) |
| First SFT: 64 examples, 4 updates | 57/64 (89.06%) | 27/32 (84.38%) | 260/300 (86.67%) |
| Overfit diagnostic: 15 examples, 60 updates | 15/15 (100%); original was 8/15 | 26/32 (81.25%) | Not evaluated |

The first SFT run lowered reference-token NLL from 0.9330 to 0.2986 and improved
format compliance, without improving whole-problem correctness. The deliberately
selected 15-example diagnostic lowered NLL from 0.930229 to 0.000039932 and
reproduced all 15 references (ignoring outer whitespace), passing all 126 cases.
All seven prior failures improved, with no regressions among eight prior successes.
Its validation result was worse: three original failures improved, but six original
successes regressed. This establishes memorization, not generalization.

The overfit diagnostic used a fresh rank-16 LoRA adapter, frozen BF16 backbone,
learning rate 1e-4, microbatch 1, accumulation 5, and 20 epochs. Optimizer training
took 158.7 seconds. Modal reported $0.30554 for training/canary/training evaluation
and $0.23399 for the validation follow-up, including recovery. Provider billing
may lag. The earlier broad paired evaluations reported $5.18 in total. Reports
record timestamps, application IDs, checkpoint hashes, settings, and cleanup.

## Evidence

- [Paired pilot accuracy](reports/sft-accuracy-comparison-2026-10-09.json)
- [Initial SFT training](reports/sft-pilot-2026-10-09.json)
- [15-example memorization](reports/sft-overfit-15-2026-10-09.json)
- [Overfit validation](reports/sft-overfit-validation-2026-10-09.json)
- [Mixed-outcome discovery](reports/discovery-2026-10-09.json)
- [Independent checks on two alignment problems](reports/discovery-2026-10-09-independent-checks.json)

## Alignment experiment scope

Subsequent alignment experiments start independently from **original Qwen**, not
from the deliberately overfit adapter. This supersedes the original proposed
SFT-first branching plan in README sections 5–7. Only alignment-pool problems may
provide GRPO updates. The existing 32 validation problems remain held out from
updates; they have been inspected repeatedly, so they are development evidence,
not an untouched final test. No further 300-problem evaluation is planned.

## Completed GRPO diagnostic

GRPO started from original Qwen with a fresh rank-16 LoRA adapter. It used **two
previously audited alignment-pool problems**, four sampled answers per problem,
ten fresh rollout rounds (80 training answers), and two optimizer passes per
round (20 updates). Rewards were binary execution-test outcomes; reference
solutions were not training targets. This is independent of both SFT adapters.

| Evaluation | Original Qwen | After GRPO |
|---|---:|---:|
| Two training problems, greedy | 1/2 (50%) | 2/2 (100%) |
| Training answers, four fixed-seed samples per problem | 3/8 (37.5%) | 7/8 (87.5%) |
| Existing validation, greedy | 29/32 (90.63%) | 29/32 (90.63%) |

All 32 validation pass/fail outcomes were unchanged, with 231/240 individual cases
passing after GRPO. The same three problems failed. No 300-problem test evaluation
was run. This demonstrates a training response to execution rewards, with **no
measured validation gain**. Eight sampled answers are a noisy diagnostic, and the
SFT and GRPO experiments used different data sizes and budgets; these results do
not establish that GRPO is a better algorithm.

Twelve of 20 prompt groups had mixed rewards. Eight had zero reward variance and
therefore zero task advantage, although KL regularization and optimizer momentum
can still move parameters. Round 3 had no mixed groups at all. Mean rollout reward
rose from 3/8 in round 1 to 7/8 in round 10, with fluctuations between rounds. The
last update's sampled reference-KL estimate was 0.0050 and clipping fraction was
0.00383; these are token-averaged optimization diagnostics, not accuracy metrics.

The 20 optimizer updates took 64.7 seconds combined. The full app ran about 30
minutes, including generation, evaluation, checkpoint checks, and coordination.
First-round peak allocated GPU memory was 11.81 GiB. Later cumulative peaks reached
16.08 GiB and include the extra model loaded for the checkpoint-reload check; they
should not be interpreted as steady-state training requirements. Both first-round
and final adapter reloads reproduced the checked logits exactly. Five GRPO tests
and nine existing inference/recovery tests passed before launch.

Modal reported **$1.34315** for this combined training/evaluation app as of
2026-10-09 22:38 UTC; billing may lag. The conservative runner estimate was $2.93
against its $4 allowance. The app is stopped with zero tasks, and no evaluator
containers remain. Code revision `3aa14fd` was committed and pushed before the run.

Full scores, hashes, settings, per-round diagnostics, billing, and cleanup:
[GRPO report](reports/grpo-pilot-2026-10-09.json).
