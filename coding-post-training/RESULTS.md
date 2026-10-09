# Experiment results

All runs use pinned **Qwen/Qwen3-4B-Instruct-2507**, a non-thinking model, and the
same greedy 512-token evaluation with the Python-output-v2 evaluator. A problem
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

## Interpretation and next experiment

Subsequent alignment experiments start independently from **original Qwen**, not
from the deliberately overfit adapter. This supersedes the original proposed
SFT-first branching plan in README sections 5–7. Only alignment-pool problems may
provide GRPO updates. The existing 32 validation problems remain held out from
updates; they have been inspected repeatedly, so they are development evidence,
not an untouched final test. No further 300-problem evaluation is planned.

The next GRPO diagnostic uses the two independently checked mixed-outcome
alignment problems, four sampled answers per problem, ten fresh rollout rounds,
and two optimizer passes per round. It measures whether execution rewards can
change behavior without demonstration targets. Report zero-variance groups,
reward, clipping, reference divergence, greedy training accuracy, and the same
32-problem validation score. A gain is a hypothesis, not a guaranteed outcome.
