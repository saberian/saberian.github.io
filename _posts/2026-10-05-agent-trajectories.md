---
layout: post
title: "Gaps in Frontier Coding Agents for ML Tasks"
description: "Evidence-backed gaps in feature exploration, time management, validation consistency, and delivery across six ML-agent configurations."
image: /assets/images/agent-gap-frequency.png
---

In the [previous post]({% post_url 2026-09-16-spread-in-practice %}), we showed that repeated attempts by the same coding agent produced substantially different recommendation quality, with differences of up to 3x. We also verified that those differences were larger than the variation caused by different random seeds. As a reminder, each agent had one hour of CPU time to train a two-tower neural network for music retrieval. More details about the task, the results of the agents' attempts, and their traces are [here](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base).

The large variation in agent performance shows that the agents are not reliable and have significant gaps in their performance. To understand these gaps and shortcomings, below we review **the traces of 30 modeling attempts at high reasoning effort**: five each from Opus 5.5, Fable 5.1, Sol, Astra, Grok 4.6, and Gemini 3.8 Flash, all on the same two-tower retriever task.

## Summary of gaps

The reviewed attempts revealed gaps in ten areas, ordered by how many of the 30 attempts showed each gap:

<picture>
  <source media="(max-width: 600px)" srcset="{{ '/assets/images/agent-gap-frequency-mobile.svg' | relative_url }}" width="392" height="872">
  <img src="{{ '/assets/images/agent-gap-frequency.svg' | relative_url }}" width="1024" height="744" alt="Attempts showing each gap, out of 30: Implementation errors 21; Poor resource management 20, including one inferred case; Poor time management 15; Misinterpreting results 13; Poor experiment tracking and reproducibility 9; Delivery failures and task violations 7; Poor experiment design 6; Insufficient validation 6; Insufficient problem exploration 5; Information leakage 2.">
</picture>

- **Implementation errors:** Writing model, feature, training, evaluation, or experiment-control code that does not behave as intended.
- **Poor resource management:** Using memory, CPU, data access, or concurrent processes inefficiently, causing avoidable stalls, failed runs, or repeated work.
- **Poor time management:** Misjudging runtimes or remaining time, or allocating the budget poorly across exploration, training, validation, refitting, and delivery.
- **Misinterpreting results:** Drawing conclusions that the measurements do not support, overlooking relevant results, or using them incorrectly to choose the next step.
- **Poor experiment tracking and reproducibility:** Failing to preserve and correctly associate code, settings, weights, predictions, and scores, making experiments difficult to compare, recover, or reproduce.
- **Delivery failures and task violations:** Failing to turn the selected model into a complete, checked submission, or pursuing or delivering an approach that violates explicit task requirements.
- **Poor experiment design:** Running comparisons that cannot answer the intended question—for example, changing several factors together without controls that separate their effects.
- **Insufficient validation:** Selecting a model without adequately checking its quality or robustness, or failing to verify that the final configuration matches the one evaluated.
- **Insufficient problem exploration:** Committing to an approach without adequately exploring simple baselines, promising data signals, or relevant features, representations, and learning objectives.
- **Information leakage:** Allowing future information or held-out outcomes to influence training or features beyond what the evaluation protocol permits.

Expand a model below for the detailed analysis and links to the supporting traces.

<details markdown="1">
<summary><strong>Opus 5.5</strong></summary>

Opus sometimes drew conclusions that its experiments did not support and used a final training schedule it had not validated. It also repeatedly misjudged the remaining time, ending three attempts after roughly 20–31 minutes of the available hour.

### Poor time management

In run 3, Opus said it was [out of time for more tuning](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r03-20260926T034923Z#step-20), then [finished the entire attempt after 19.6 minutes](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r03-20260926T034923Z#step-25), leaving about 40 minutes unused. Run 4 ended after 31.3 minutes with the same claim: it [had run out of time to tune](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r04-20260926T034923Z#step-40).

In run 5, the mistaken estimate affected an experiment already in progress. Opus [stopped a validation job because competing jobs were stalling and it believed only nine minutes remained](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-35). It later completed a validation check, but [ended the attempt after 26.3 minutes while saying that most of the hour had gone to infrastructure problems](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-40). Its estimates made the remaining budget seem much smaller than it was, limiting the experiments it was willing to attempt.

### Poor resource management

In run 5, Opus [ran out of memory while preparing validation data and training the final model at the same time](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-29). After reducing memory use, it [launched validation alongside the final fit again](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-31). The jobs then [stalled while competing for CPU time](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-35). Run 2 showed a similar scheduling problem: [a training job was killed while a popularity-baseline job ran alongside it](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r02-20260926T034923Z#step-11).

### Misinterpreting results

In run 5, the [neural model scored 0.0257 on public validation, below the recent-popularity baseline of 0.0281](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-39). Opus [called it undertrained because training loss was still falling](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r05-20260926T034923Z#step-40) and suggested more epochs. Falling loss showed that the model was fitting its training data more closely; it did not show that more training would improve its recommendations. Undertraining was a hypothesis to test against held-out results.

### Poor experiment tracking and reproducibility

In run 1, a later experiment [overwrote the selected validation checkpoint](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r01-20260926T034923Z#step-24). The selected configuration had scored about 0.0333, while the new experiment scored about 0.0308. Opus correctly rejected the weaker model, but its weights had already replaced the better model's saved checkpoint. The selected validation ranking and final test model survived, but the exact validation model could no longer be inspected or reused directly.

### Poor experiment design

In run 2, Opus proposed dropout and weight decay to reduce overfitting, but [tested them while also doubling training from one epoch to two](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r02-20260926T034923Z#step-28). Public-validation NDCG@10 fell from [0.0275](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r02-20260926T034923Z#step-26) to 0.0242, and Opus concluded that [“regularization hurt performance further”](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r02-20260926T034923Z#step-29). The comparison showed that the new configuration was worse, but could not establish whether regularization was responsible. Testing regularization for one epoch against its existing one-epoch baseline would have answered that question with a short additional experiment.

### Insufficient validation

Opus chose four epochs for the final refit without first tuning the epoch count on held-out validation data. In run 4, it trained the final model on training plus validation data for four epochs, then [trained a separate validation model on training data alone for two epochs](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r04-20260926T034923Z#step-35). That model's validation score was about 0.0208 after the first epoch and [fell to 0.0189 after the second](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r04-20260926T034923Z#step-37). Its [final account acknowledged that the four-epoch schedule had never been validated](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r04-20260926T034923Z#step-39). Opus should first have used held-out validation results to choose the epoch count, then refitted on training plus validation data with that choice fixed.

### Information leakage

In run 1, Opus [checked whether a like was active before each historical training cutoff](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r01-20260926T034923Z#step-6), but [calculated its recency using the latest like timestamp across the full loaded history](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/opus55-high-2-1-283-base-r01-20260926T034923Z#step-8). If a user liked the same track again after the cutoff, that later event could change the weight assigned to the earlier training example. The recency calculation needed the same time filter as the active-like check, so that every feature reflected only information available at that point.

</details>

<details markdown="1">
<summary><strong>Fable 5.1</strong></summary>

Fable's code did not always match the experiments it intended to run: two attempts built history features incorrectly, and another never applied its requested dropout settings. It also submitted a changed configuration without validation and stopped with nearly half the hour still available.

### Implementation errors

In run 2, Fable [sorted interaction events by the original user IDs, then remapped those IDs with target users first](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-6). That remapping changed the user order, but the history builder [searched the remapped IDs as though they were still sorted](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-8). The resulting boundaries could give a user an empty history or include another user's events.

Run 1's recent-history feature had a different indexing error. Fable [placed each user's events at the beginning of a padded 200-position array, then selected the final 20 positions for its recent-history summary](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r01-20261003T075109Z#step-5). For a user with 50 events, the latest 20 occupy positions 30–49; the code instead selected positions 180–199, all padding. Users with 180 or fewer events therefore received an empty recent-history feature.

Run 4 tuned dropout without changing the model's dropout. Its [command-line parser accepted a dropout value, but the model constructor did not pass it to the user tower](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-6). Commands requested [0.3](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-9) and later [0.5](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-15), while the tower kept its default of 0.2. Other settings changed, but the intended dropout comparison never happened.

A smaller process-control error cost several minutes in run 5. Fable [waited for a process-name pattern that could match the waiting command itself](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-16). The refit had written its output at 08:49:10, but Fable was still [investigating the wait after 08:53](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-18). It eventually [completed and checked the selected refit](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-19), so the submission was preserved.

### Poor resource management

In run 4, Fable [lost a parallel experiment during preprocessing](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-13), leaving its [planned comparison of training targets incomplete](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-25). Fable suspected memory pressure, but the cause was not confirmed.

Other attempts also encountered failures while running memory-intensive jobs together. In run 2, two training jobs were using about 3.5 GB each when [a baseline-scoring job that built a dense score matrix was killed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-12). Run 1 [switched to sequential execution after parallel data loads were killed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r01-20261003T075109Z#step-12).

### Poor experiment design

In run 1, Fable [changed both weight decay and learning rate](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r01-20261003T075109Z#step-14), improving validation NDCG from about 0.0309 to 0.0322, then [credited weight decay](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r01-20261003T075109Z#step-15). Run 5 made a similar attribution: it [doubled training from six to twelve epochs while also raising the learning rate](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-13), then [attributed the worse result to longer training](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-21). These comparisons could identify a better combined configuration, but could not establish which change caused the improvement or decline.

A further experiment could compare training with and without the full eligibility masks. This is an opportunity for exploration, not an established design error. The training objective used items that would be excluded from final ranking. Run 3 [constructed lists of previously seen items but did not use them to mask the training scores](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-5). Run 4 [masked only the retained history of up to 200 positive-listen items](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r04-20261003T075109Z#step-6). Final ranking applied the full eligibility masks. Training against those extra items might help or hurt; the traces do not establish which.

### Poor time management

In run 3, Fable [said there was no time to revalidate the changed configuration](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-20), then [finished after 32.6 minutes while estimating that about 53 minutes had passed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-23). It left roughly 27 minutes unused after ruling out the validation check for lack of time.

### Misinterpreting results

In run 3, Fable justified changes to its final configuration by [pointing to a similar training-loss curve and describing the changes as hyperparameters only](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-23). Yet the vocabulary cutoff changed which items could be represented in a user's history, and the other settings changed how the model learned. Similar training losses could not establish similar recommendation quality.

Run 2 showed that Fable could make this distinction: when [validation quality declined while training loss fell](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-20), it [validated a shorter schedule](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-21) and [refit those settings](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r02-20261003T075109Z#step-22).

### Insufficient validation

In run 3, Fable measured [public-validation NDCG@10 of 0.0217](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-12), then changed the batch size, vocabulary cutoff, learning rate, and epoch count to make training faster. It launched validation and the final refit together, but [stopped validation when the two jobs approached the memory limit](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-17). It submitted the revised model anyway; its [documentation acknowledged that the submitted configuration had no validation score](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r03-20261003T075109Z#step-20). The earlier score could not establish the quality of the configuration it delivered.

### Insufficient problem exploration

Run 5 drew reassurance from an incomplete baseline comparison. Its [baseline script was killed after reporting lifetime popularity at 0.0107, before completing the recent-popularity measurements](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-11). Fable then [described its model at about 0.022 as roughly twice as good as popularity](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/fable51-high-2-1-288-base-r05-20261003T075109Z#step-12). That claim held for lifetime popularity, but left unanswered whether the model beat the recent-activity baselines it had intended to test.

</details>

<details markdown="1">
<summary><strong>Sol</strong></summary>

Sol sometimes tuned a neural model before establishing what simple recent-activity signals could achieve. Run 2 also contained two mismatches between intent and implementation: its hard-negative term emphasized easy negatives, and recent-history windows grew during the final refit.

### Poor resource management

Several runs repeatedly decompressed a whole eligibility array for individual lookups. In run 1, Sol initially treated a stalled diagnostic as a slow history scan. After it had run for 217 seconds, [interrupting it exposed repeated NumPy archive reads](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r01-2026-09-18#step-28). [Loading the array once let the repaired diagnostic finish in 16.5 seconds](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r01-2026-09-18#step-29).

Run 2 [stalled on the same mistake](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-25) before [caching the array](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-26). Run 3 [repeated the archive reads during its final eligibility audit](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r03-2026-09-18#step-66) before [switching to a vectorized check](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r03-2026-09-18#step-67).

### Insufficient problem exploration

In run 1, a [ten-day item-activity baseline scored 0.0298](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r01-2026-09-18#step-62), compared with [0.0314 when blended with the neural score](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r01-2026-09-18#step-66). Much of the final quality was already present in that simple baseline. Sol discovered the strength of recent activity only after building a [neural model that scored 0.0052](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r01-2026-09-18#step-57). Testing the baseline earlier would have given it a stronger reference for deciding what the neural model contributed.

Run 2 [described popularity alone as demonstrably weak](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-54) after testing the catalog-popularity baseline. It explored popularity coefficients and [retained a useful novelty prior](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-60), but did not compare alternative global recent-popularity or engagement-based trends. Its user-history features already included seven-, thirty-, and ninety-day counts; the missing comparison concerned global item signals. A weak lifetime-popularity baseline did not establish that those signals would be unhelpful. Run 3 [tested a strong recent-item baseline early](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r03-2026-09-18#step-24), showing that this exploration was practical.

**Further exploration opportunities.** Sol [finished run 4 after 19.5 minutes](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r04-retry01-2026-09-18#step-60) and [run 5 after 23.7 minutes](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r05-2026-09-18#step-75), leaving roughly 40 and 36 minutes of the one-hour budget unused. Both attempts had working submissions saved. Early completion alone does not establish poor time management, but the remaining budget offered an opportunity to investigate another feature or modeling idea.

Other modeling alternatives also remained untested. Run 2 [trained another seed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-55) and [combined the models](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-73); run 3 tried an ensemble and [correctly rejected it when quality fell](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r03-2026-09-18#step-66). Those experiments explored initialization and model combination, but did not test a different learning objective.

None of the five reviewed attempts tested a variational encoder with KL regularization, or reconstruction pretraining followed by temporal fine-tuning. Under the same broad task constraints, Astra [implemented KL regularization](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-29) and [tested a variational model](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-33). It also [initialized temporal fine-tuning from a reconstruction-trained model](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-18) and [measured the result](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-35). These approaches provide concrete ideas for follow-up experiments; the traces do not establish that they would improve Sol's results.

### Implementation errors

Run 2's hard-negative term emphasized the easiest negative. The [code defined the margin as the positive score minus the negative score, then selected the largest margin](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-39). That selects the lowest-scoring negative; selecting the smallest margin would target the hardest one. The ordinary pairwise loss still trained against all sampled negatives, but the extra term did the opposite of its stated purpose. The error remained in the final implementation.

Run 2's seven-day history feature grew to cover roughly fourteen days during the final refit. Its [history builder added validation-week events to cached seven-, thirty-, and ninety-day counts computed at the original training cutoff](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-39), without moving the old windows forward. All three windows therefore grew by a week. Incorporating validation events into the final fit was required; recomputing recency relative to the new cutoff was missing. The [final schema and eligibility checks passed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-77), but could not detect that the features had changed meaning. The windows needed to be recalculated from the cutoff used in each stage.

### Misinterpreting results

In run 2, Sol added historical organic-like supervision while also increasing training from [15 epochs](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-42) to [25 epochs](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-48). Public NDCG improved from about 0.0133 to 0.0144, and Sol [credited the added supervision](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-sol-high-cli-0-153-4-r02-2026-09-18#step-54). The combined recipe improved, but the comparison could not establish how much of the gain came from likes rather than longer training.

</details>

<details markdown="1">
<summary><strong>Astra</strong></summary>

Astra's clearest gaps were in running experiments reliably and keeping track of their results. Memory failures interrupted work, completed training was lost before checkpointing, and one final selection overlooked a higher validation score.

### Poor resource management

Four of the five Astra runs hit memory limits. Run 5's container recorded [seven out-of-memory kills](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-77) as Astra repeatedly relaunched work. An initial scoring-buffer repair was insufficient; [preallocating both the base and working score buffers](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-89) finally allowed the search to complete.

These failures also interrupted other experiments. In run 4, a [calibration job exhausted memory and a concurrently training variational model was also killed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r04-2026-09-10T17-31-17#step-64). Run 3 [lost a tuning process while several jobs were active](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r03-2026-09-10T16-05-15#step-33), then encountered further calibration failures before [rewriting scoring to reuse preallocated buffers](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r03-2026-09-10T16-05-15#step-67). A training job fitting by itself did not establish that training, calibration, and refitting could fit together.

### Poor experiment tracking and reproducibility

In one run, Astra lost completed training because it saved checkpoints after evaluation. Two jobs [finished a training epoch and then died during evaluation without saving their weights](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/astra-high-0.153.4-aws-gap-r01-2026-09-24#step-112). Astra [moved checkpoint saving before evaluation and reran the work](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/astra-high-0.153.4-aws-gap-r01-2026-09-24#step-114). The earlier submission survived, and the final refit succeeded, but the lost training had to be repeated.

In run 5, intermediate refits wrote to the same submission file as the calibrated fallback Astra intended to preserve. A [completed warmup refit replaced it with its own predictions](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-70). Astra later [gave experiments separate output filenames and regenerated the calibrated rankings from saved vectors](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-86). It saved a separate backup, restored the intended ranking to the submission path, and checked the final submission. The shared filename had allowed an unselected intermediate result to overwrite the fallback.

Evaluation sometimes began before the predictions were ready. In run 1, Astra launched the public evaluator while a new ensemble was still exporting. It [read the previous model's score of 0.0455](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-91); Astra caught the discrepancy and [reran it to obtain the new ensemble's 0.0548](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r01-redo-01-2026-09-09T00-00-30#step-94). Run 5 likewise [started evaluation before its prediction file existed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-111). Both runs subsequently verified their selected validation rankings.

### Implementation errors

Run 3 initially gave some users another listener's representation. Its [user lookup returned index -1 for users missing from the aggregated histories, then used that index to select a vector](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r03-2026-09-10T16-05-15#step-11). In NumPy, `-1` selects the last row. Users without usable catalog history therefore received the final represented user's vector. A ranking could still have the correct number of rows and eligible items while making this mistake.

Astra reached export before [identifying seven validation users and six test users affected by the missing-history case](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r03-2026-09-10T16-05-15#step-101). It [added an explicit empty-history path and refreshed the cached vectors](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r03-2026-09-10T16-05-15#step-104), repairing the final output. Run 4 [caught a related missing-user assumption during its first validation pass](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r04-2026-09-10T17-31-17#step-16).

### Misinterpreting results

In one run, Astra overlooked a newer, higher validation score when choosing its final model. A calibration of the existing single model [scored 0.053607](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/astra-high-0.153.4-aws-gap-r01-2026-09-24#step-124). Later, a version with a small neural network added to the item tower scored 0.053390. Astra [described it as matching the best single-model result](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/astra-high-0.153.4-aws-gap-r01-2026-09-24#step-132) and selected it, apparently comparing against the older score of 0.053387. Both candidates used 601-dimensional vectors, and Astra gave no reason for choosing the lower-scoring one. The difference was small, but the final comparison still needed to account for the newer result.

### Insufficient validation

Run 5 let a [changed default of 2,048 sampled negatives](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-19) carry into final refitting, although the original warmup had used full-catalog softmax. Astra [restored full softmax for the warmup](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-98) and [repeated both final training branches from the corrected checkpoint](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/codex-astra-high-base-r05-2026-09-10T17-31-17#step-107), correcting the submitted model's recipe at the cost of retraining.

</details>

<details markdown="1">
<summary><strong>Grok 4.6</strong></summary>

Grok tried several modeling approaches, but recurring implementation and tracking errors weakened the process. Features sometimes changed meaning between training and prediction, historical examples depended on later events, and one run left a better validation candidate unsubmitted.

### Implementation errors

Run 2's use of position embeddings could not encode track order. Grok [added position embeddings to the tracks in a user's recent history, then immediately averaged them](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-43). Swapping two tracks leaves that average unchanged: the same track embeddings and the same position embeddings are still being added together. This channel could identify which tracks were in the recent history, but could not distinguish their order.

Run 5 used different preference features during training and prediction. Its [training histories retained likes and dislikes after they had been reversed, while prediction used the supplied active preference states](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-19). For users with no active preferences, the prediction code fell back to raw historical actions, potentially bringing canceled preferences back. It also selected different parts of the history: training kept recent actions, while prediction capped the supplied preferences at 64 likes and 32 dislikes in item-ID order. For histories exceeding those caps, it kept the largest IDs rather than the most recent preferences.

Other features did not measure what they were intended to represent. In run 3, the [average playback-completion feature counted both likes and dislikes as 100% completed listens](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-20). A dislike could therefore raise a feature meant to describe how much of a track people listened to. In run 1, the [organic-history channel used each track's last positive event to decide whether it was organic, while its count included both organic and recommended events](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r01-61e7dce0-20260929#step-25). A later recommended listen completed at least 50% could remove the track from that channel, even if the user had previously sought it out.

Run 5's target construction also [discarded a later organic like when a strong organic listen for the same track had already entered the target window](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-19). The target kept the weaker listen weight instead of receiving the higher weight Grok assigned to likes.

### Poor resource management

Several Grok jobs were killed by the environment, apparently because of excessive memory use. Grok then tried to fix these failures by reducing memory consumption. In run 4, the [first smoke test failed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-18) with code that expanded eligibility into sets containing roughly [29,000 Python integers per user](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-20). It replaced those sets with compact bit-mask checks. In run 5, a process [was killed](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-33), after which Grok [replaced full-matrix ranking with chunked scoring and reduced other allocations](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-34). Run 2 was also [killed while preparing the final refit](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-54). Grok [separated stages into fresh processes](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-56) so memory from the earlier training stage could be released before the refit. These repairs recovered the pipelines, but only after failed attempts had consumed part of the hour.

### Poor experiment tracking and reproducibility

Run 1 [backed up the predictions and metrics from its 0.0303 model](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r01-61e7dce0-20260929#step-35), but allowed a [weaker experiment scoring 0.0287 to overwrite the checkpoint](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r01-61e7dce0-20260929#step-37). Restoring the earlier ranking did not restore the weights that produced it. Run 4 also [could not reload an older checkpoint after changing the model definition](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-45): it had kept the weights without preserving compatible code.

The final instructions did not consistently identify the selected experiment. Run 4's [README described its initial architecture and left the validation score pending](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-23), while the [selected model used a different training sequence and history configuration](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-62). Following its documented command would not reproduce that selection.

### Poor time management

In run 3, Grok [started a training job about 54 minutes into its one-hour allowance](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-63). Loading data and pretraining consumed most of the remaining time. With less than a minute left, the job allocated [fourteen minutes to fine-tuning](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-65) and timed out.

Run 5's time accounting had a specific defect: restarting a training script gave it a new time allowance. About 47 minutes into its one-hour allowance, the [training log reported roughly 41 minutes remaining before refitting](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-50), although only about 13 minutes remained. The deadline needed to stay tied to the start of the attempt.

### Misinterpreting results

Run 3 built an explanation around a broken baseline. Its [organic-popularity ranking scored 0.000015, compared with 0.0107 for popularity across the full training history](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-16). The organic counts were unsigned integers; negating them to sort in descending order put zero-count tracks ahead of tracks with positive counts. Grok [noticed that the result was suspicious, decided the code appeared correct, and attributed it to candidate eligibility](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-17). It moved on with the sorting error still present, treating a broken ranking as evidence about the data.

In run 4, Grok [treated a score near 0.0142 as evidence that the two-tower approach was near its ceiling, citing the small number of relevant items among thousands of candidates](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-46). NDCG is normalized against the ideal ranking, so sparse relevance alone does not establish a low ceiling. Grok's later two-tower model [reached 0.0189](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r04-61e7dce0-20260929#step-61).

### Delivery failures and task violations

Run 3 had a better validation candidate but never turned it into a submission. It [reached 0.0123 on validation, above its 0.0107 popularity baseline](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-45), then [stopped the job and changed approach before exporting that model's submission](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r03-61e7dce0-20260929#step-46). The later pipeline timed out, leaving the original fallback. The saved neural checkpoint never became the delivered ranking.

Run 5 [deferred the required README until the final minutes](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r05-61e7dce0-20260929#step-53), and the trace ends without it being written.

### Insufficient validation

In run 2, Grok [combined the selected validation model with its refitted version](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-62) only when preparing the final submission. It had not tested whether this ensemble recipe improved ranking quality. The [reported public score of 0.0194](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-68) still belonged to the earlier single model. The ensemble needed to be evaluated during model selection, before validation events became training data.

### Information leakage

In run 2, Grok [kept each track's last positive listen across the full training history before applying the cutoff for each historical training example](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-17). If a user played a track on days 1 and 8, a day-7 example would omit it because only the day-8 occurrence survived. That later replay could therefore change the earlier history and make a familiar track appear to be a new discovery. The [same logic remained in the revised sample builder](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/grok-1-0-44-base-r02-61e7dce0-20260929#step-50). Public validation still used training data only; the leakage was within the historical examples used to train the model. Events needed to be filtered to each cutoff before deduplication and label construction.

</details>

<details markdown="1">
<summary><strong>Gemini 3.8 Flash</strong></summary>

Gemini's main failure was turning experiments into a finished model. All five attempts exhausted the one-hour budget and left the initial popularity fallback as the submission, each receiving a hidden NDCG@10 of 0.009092. The traces also show errors in evaluation and time spent tuning an approach the task prohibited.

### Poor time management

Run 5 kept searching until there was no time left to deliver a trained model. An experiment [launched about 53 minutes into the attempt reached 0.0184 on validation](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r05-3d6fca60#step-79). With roughly five minutes left, Gemini [reran training with a different negative sampler](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r05-3d6fca60#step-81). With about two minutes left, it [started another training run with a fixed mixing weight](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r05-3d6fca60#step-85). Those final minutes went to more experiments while refitting and exporting a neural submission remained unfinished.

### Delivery failures and task violations

Run 4 shows the delivery gap most clearly. Gemini [saved a popularity fallback early](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r04-3d6fca60#step-43). A later neural model reached [0.0296 on validation at epoch four](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r04-3d6fca60#step-76), and its code saved the best weights. Gemini then [prepared another experiment](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r04-3d6fca60#step-77) instead of completing the selected model's refit and export. Its best weights were saved, but the delivered ranking still came from the initial baseline.

Run 2 spent part of its budget tuning a method the task explicitly prohibited. Gemini [constructed a global user–item interaction matrix and derived item co-occurrence scores from it](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r02-3d6fca60#step-67), then tuned the method to [0.0334 on public validation](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r02-3d6fca60#step-80). The task prohibited constructing or decomposing a global interaction matrix and required separate trainable neural towers. Its [subsequent SVD approximation](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r02-3d6fca60#step-81) still did not meet those requirements. Gemini eventually returned to neural training, but no trained model replaced the fallback before the deadline.

### Implementation errors

In run 1, Gemini [decoded candidate eligibility with the wrong bit order](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r01-3d6fca60#step-81). The task required little-endian decoding, but its neural experiments used NumPy's default, changing which items appeared eligible. Its scoring code checked the ranking's structure and calculated NDCG without checking candidate eligibility. Gemini was therefore selecting models using scores from the wrong candidate sets, and the error remained in its final evaluation code.

Run 3's prototype [left dropout active during validation](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r03-3d6fca60#step-57). It disabled gradient calculation but never switched the model to evaluation mode. Randomly dropped activations therefore changed the user vectors during scoring, adding noise to its comparisons between epochs.

### Misinterpreting results

Run 1's baseline comparison produced a discrepancy that should have prompted a check of the ranking code. The original popularity baseline scored 0.0107, while a [normalized popularity calculation scored 0.0054](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r01-3d6fca60#step-56). Gemini [attributed the drop to rounding and tie-breaking](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r01-3d6fca60#step-58). But the second calculation also used the wrong eligibility mask, so the two rankings were not even choosing from the same candidate sets. Its explanation left that more basic error unresolved.

### Poor experiment tracking and reproducibility

In run 3, a prototype reached [0.0226 at epoch two](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r03-3d6fca60#step-63), then continued through eight epochs without saving the best checkpoint. When Gemini later [reran the prototype to recover the result](https://verimium.com/seniormle-bench/yambda-discovery-ranking-two-tower-base/attempts/gemini-3-8-flash-high-base-fixed-r03-3d6fca60#step-80), its best score was 0.0195. The original weights were gone. Saving the checkpoint when the score improved would have preserved the candidate for comparison, refitting, and export without relying on another training run to recreate it.

</details>
