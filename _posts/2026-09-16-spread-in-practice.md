---
layout: post
title: "How consistent are coding agents at building a music recommender?"
permalink: /blog/how-consistent-are-coding-agents/
redirect_from: /blog/spread-in-practice/
hide_title: false
description: Measuring variation across coding-agent attempts at a two-tower music-recommendation task.
image: /assets/images/spread-scores.png
math: true
---

<picture>
  <source media="(max-width: 600px)" srcset="{{ '/assets/images/spread-scores-mobile.svg' | relative_url }}" width="380" height="430">
  <img src="{{ '/assets/images/spread-scores.svg' | relative_url }}" width="1040" height="300" alt="Final hidden NDCG@10 scores for 19 task attempts across three model/harness groups.">
</picture>

*Figure 1. Each dot represents one independent attempt with a 60-minute budget. Higher hidden-test NDCG@10 scores are better.*

In the previous posts, we introduced [RL environments]({% post_url 2026-09-10-what-is-rl-environemnt %}). Here, we look at a concrete example: an RL environment we built for a coding agent (model + harness) to develop a music recommender system.

The task is to build a PyTorch two-tower model. One tower represents the listener, the other represents each track, and their dot product scores how well they match. We chose this constraint to reflect a realistic approach to the large-catalog retrieval challenges most companies face: track embeddings can be computed offline and stored in an approximate nearest-neighbor (ANN) index. At request time, the system computes the listener's embedding and retrieves tracks with high dot-product scores.

The environment provides user listening histories and a set of eligible tracks for each user. The agent has 60 minutes in an 8-vCPU, 8 GB container with no GPU, running on an AWS c6a.2xlarge instance. The agent can inspect the data, train models, and use a public validation dataset to choose its next experiment. Running the reference solution end-to-end takes about six minutes on that hardware, including data preparation, training, validation, ranking, and evaluation. This leaves room for several experiments, although agents also need time to write code and plan.

The environment's automated verifier checks the submitted ranking's format and coverage. It then evaluates ranking quality on hidden test data using NDCG@10. This metric rewards placing relevant tracks near the top of the list.

## Agent performance

We tested three model/harness pairings, all at high reasoning effort: five attempts with Opus 5.0 on Claude Code 2.1.251, ten with GPT-5.6-Sol on Codex CLI 0.147.0, and five with Grok 4.6 on Grok Build 1.0.5.

Nineteen of the 20 attempts produced structurally valid submissions. One Grok attempt did not submit a solution by the deadline. We manually reviewed the trajectories of all 20 attempts to understand how the agents approached the task.

The agents averaged **5.3 scored experiment rounds per attempt**. A round is an experiment or grouped parameter sweep that produced recommendation-quality feedback. These rounds include evaluating baselines and tuning existing solutions, not just training new models. Some agents ended their sessions early, declaring the task complete. The following table shows the average number of scored rounds and elapsed time for each pairing. Elapsed time includes coding, data preparation, debugging, and final refitting.

| Model / harness | Avg. scored rounds | Avg. time used |
| --- | ---: | ---: |
| Opus 5.0 / Claude Code | 3.4 | 24.4 min |
| GPT-5.6-Sol / Codex | 6.8 | 40.8 min |
| Grok 4.6 / Grok Build | 4.2 | 56.0 min |
{: tabindex="0" aria-label="Agent experiment rounds and time use" }

As Figure 1 shows, even with the same model and harness, the recommendation quality of the submitted solutions varied substantially. For example, Opus's hidden NDCG@10 ranged from about 0.0092 to 0.0304, more than a threefold difference.

Ten of the 19 scored attempts exceeded the reference solution's NDCG@10 of about 0.018. The agents could produce strong solutions but did not do so consistently. Ideally, an agent should consistently produce solutions that meet or exceed the reference's quality.

How much of this variation might come from randomness in model training? To establish a point of comparison, we ran the same reference implementation with ten different training seeds. Hidden NDCG@10 ranged from 0.0148 to 0.0212, with a mean of 0.01766 and a sample standard deviation of approximately 0.00217 ($$\sigma_r$$).

For each model/harness pairing, we also look at the sample standard deviation of its final submission scores ($$\sigma_a$$). Following the [previous post]({% post_url 2026-09-13-what-makes-a-good-rl-task %}), we define spread as:

$$
S = \frac{\sigma_a}{\sigma_r}
$$

| Model / harness | Spread |
| --- | ---: |
| Opus 5.0 / Claude Code | 4.25× |
| GPT-5.6-Sol / Codex | 2.12× |
| Grok 4.6 / Grok Build | 2.87× |
{: .table-centered tabindex="0" aria-label="Spread by model and harness" }

As the table shows, agent scores varied substantially more than the fixed reference did across training seeds: their sample standard deviations were roughly two to four times as large. **This suggests that stronger attempts made better decisions and that there is room to improve agents' reliability in making these design and modeling decisions.** In the next post, we will examine the agent trajectories more closely to study those decisions and identify areas for improvement.
