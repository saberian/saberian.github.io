---
layout: post
title: "What makes an RL task useful for an agent?"
hide_title: true
description: A practical first screen for useful reinforcement learning tasks.
image: /assets/images/rl-task-cover.png
math: true
---

![An AI agent considering several homework assignments.](/assets/images/rl-task-cover.png)

In [the previous post]({% post_url 2026-09-10-what-is-rl-environemnt %}), we discussed RL environments and how they let agents act and receive feedback. Here, we look at what makes a task useful for RL training.

Ultimately, a task is valuable if including it in training improves performance on a held-out evaluation set, compared with an otherwise matched training run without that task. But testing every task this way requires repeated RL fine-tuning and evaluation, which is often prohibitively expensive. For many closed models, we also lack access to the weights or a suitable RL training interface. We need a more affordable way to identify promising tasks.

To develop one, we can look at how reinforcement learning uses feedback from an environment. One approach is Group Relative Policy Optimization, or GRPO. During training, GRPO samples several independent attempts at the same task from the current model. It collects their rewards and calculates an advantage estimate for each attempt:

$$
A_i = \frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)}
$$

Here, $$r_i$$ is the attempt’s reward, and the mean and standard deviation come from the group’s rewards. For a group with nonzero reward variation, attempts above the average receive positive advantages and those below it receive negative advantages. The reward term in training encourages higher-scoring trajectories and discourages lower-scoring ones. This is the outcome-supervision formulation introduced in [DeepSeekMath](https://arxiv.org/html/2402.03300v3#S4.SS1.SSS2).

If every attempt receives the same reward, the group provides no relative reward signal. When rewards differ, GRPO has a contrast it can use. However, random reward fluctuations can also produce positive and negative advantages. In general, observed reward variation combines differences in approach with randomness in execution. What matters is **how reliably those advantages reflect the quality of the agent’s decisions**.

To investigate these sources of variation, we propose a simple screening measure called **spread**. First, run independent attempts with the same model and harness, holding the task version, evaluation, resource budget, and seeds used inside the task fixed. For example, these seeds might control random initialization during model training. The agent should still sample its actions independently across attempts. Measure the standard deviation of the resulting rewards, $$\sigma_a$$.

Then rerun a fixed reference solution with different random seeds, keeping its code, data, evaluation, and compute fixed. Measure the standard deviation of those rewards, $$\sigma_r$$. This estimates how much reward variation an unchanged solution produces through randomness alone. We define the spread ratio as:

$$
S = \frac{\sigma_a}{\sigma_r}
$$

If the reference reward's standard deviation is zero, the ratio is undefined. If it is very small or estimated from too few runs, the ratio can be unstable; inspect the reward distributions and uncertainty before interpreting it.

A ratio substantially above one suggests that agent attempts differ by more than the variation observed when reseeding the reference solution. If that reference provides a representative noise baseline, the ratio supports the interpretation that reward differences reflect meaningful differences in approach.

This compares variation between agent approaches under fixed task randomness with the seed sensitivity of one reference solution. Other solutions may respond differently to random seeds, and limited samples make both estimates uncertain. Inspecting trajectories and rerunning representative agent solutions helps check whether higher rewards correspond to reliably better decisions.

We therefore treat spread as an inexpensive first screen for promising tasks for a particular model and harness. We can measure it without changing the model’s weights. Confirming that those tasks improve the model still requires RL fine-tuning and held-out evaluation.
