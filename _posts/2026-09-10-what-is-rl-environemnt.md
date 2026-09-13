---
layout: post
title: "What is RL Environemnt"
description: A simple explanation of the worlds agents learn in.
image: /assets/images/what_is_rl_env.png
---

![An illustration explaining an RL environment.](/assets/images/what_is_rl_env.png)

There has been a lot of talk about RL environments recently. Here is a simple explanation of what they are.

In supervised machine learning, we started with labels: one item, one answer. This was a clean fit for single-turn questions like “Is this an image of a cat or a dog?” But when models had to make sequences of decisions, like writing a full sentence or solving a coding task, feedback and labeling became harder. Instead of labeling one answer, we needed to generate an entire trajectory, and the reward might arrive only at the end. It is unclear which actions deserve credit. This led to 1) a move to reinforcement learning instead of supervised learning, 2) a need for many more trajectories to find the right action that deserves the reward, and 3) longer training runs, with labeling becoming much, much more complex and expensive. The RL environment is a solution to this problem.

An RL environment is a sandboxed world the agent can act in—a terminal, browser, codebase, or spreadsheet—plus the machinery that executes its actions and evaluates the outcome. The evaluation is the critical part: the environment must return a reward that accurately reflects whether the agent actually succeeded, and the more detailed the reward, the better. Building this reward signal is often the hardest part, as it requires expertise in each domain. Once an RL environment for a task is available, the agent can try it many times, get feedback on its actions, and update its policy.
