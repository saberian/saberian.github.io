---
layout: post
title: "Why program LLMs in words?"
description: Prompts and reasoning can move beyond readable text. What do we gain, and what do we lose when human oversight matters?
stylesheets:
  - /assets/prompt-lab.css
modules:
  - /assets/prompt-lab.js
---

We can think of an LLM as a general-purpose computer and its prompt as a program. The same model can summarize an article, classify a support ticket, or translate a sentence without changing its weights. What changes is the input that specifies the task. This is a useful perspective, not an equivalence to conventional software: a prompt does not have the precise execution guarantees of a Python function. Still, it raises a question: **if a prompt is a program, why must we write it in words?**

There is a useful parallel in hardware. Some chips can be configured to implement different circuits without manufacturing a new chip. These are called *field-programmable gate arrays*, or FPGAs. A configuration specializes the chip; a prompt specializes the behavior of a reusable model. The analogy has a limit: an [FPGA configuration controls logic and routing](https://docs.amd.com/r/en-US/ug1291-viv/FPGA-Architecture), while a prompt changes neither the model's weights nor its physical wiring. It changes the activity produced as the model processes the input.

## What the words do

The model first splits text into tokens and maps each token to an embedding: a vector of numbers. These representations are transformed through successive layers. In an [attention head](https://arxiv.org/abs/1706.03762), dot products between *query* and *key* vectors produce scores. After scaling, a softmax turns those scores into weights, which determine how *value* vectors are mixed. Prompt tokens participate in this computation: their representations influence what information later positions receive. Attention is only part of the model, but it makes one thing concrete: instructions affect numerical computation, rather than sitting outside it as a separate specification.

{% include attention-demo.html %}

We can improve those instructions systematically. Start with a task, examples, and a way to measure success; then search for a prompt that performs better. [GEPA](https://arxiv.org/abs/2507.19457), for example, uses an LLM to examine execution traces and feedback, diagnose failures, and propose textual revisions. It evaluates the candidates rather than assuming a better-sounding instruction works better. This turns prompt writing into an optimization problem. But the search still produces text, with the opportunity for a person to review what changed.

## A program without words

Why stop at text? A textual prompt selects vectors from the model's vocabulary embedding table. *Soft-prompt tuning* instead learns a small set of input vectors directly, using gradient descent on task examples while leaving the model weights frozen. [Lester et al. (2021)](https://aclanthology.org/2021.emnlp-main.243/) demonstrate this input-level approach. The related [prefix-tuning method of Li and Liang (2021)](https://aclanthology.org/2021.acl-long.353/) learns continuous prefixes across model layers. Neither requires the learned prompt to be a sentence. The task-specific program is now a set of trainable numbers.

This can work remarkably well. On SuperGLUE, a benchmark of language-understanding tasks, Lester et al. found that soft-prompt-tuned T5-Large outperformed few-shot GPT-3 with 175 billion parameters, despite being over 220 times smaller. That comparison used different models and supervised prompt training versus few-shot examples; it does not prove that vectors always beat words on the same model. It does show that a small learned prompt can be a powerful way to specialize a frozen model.

A learned vector need not coincide with any vocabulary embedding. Replacing it with the nearest token changes the input, and need not preserve its behavior. In their interpretability analysis, Lester et al. found meaningful clusters of nearby words for individual vectors, but little interpretability in the prompts as sequences. We gain freedom to optimize, but lose the familiar ability to read an instruction, question a clause, or revise its intent. Even text optimization must preserve readability deliberately: a sequence of valid tokens is not necessarily an understandable instruction.

{% include embedding-demo.html %}

## The same choice inside the reasoning

This tension also appears in reasoning traces. [DeepSeek-R1's report](https://arxiv.org/html/2501.12948v1#S2.SS3.SSS2) describes poor readability and language mixing in R1-Zero. During R1 training, the team introduced a language-consistency reward and reported a slight performance reduction in exchange for readability. This does not mean models generally prefer gibberish: mixing languages is not the same as producing meaningless text. It means optimizing for correct answers does not automatically optimize for a human reader. [Coconut](https://arxiv.org/abs/2412.06769) takes a different approach, demonstrating intermediate reasoning with hidden-state vectors fed back into the model instead of decoded words.

Readable instructions let us inspect what we ask a model to do; readable reasoning traces can help us check its stated steps and detect concerning behavior. That makes retaining useful visibility an AI-safety objective, not just a writing preference. But **readability is not faithfulness**. A fluent trace may omit important influences or fail to reflect how the answer was produced. Research on [chain-of-thought monitorability](https://arxiv.org/abs/2507.11473) therefore treats monitoring as a valuable additional safety layer, not a complete verification method. Making a trace look reassuring is not the same as making the system easier to oversee.

The lesson is not that every computation should be translated into prose, or that language always costs performance. It is that performance is not our only objective. We can optimize prompts outside language and develop reasoning that does not proceed entirely through words. When human oversight matters, however, we may deliberately retain readable instructions and monitorable reasoning. **Language is a constraint on how we program LLMs, but it is also one way we keep those programs open to human review.**
