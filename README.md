# Moonrun LLM Onboarding Track

Welcome to Moonrun. This is your first technical assignment. It is designed to take **1–2 weeks**, with a hard ceiling of **4 weeks**. By the end, you will have trained and served two language models from scratch — one on a single GPU, one distributed across multiple GPUs.

The goal is not to make you an LLM expert. The goal is to give you a solid enough foundation that you can participate — everything at Moonrun is built around this technology, and we need you to be able to follow conversations, ask the right questions, and contribute without needing to be caught up every time. The faster you build this foundation, the faster you can do real work.

This is not a course. It is a sequence of things to build. Read what you need, then build.

---

## Timeline

| Week | Focus |
|------|-------|
| 1–2 | Assignment 1 — single-GPU Shakespeare LLM |
| 2–4 | Assignment 2 — multi-GPU LLM with parallelism strategies |

---

## Phase 1: Foundations

### Watch first

**Andrej Karpathy — Neural Networks: Zero to Hero**
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

These videos are long. You can go through all of them, but it will take you some time, so use them as reference and pull out what you need. By the end of the series you will have built a GPT from scratch, character by character, on the Shakespeare dataset.

Key episodes:
- *The spelled-out intro to language modeling* — builds bigram model, introduces loss
- *Building makemore* (parts 1–5) — MLP → BatchNorm → backprop from scratch
- *Let's build GPT* — the full transformer, trained on Shakespeare

### Read next

**The Illustrated Transformer — Jay Alammar**
https://jalammar.github.io/illustrated-transformer/

Best visual explanation of attention that exists. Read it slowly, not skimming.

**Attention Is All You Need (2017)**
https://arxiv.org/abs/1706.03762

Read the architecture section and the multi-head attention equations. You do not need to read the experiments section deeply on the first pass.

**The Illustrated GPT-2 — Jay Alammar**
https://jalammar.github.io/illustrated-gpt2/

Bridges the transformer paper to autoregressive language modeling.

### Reference material (use as needed, not front-to-back)

- **Hugging Face NLP Course** — https://huggingface.co/learn/nlp-course/
  Start at Chapter 1. Use Chapters 2–4 when you hit tokenization or fine-tuning questions.
- **Sebastian Raschka — Build a Large Language Model From Scratch**
  https://github.com/rasbt/LLMs-from-scratch
  Code-first companion to his book. Good when you want to see a clean implementation alongside your own.
- **Lilian Weng's blog** — https://lilianweng.github.io/
  Authoritative deep-dives on specific topics (scaling laws, RLHF, etc.). Use when you need to go deeper on one concept.
- **fast.ai Practical Deep Learning**
  https://course.fast.ai/
  Optional but useful if you want more intuition on training dynamics before starting the assignments.

---

## Assignment 1: Shakespeare LLM — Single GPU

**Goal:** Train a character-level GPT on the Shakespeare dataset and serve it via an HTTP API on a single GPU.

### What you are building

A transformer language model that generates Shakespeare-style text. This is intentionally small — the point is to own every part of the pipeline end-to-end, not to achieve state-of-the-art perplexity.

Once it trains, the interesting part is measuring it: how long a step takes, where that time goes, what the GPU was doing while it ran, and what you can change to make it faster.

### Requirements

**Training**
- Implement the transformer architecture yourself or adapt [nanoGPT](https://github.com/karpathy/nanoGPT). Do not copy-paste blindly — understand every line.
- Dataset: [Shakespeare text](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB, ~40k lines)
- Tokenization: character-level (no BPE required, though you may add it)
- Logging: track train loss and val loss. Plot them.
- Checkpoint: save a checkpoint after training
- Hardware target: **1× L40S GPU** (see [Compute](#compute) below)

**Serving**
- Load your checkpoint and expose a `/generate` endpoint
- Input: a prompt string + max tokens to generate
- Output: generated text
- Use FastAPI. No UI required.
- The server must start cleanly from a fresh clone with documented steps.

### Deliverables

- [ ] `train.py` — training script, runs end-to-end
- [ ] `model.py` — transformer implementation
- [ ] `serve.py` — FastAPI inference server
- [ ] `requirements.txt` or `pyproject.toml`
- [ ] `README_assignment1.md` — documents: how to train, how to serve, your final train/val loss, one sample generation

### Sign-off: Presentation to the Team

When your deliverables are ready, schedule a meeting and invite:
- raghuvir@moonrun.ai
- ram@moonrun.ai
- evgueni@moonrun.ai

Walk through what you built — the architecture decisions, the training run, a live demo of the served model. Your colleagues will tell you whether the work is complete or needs more. **You do not move to Assignment 2 until the team signs off on Assignment 1.**

What to cover in the presentation:

Bring graphs. We are looking for two things: that you understand the model you built, and that you understand how it runs on the GPU. Plots are how you show us both, so label the axes and say what each one led you to conclude.

- **Loss curves.** Train and validation on the same axes. Where does validation stop following train, and what did you do about it? Explain any kink, spike or plateau rather than leaving it on the slide.
- **Step time.** Milliseconds per training step, and where they go: forward, backward, optimizer, data loading.
- **Time to train.** Wall clock for the full run, and what you would change to halve it.
- **What you did to speed it up.** Batch size, mixed precision, warmup, keeping the GPU fed, whatever you tried. Give before and after numbers, including for the things that did not work.
- **What the GPU was actually doing.** Utilization and memory across the run. An idle GPU is the interesting case, so tell us why it was idle.
- **Your code, then a live generation.** Walk the attention path as you wrote it, not the theory, then generate from the served endpoint.
- **What was harder than you expected**, and why.

### Notes

- Target ~6 layers, ~6 heads, ~384 embedding dim. You can go larger if you want, but do not chase scale here.

---

## Assignment 2: Multi-GPU LLM — Parallelism Strategies

**Goal:** Train a larger language model distributed across multiple GPUs using two parallelism strategies, benchmark both, and serve the result with vLLM.

The main focus is the benchmarking. Training the model is the means. What we are after is how it ran on the GPUs, where the bottlenecks were, and what you would do to speed it up.

### Why this matters at Moonrun

Moonrun operates at datacenter scale. Most of the work is finding out why a run is slower than the hardware allows, and then fixing it. This assignment is a small version of that.

### Background reading (required before starting)

**FSDP (Fully Sharded Data Parallel)**
- PyTorch FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

**Tensor Parallelism**
- Megatron-LM paper: https://arxiv.org/abs/1909.08053
- Practical intro: https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism

**Pipeline Parallelism**
- GPipe paper: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377

**Useful survey**
- https://huggingface.co/docs/transformers/perf_train_gpu_many — covers all strategies with diagrams

### What you are building

A language model (your choice of dataset — OpenWebText, books corpus, or continue with Shakespeare at larger scale) trained across **at least 2 GPUs** using **two of FSDP, tensor parallelism and pipeline parallelism**.

### Requirements

**Training**
- Pick two of the following three and implement both:
  - **FSDP**
  - **Tensor Parallelism**
  - **Pipeline Parallelism**
- Write a short comparison: wall-clock time per step, GPU memory usage, and MFU (model FLOP utilization) for each strategy
- Checkpoint the final model

**Serving**
- Serve using [vLLM](https://github.com/vllm-project/vllm)
- Use tensor parallelism in vLLM (`--tensor-parallel-size N`)
- Expose a `/v1/completions` compatible endpoint
- Document startup command

**Hardware target: 2× or 4× L40S GPUs, your choice** (see [Compute](#compute) below)

### Deliverables

- [ ] `train_distributed.py` — distributed training entry point
- [ ] `model.py` — model definition compatible with your parallelism approach
- [ ] `serve_distributed.py` (or vLLM launch script)
- [ ] `benchmark.md` — side-by-side comparison of your two parallelism strategies:
  - GPU memory per device
  - Steps/sec or tokens/sec
  - Notes on what you had to change when switching strategies
- [ ] Updated `README_assignment2.md` — how to launch training, how to launch serving

### What we will check

We are not assessing the quality of the model. Loss values and sample text are not the point here. What we want to see is that you understand how the thing performs on the hardware.

- Which of your two strategies performed better, and why. This is the question we care about most.
- How it could be improved.

### Sign-off: Presentation to the Team

Same process as Assignment 1. Schedule a meeting and invite:
- raghuvir@moonrun.ai
- ram@moonrun.ai
- evgueni@moonrun.ai

Present to the team and get explicit sign-off before the assignment is considered complete. **You do not submit your onboarding as done until the team signs off on Assignment 2.**

What to cover in the presentation:

Same bar as Assignment 1, now across GPUs. The goal is a richer presentation than the first one: more to show, and judgement about what is worth showing. Focus on the key points rather than walking us through everything you did.

- **Loss curves for both strategies**, on the same axes.
- **Step time for each strategy**, and where that time goes.
- **Throughput and MFU.** Tokens per second and time to train for each strategy, against the hardware peak.
- **Scaling.** Step time and throughput at 1, 2 and 4 GPUs. Quantify the speedup you did not get, and account for it.
- **Memory per device** for each strategy, and why the two differ.
- **How you split the model.** A diagram per strategy showing what lives where and what crosses the wire.
- **Which strategy won, and why.** Then how you would improve it: the change you would make first, and the number you expect it to move.
- **A live request to the vLLM endpoint**, and what you would do differently at 10× the scale.

---

## Compute

Use whichever cloud provider you like. L40S instances are cheap and widely available, and that is the card both assignments are sized for. nebius.com has them. Pick whoever gives you the best price and the fastest path to a running machine.

**We reimburse your compute costs.** Keep the receipts handy in case we ask. Both assignments together should land somewhere between 10 and 50 dollars on spot pricing; well above that usually means something is running when it should not be.

Sizing:
- Assignment 1: 1x L40S
- Assignment 2: 2x or 4x L40S, your choice. More GPUs means more interesting parallelism configurations, but 2 is enough to meet every requirement.

Practical notes:
- Spot and preemptible instances are much cheaper and fine for this work. Checkpoint frequently so a reclaim costs you minutes, not days.
- Stop the instance when you are not actively using it. Never leave a GPU running overnight unattended.
- If L40S is unavailable wherever you look, anything with comparable memory works. Note which card you used in your README, since your benchmark numbers depend on it.

---

## How to Submit

Push your work to a repo under the `Moonrun-ai` GitHub organization: `Moonrun-ai/llm-onboarding-<yourname>`.

When ready for review, open a PR and send us the link. Include in the PR description:
- What surprised you
- What you would do differently
- One thing you still do not fully understand (honesty is valued here)

---

## If You Get Stuck

In order:
1. Re-read the relevant background material
2. Add print statements and inspect shapes, dtypes, and loss values
3. Search the linked repos' GitHub Issues
4. Ask a colleague — but come with a minimal reproduction, not "it does not work"

The goal is not to finish fast. The goal is to understand deeply enough that you can debug problems you have never seen before. That is the job.
