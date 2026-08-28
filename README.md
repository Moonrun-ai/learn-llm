# Moonrun LLM Onboarding Track

Welcome to Moonrun. This is your first technical assignment. It is designed to take **1–2 weeks**, with a hard ceiling of **4 weeks**. By the end, you will have trained and served two language models from scratch — one on a single GPU, one distributed across multiple GPUs.

The goal is not to make you an LLM expert. The goal is to give you a solid enough foundation that you can participate — everything at Moonrun is built around this technology, and we need you to be able to follow conversations, ask the right questions, and contribute without needing to be caught up every time. The faster you build this foundation, the faster you can do real work.

This is not a course. It is a sequence of things to build. Read what you need, then build.

---

## Timeline

| Week | Focus |
|------|-------|
| 1 | Foundations — read, watch, absorb the basics |
| 1–2 | Assignment 1 — single-GPU Shakespeare LLM |
| 2–4 | Assignment 2 — multi-GPU LLM with parallelism strategies |

---

## Phase 1: Foundations

Do not skip this. The assignments will not make sense without it.

### Watch first

**Andrej Karpathy — Neural Networks: Zero to Hero**
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

Start from the beginning. By the end of this series you will have built a GPT from scratch, character by character. The Shakespeare dataset appears explicitly. Watch at 1.25× if you need to move faster. Do not skip the math.

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

### Requirements

**Training**
- Implement the transformer architecture yourself or adapt [nanoGPT](https://github.com/karpathy/nanoGPT). Do not copy-paste blindly — understand every line.
- Dataset: [Shakespeare text](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB, ~40k lines)
- Tokenization: character-level (no BPE required, though you may add it)
- Logging: track train loss and val loss. Plot them.
- Checkpoint: save a checkpoint after training
- Hardware target: **1× L40S GPU** (see provisioning notes below)

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
- How the transformer works in your implementation (not a theory lecture — show your code)
- Your loss curve and what it tells you
- A live generation from the served endpoint
- What was harder than you expected and why

### Notes

- Target ~6 layers, ~6 heads, ~384 embedding dim. You can go larger if you want, but do not chase scale here.
- If training is stable but slow, check your learning rate and batch size before assuming hardware issues.

---

## Assignment 2: Multi-GPU LLM — Parallelism Strategies

**Goal:** Train a larger language model distributed across multiple GPUs, using at least two distinct parallelism strategies. Then serve it with vLLM.

### Why this matters at Moonrun

Moonrun operates at datacenter scale. Understanding how models are split across GPUs — and what the tradeoffs are — is foundational to the work here. This assignment is where you learn that.

### Background reading (required before starting)

**Data Parallelism**
Each GPU holds a copy of the model. Each sees a different batch. Gradients are averaged across GPUs.
- PyTorch DDP guide: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

**Tensor Parallelism**
Each layer's weight matrices are split across GPUs. GPUs cooperate within a single forward pass via all-reduce/all-gather.
- Megatron-LM paper: https://arxiv.org/abs/1909.08053
- Practical intro: https://huggingface.co/docs/transformers/perf_train_gpu_many#tensor-parallelism

**Pipeline Parallelism**
Different layers live on different GPUs. Forward pass flows through them in sequence (with microbatching to reduce bubble time).
- GPipe paper: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377

**FSDP (Fully Sharded Data Parallel)**
Parameters, gradients, and optimizer states are sharded across GPUs. Reduces per-GPU memory significantly at the cost of communication.
- PyTorch FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

**Useful survey**
- https://huggingface.co/docs/transformers/perf_train_gpu_many — covers all strategies with diagrams

### What you are building

A language model (your choice of dataset — OpenWebText, books corpus, or continue with Shakespeare at larger scale) trained across **at least 2 GPUs** using **at least 2 parallelism strategies**.

### Requirements

**Training**
- Implement or configure two of the following:
  - **DDP** — data parallelism with `torch.nn.parallel.DistributedDataParallel`
  - **FSDP** — sharded data parallel with `torch.distributed.fsdp`
  - **Tensor Parallelism** — use [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) or implement column/row-parallel linear layers manually
  - **Pipeline Parallelism** — use `torch.distributed.pipeline.sync` or implement stage splitting manually
- Write a short comparison: wall-clock time per step, GPU memory usage, and MFU (model FLOP utilization) for each strategy
- Checkpoint the final model

**Serving**
- Serve using [vLLM](https://github.com/vllm-project/vllm)
- Use tensor parallelism in vLLM (`--tensor-parallel-size N`)
- Expose a `/v1/completions` compatible endpoint
- Document startup command

**Hardware target: 2× or 4× L40S GPUs — your choice** (see provisioning notes below)

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

- Do both strategies actually train? (loss curves for both)
- Is your benchmark honest? (we will re-run it)
- Can you explain why one strategy used more memory than the other?
- Does the vLLM server respond to a curl request?

### Sign-off: Presentation to the Team

Same process as Assignment 1. Schedule a meeting and invite:
- raghuvir@moonrun.ai
- ram@moonrun.ai
- evgueni@moonrun.ai

Present to the team and get explicit sign-off before the assignment is considered complete. **You do not submit your onboarding as done until the team signs off on Assignment 2.**

What to cover in the presentation:
- How you split the model across GPUs for each strategy (diagrams welcome)
- Your benchmark results and what drove the differences
- A live request to the vLLM-served model
- What you would do differently at 10× the scale

### Common pitfalls

- NCCL errors on startup usually mean `MASTER_ADDR` / `MASTER_PORT` are not set, or you launched with the wrong number of processes
- Pipeline parallelism bubble time is real — measure it, do not assume 1/N speedup
- vLLM expects HuggingFace-compatible model weights — convert your checkpoint if needed

---

## Compute: Nebius GPU Instances

Moonrun uses Nebius for GPU instances. You will be given credentials.

**Single-GPU (Assignment 1)**
```bash
nebius compute instance create \
  --name raml40s-learn-<yourname>-1 \
  --parent-id <project-id> \
  --resources-platform gpu-l40s-a \
  --resources-preset 1gpu-8vcpu-32gb \
  --network-interfaces '[{"name":"eth0","subnet_id":"<subnet>","public_ip_address":{}}]' \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 200 \
  --preemptible \
  --ssh-key RP-windows-laptop
```

**Multi-GPU (Assignment 2)**

Choose 2 or 4 L40S GPUs — both are valid. More GPUs means you can explore more interesting parallelism configurations, but 2 is sufficient to complete all requirements. Confirm the preset name with your manager before provisioning.

```bash
# 2-GPU
nebius compute instance create \
  --name raml40s-learn-<yourname>-multi \
  --parent-id <project-id> \
  --resources-platform gpu-l40s-a \
  --resources-preset 2gpu-16vcpu-64gb \
  --network-interfaces '[{"name":"eth0","subnet_id":"<subnet>","public_ip_address":{}}]' \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 200 \
  --preemptible \
  --ssh-key RP-windows-laptop

# 4-GPU
nebius compute instance create \
  --name raml40s-learn-<yourname>-multi \
  --parent-id <project-id> \
  --resources-platform gpu-l40s-a \
  --resources-preset 4gpu-32vcpu-128gb \
  --network-interfaces '[{"name":"eth0","subnet_id":"<subnet>","public_ip_address":{}}]' \
  --boot-disk-managed-disk-type network_ssd \
  --boot-disk-managed-disk-size-gibibytes 200 \
  --preemptible \
  --ssh-key RP-windows-laptop
```

**Important:**
- Stop your instance when you are not actively using it. Preemptible instances are cheaper but can be reclaimed — checkpoint frequently.
- Never leave a GPU instance running overnight unattended.
- Ask your manager before provisioning the multi-GPU instance — confirm the preset exists and get approval on cost.

---

## How to Submit

Push your work to a repo under the `Moonrun-ai` GitHub organization: `Moonrun-ai/llm-onboarding-<yourname>`.

When ready for review, open a PR and add your manager as reviewer. Include in the PR description:
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
