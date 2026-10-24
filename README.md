# Moonrun LLM Interview Assignment

This is the technical assignment for the role. By the end you will have trained and served two language models from scratch, one on a single GPU and one distributed across multiple GPUs.

Writing the code is a few hours of work. Hand the requirements to Claude and most of it comes back quickly. The time goes into what happens after that: running it, measuring it, changing one thing at a time and working out why the numbers moved. Budget your effort there, not on the code. We expect you to be ready to present about a week after you start.

We are not looking for an LLM expert. Everything at Moonrun is built around this technology, so what we need to know is whether you can follow the conversation, ask the right questions and contribute without being caught up every time. This assignment is how we find that out.

It is not a course. Read what you need, build it, then go find out how it behaves.

---

## Selection criteria

This is the key interview. Using Claude to write the code is fine and we expect it. What we are selecting on is the depth of your understanding of what you built, and we read that relative to how much exposure you have already had to LLMs.

So spend the time. Go through the assignments properly, build the graphs and the visuals, whatever it takes until you are confident you can explain what was happening and why.

Be creative with it. What to present under each assignment is a floor, not the list of experiments worth running. Run your own assessments and analysis, and tell us why you ran each one.

**What we are always chasing at Moonrun is the highest performance at the lowest cost, so that is the lens to think in.** Experiments that went nowhere are as welcome as the ones that worked, and you should bring both. We are not scoring results. We want to see how you think.

---

## Timeline

A week, and Assignment 1 before Assignment 2 since the second builds on the first.

If you need longer, tell us why and we will work with it.

---

## Foundations

### Watch first if you are not familiar with LLMs

**Andrej Karpathy — Neural Networks: Zero to Hero**
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

Caution: these videos are long and detailed, and going through all of them takes real time. Use them as reference and pull out what you need. By the end of the series you will have built a GPT from scratch, character by character, on the Shakespeare dataset.

Key episodes:
- *The spelled-out intro to language modeling* — builds bigram model, introduces loss
- *Building makemore* (parts 1–5) — MLP → BatchNorm → backprop from scratch
- *Let's build GPT* — the full transformer, trained on Shakespeare

### Other references

None of these are prerequisites. Pull from whichever one answers the question in front of you.

- **The Illustrated Transformer** (Jay Alammar)
  https://jalammar.github.io/illustrated-transformer/
  The clearest visual explanation of attention there is.
- **Attention Is All You Need** (2017)
  https://arxiv.org/abs/1706.03762
  The architecture section and the multi-head attention equations are the parts that matter here.
- **The Illustrated GPT-2** (Jay Alammar)
  https://jalammar.github.io/illustrated-gpt2/
  Bridges the transformer paper to autoregressive language modeling.
- **Build a Large Language Model From Scratch** (Sebastian Raschka)
  https://github.com/rasbt/LLMs-from-scratch
  A clean implementation to read alongside your own.
- **Hugging Face NLP Course**
  https://huggingface.co/learn/nlp-course/
  Chapters 2 to 4 for tokenization and fine-tuning questions.
- **Lilian Weng's blog**
  https://lilianweng.github.io/
  Deep dives on one topic at a time, for when you need to go further than the rest of this list.
- **fast.ai Practical Deep Learning**
  https://course.fast.ai/
  Intuition on training dynamics.
- **PyTorch performance and profiling docs**
  https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
  Relevant to both assignments, since measuring the run is most of the work.

---

## Assignment 1: Shakespeare LLM — Single GPU

**Goal:** Train a character-level GPT on the Shakespeare dataset and serve it via an HTTP API on a single GPU.

### What you are building

A transformer language model that generates Shakespeare-style text. This is intentionally small — the point is to own every part of the pipeline end-to-end, not to achieve state-of-the-art perplexity.

Once it trains, the interesting part is measuring it and then making it faster.

### Requirements

**Training**
- Implement the transformer architecture yourself or adapt [nanoGPT](https://github.com/karpathy/nanoGPT). However you write it, you must be able to explain every line.
- Dataset: [Shakespeare text](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) (~1MB, ~40k lines)
- Tokenization: character-level (no BPE required, though you may add it)
- Logging: track train loss and val loss, step time, and GPU utilization and memory across the run. Plot them.
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

### What to present

Bring graphs. We want to see that you understand the model you built and that you understand how it ran on the GPU. Plots are how you show us both, so label the axes and say what each one led you to conclude.

Do not walk us through the whole list. Measure all of it, then decide for yourself which few things actually mattered for making this run faster or cheaper, and present those with the data behind them. Three findings you can argue properly beat every number you collected. Keep the rest to hand in case we ask.

Measure at least this much, then filter:

- **Loss curves.** Train and validation on the same axes. Where does validation stop following train, and what did you do about it? Explain any kink, spike or plateau rather than leaving it on the slide. Tell us where in the run you stopped buying anything with the time you were spending.
- **Step time.** Milliseconds per training step, and where they go: forward, backward, optimizer, data loading. This is the number everything else is paid out of.
- **Time to train.** Wall clock for the full run and the dollars that bought, and what you would change to halve either one.
- **If you did anything to speed it up.** Batch size, mixed precision, warmup, keeping the GPU fed, whatever you tried. Give before and after numbers, including for the things that did not work, and say what each one did to the cost of the run.
- **What the GPU was actually doing.** Utilization and memory across the run. You are paying for the whole card the whole time, so an idle GPU is the interesting case and we want to know why it was idle.
- **The cheapest version of this run.** Having measured it, what is the least money you could reach the same loss for, and what is in the way of doing that today.
- **Your code, then a live generation.** Walk the attention path as you wrote it, not the theory, then generate from the served endpoint.
- **Your own experiments.** What you assessed beyond this list, why you thought it was worth measuring, and what came back. Bring the ones that failed too.
- **What was harder than you expected**, and why.

---

## Assignment 2: Multi-GPU LLM — Parallelism Strategies

**Goal:** Train a larger language model distributed across multiple GPUs using two parallelism strategies, benchmark both, and serve the result with vLLM.

The main focus is the benchmarking. Training the model is the means. What we are after is how it ran on the GPUs, where the bottlenecks were, and what you would do to speed it up.

### Why this matters at Moonrun

Moonrun operates at datacenter scale. Most of the work is finding out why a run is slower than the hardware allows, and then fixing it. This assignment is a small version of that.

### Background reading

**FSDP (Fully Sharded Data Parallel)**
- PyTorch FSDP tutorial: https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html

**Tensor Parallelism**
- Megatron-LM paper: https://arxiv.org/abs/1909.08053

**Pipeline Parallelism**
- GPipe paper: https://arxiv.org/abs/1811.06965
- PipeDream: https://arxiv.org/abs/1806.03377

**Useful survey**
- https://huggingface.co/docs/transformers/perf_train_gpu_many — covers all strategies with diagrams

### What you are building

A language model trained across **at least 2 GPUs** with two different parallelism strategies, so you can compare them. Dataset is your choice: OpenWebText, a books corpus, or Shakespeare again at larger scale.

### Requirements

**Training**
- Pick two of the following three and implement both:
  - **FSDP**
  - **Tensor Parallelism**
  - **Pipeline Parallelism**
- Record what you need to compare the two: wall-clock time per step and where that time goes, GPU memory per device, throughput, and MFU (model FLOP utilization). Capture the same numbers at each GPU count you run.
- Checkpoint the final model

**Serving**
- Serve using [vLLM](https://github.com/vllm-project/vllm)
- Use tensor parallelism in vLLM (`--tensor-parallel-size N`)
- Getting your checkpoint into a form vLLM will load is part of this
- Expose a `/v1/completions` compatible endpoint
- Document startup command

**Hardware target: 2× or 4× L40S GPUs, your choice** (see [Compute](#compute) below)

### Deliverables

- [ ] `train_distributed.py` — distributed training entry point
- [ ] `model.py` — model definition compatible with your parallelism approach
- [ ] `serve_distributed.py` (or vLLM launch script)
- [ ] `benchmark.md` — side-by-side comparison of your two parallelism strategies:
  - GPU memory per device
  - Step time, and steps/sec or tokens/sec
  - MFU, with the peak figure and the FLOP count you used to arrive at it
  - Which card you used and how the GPUs were connected to each other
  - Notes on what you had to change when switching strategies
- [ ] `requirements.txt` or `pyproject.toml`
- [ ] `README_assignment2.md` — how to launch training, how to launch serving

### What we will check

We are not assessing the quality of the model. Loss values and sample text are not the point here. What we want to see is that you understand how the thing performs on the hardware.

- Which of your two strategies performed better, and why. This is the question we care about most.
- How it could be improved.

### What to present

Same bar as Assignment 1, now across GPUs and with a bill that multiplies by the GPU count. The benchmark is the centre of this one, so lead with the comparison and let the graphs carry it.

Filter the same way. Measure all of it, then build the presentation around the few findings that decide which strategy you would actually run and what you would change to make it faster or cheaper. Everything else is backup.

Measure at least this much, then filter:

- **Loss curves for both strategies**, on the same axes, far enough to show the two runs are comparable. We are not reading them for model quality.
- **Step time for each strategy**, and where that time goes.
- **Throughput and MFU.** Tokens per second and time to train for each strategy, against the hardware peak. MFU is the share of the hardware you are paying for that you actually used, so treat the gap as the finding.
- **Cost of the same work under each strategy.** Dollars to train the same model each way, and the throughput you got per dollar.
- **Scaling.** Step time and throughput at each GPU count you ran. Twice the GPUs is twice the bill, so quantify the speedup you did not get and account for it.
- **Memory per device** for each strategy, and why the two differ. Memory you did not need is a smaller instance you could have rented.
- **How you split the model.** A diagram per strategy showing what lives where and what crosses the wire.
- **Which strategy won, and why.** Be specific about won at what: step time, memory, tokens per dollar. Then how you would improve it: the change you would make first, and the number you expect it to move.
- **A live request to the vLLM endpoint**, and what you would do differently at 10× the scale and at 10× the cost pressure.
- **Your own experiments.** The configurations and limits you probed beyond this list, why each one seemed worth the time, and what it cost against what it bought. The dead ends count.

---

## Presentation

One session covers both assignments. Send a note to ram@moonrun.ai when you are ready and we will get it scheduled: a two hour interview, roughly 45 minutes on each assignment and the rest for questions.

What to present is listed under each assignment. Those lists are what to measure. What to say is your call, and choosing well is part of what we are reading.

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
- If L40S is unavailable wherever you look, anything with comparable memory works. Note which card you used in `benchmark.md`, since your numbers depend on it.
