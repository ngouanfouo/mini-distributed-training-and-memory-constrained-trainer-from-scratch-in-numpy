# Mini Distributed Training & Memory-Constrained Trainer — Pure NumPy
 
A from-scratch training stack, built entirely in NumPy, that mirrors the core techniques modern deep learning frameworks use to **scale models** and **fit them into limited memory**. No PyTorch, no autograd engine — every gradient is derived and implemented by hand.
 
This project walks through an MLP with manual backprop, then layers on the same tricks used in large-scale training: gradient accumulation, activation checkpointing, mixed precision, data-parallel all-reduce, and ZeRO-style optimizer state sharding — each implemented under an explicit, realistic memory budget.

### Author:Tiayo Durel
 
## Why this project
 
Frameworks like PyTorch and JAX hide an enormous amount of machinery behind `.backward()` and a `Trainer` class. This project rebuilds that machinery from first principles so the *mechanics* — not just the API — are transparent:
 
- How does a forward/backward pass actually compute and store gradients?
- Why does gradient accumulation reproduce a large-batch gradient exactly?
- What does activation checkpointing trade away, and why does it save memory?
- How does mixed precision keep accuracy while halving memory footprint?
- How do multiple "workers" synchronize gradients under data parallelism?
- How does ZeRO shard optimizer state across workers without changing the math?
## Features
 
### 1. MLP with Manual Autograd
- Forward pass (`mlp_forward`) with cached activations for backprop
- Mean-squared-error loss and gradient (`mse_loss_and_grad`)
- Manual backward pass (`mlp_backward`) deriving gradients for `W1`, `b1`, `W2`, `b2`
- Verified numerically against full-batch gradients at every later stage
### 2. Gradient Accumulation
- Splits a batch into micro-batches (`split_into_micro_batches`) to fit a memory budget
- Accumulates per-micro-batch gradients (`accumulate_gradients`) and rescales once at the end
- Produces a gradient **numerically identical** to running the full batch at once — including with uneven micro-batch sizes
### 3. Activation Checkpointing
- Trades compute for memory by discarding intermediate activations and recomputing them during the backward pass
- Demonstrates the classic memory/compute tradeoff at the heart of training very deep or very large models
### 4. Mixed Precision Training
- Simulates lower-precision (e.g. fp16-style) forward/backward computation with a fp32 master-weight copy
- Covers loss scaling to prevent gradient underflow, and unscaling before the optimizer step
