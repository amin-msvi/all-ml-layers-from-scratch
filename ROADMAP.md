# all-ml-layers-from-scratch

## Table of Contents

1. Ground Rules
2. Phase I — ML Framework Foundations

   1. TinyTorch Foundation
   2. Framework Bridge
   3. Language Architecture
   4. Tiny Language Model
   5. Distributed Runtime
3. Phase II — Systems & Optimization
4. Definition of Done

# 1. Ground Rules

**Primary project:** `all-ml-layers-from-scratch`

**Primary curriculum:** TinyTorch v0.1.13

**Reference implementations:**

* TinyTorch source — inspect after implementing
* TrenTorch — inspect for alternative/refactored designs
* PyTorch — production reference when useful

**Rule:** learn because the project needs something, not because another course contains it.

For every feature:

```text
Understand problem
      ↓
Define API + expected behavior
      ↓
Implement yourself
      ↓
Test correctness
      ↓
Integrate with existing framework
      ↓
Inspect TinyTorch / TrenTorch
      ↓
Refactor if useful
      ↓
Commit
```

Do **not** inspect the reference implementation before making a serious attempt yourself.

Do **not** optimize code that has not first been tested for correctness.

---

# 2. Phase I — ML Framework Foundations

## 2.1 TinyTorch Foundation — Modules 01–08

### Reference

TinyTorch Foundation Tier:

```text
01 Tensor
02 Activations
03 Layers
04 Losses
05 DataLoader
06 Autograd
07 Optimizers
08 Training
```

### Objective

Reach:

```text
data
 ↓
Tensor
 ↓
Module
 ↓
forward
 ↓
loss
 ↓
autograd
 ↓
gradients
 ↓
optimizer
 ↓
updated parameters
```

using only `all-ml-layers-from-scratch`.

### Workflow for Each Module

#### 1. Read

Use the **module webpage as the primary material**.

Skim the slides first or afterward.

Only study supplementary material deeply when something is unclear.

#### 2. Extract the contract

Before coding, identify:

* required classes/functions
* inputs/outputs
* shape semantics
* mathematical behavior
* edge cases
* interaction with previous modules

#### 3. Implement in `all-ml-layers-from-scratch`

Your architecture, not TinyTorch's repository structure.

#### 4. Test

Maintain:

```text
tests/
├── unit/
├── integration/
└── differential/
```

Use three kinds of correctness checks:

**TinyTorch-derived tests**

Adapt relevant TinyTorch module tests.

**Differential tests**

For example:

```text
your matmul   ≈ numpy.matmul
your softmax  ≈ torch.softmax
your gradients ≈ torch autograd
your Adam step ≈ torch.optim.Adam
```

**Integration tests**

Verify that previously implemented pieces compose correctly.

#### 5. Compare

Only after your implementation works:

1. TinyTorch implementation
2. TrenTorch implementation
3. PyTorch design/docs when interesting

Then decide whether your design should change.

### Foundation Milestone

After Module 08:

* train an MLP
* solve XOR
* train on a tiny dataset
* demonstrate decreasing loss
* verify gradients numerically
* save/reload model parameters

At this point you have a genuine miniature ML framework.

---

## 2.2 Framework Bridge

Do this **after Foundation 01–08 and before building the Transformer**.

Add only these abstractions:

```text
Tensor
├── data
├── dtype
├── device
└── autograd metadata

Device
└── cpu

Backend
└── NumPyBackend

Module
├── Parameter registration
├── modules()
├── parameters()
├── named_parameters()
├── state_dict()
└── load_state_dict()
```

### Why Add Them Here?

They create architectural seams required later.

**`dtype`**

Needed eventually for:

* FP32
* FP16
* BF16
* mixed precision
* memory accounting

**`Device`**

Initially only:

```text
Device("cpu")
```

Later:

```text
cpu
cuda:0
cuda:1
...
```

and eventually distributed placement.

**`Backend`**

Prevents NumPy calls from leaking everywhere.

Today:

```text
Tensor
 ↓
NumPyBackend
```

Later:

```text
Tensor
 ├── NumPyBackend
 ├── C++Backend
 ├── CUDABackend
 └── TritonBackend
```

**`Module` / `Parameter` state**

Necessary for:

* Transformers
* checkpointing
* model replication
* DDP
* FSDP
* sharding
* parameter synchronization

### Explicitly Do NOT Build Yet

Do not prematurely build:

* custom Storage implementation
* custom allocator
* CUDA
* distributed tensors
* ProcessGroup
* DeviceMesh
* kernel fusion

Those belong later.

---

## 2.3 TinyTorch Architecture — Modules 10–13

Skip Module 09 convolution for now.

### Reference

```text
10 Tokenization
      ↓
11 Embeddings
      ↓
12 Attention
      ↓
13 Transformers
```

### Objective

Your framework should support:

```text
text
 ↓
tokenizer
 ↓
token IDs
 ↓
embeddings
 ↓
attention
 ↓
transformer blocks
 ↓
logits
 ↓
cross entropy
 ↓
autograd
 ↓
optimizer
```

Implement:

* tokenizer
* embedding
* positional representation
* LayerNorm/RMSNorm as needed
* scaled dot-product attention
* causal masking
* multi-head attention
* feed-forward network
* residual connections
* Transformer block
* GPT-style model

---

## 2.4 Tiny Language Model Milestone

Now build an actual tiny autoregressive language model.

### Objective

Prove the entire framework works on a realistic computation graph.

### Definition of Done

The model must:

* overfit one tiny batch
* train on a small text corpus
* show decreasing next-token loss
* generate tokens autoregressively
* save a checkpoint
* reload it
* produce identical logits after reload

Do not care about producing good language.

The milestone is:

> A Transformer can be trained entirely through my framework.

---

## 2.5 Distributed Runtime

Now distributed systems becomes part of the same project.

### Primary Reference

Umar Jamil's distributed-training-from-first-principles video.

Use chapters **on demand**, not as a 19-hour linear course.

### Stage A — Runtime Fundamentals

Build:

```text
Process
├── rank
├── world_size
└── local_rank

ProcessGroup
```

Then:

```text
send
recv
broadcast
reduce
all_reduce
all_gather
reduce_scatter
all_to_all
```

Initially use:

* Python multiprocessing
* CPU tensors
* simple local communication

The goal is understanding semantics, not performance.

### Stage B — Distributed Data Parallel

Start with:

```python
loss.backward()

for parameter in model.parameters():
    all_reduce(parameter.grad)
    parameter.grad /= world_size

optimizer.step()
```

Then progressively add:

```text
naive DDP
 ↓
autograd hooks
 ↓
gradient buckets
 ↓
asynchronous collectives
 ↓
communication/backward overlap
```

### Stage C — Device Mesh

Represent multidimensional process layouts.

Example:

```text
            TP
        0 ─── 1
DP      │     │
        2 ─── 3
```

This becomes the abstraction used to compose parallelism strategies.

### Stage D — Sharded Data Parallel

Implement concepts behind:

```text
ZeRO
FSDP
HSDP
```

Understand:

```text
parameter all-gather
compute
gradient reduce-scatter
parameter resharding
```

### Stage E — Tensor Parallelism

Partition operations themselves.

Implement:

* column-parallel linear
* row-parallel linear
* attention projections
* required collectives

### Stage F — Pipeline Parallelism

Partition layers across ranks.

Implement:

```text
naive pipeline
 ↓
microbatches
 ↓
GPipe
 ↓
1F1B
 ↓
bubble analysis
```

### Stage G — Advanced Parallelism

Only after the above works:

```text
Context Parallelism
Ring Attention
Mixture of Experts
Expert Parallelism
Expert Tensor Parallelism
All-to-All token dispatch
```

### Distributed Milestone

Train the tiny language model using multiple processes with combinations of:

```text
DP
FSDP
TP
PP
```

and be able to explain exactly where every tensor, parameter, gradient, and communication collective lives.

---

# 3. Phase II — Systems & Optimization

Correctness comes first.

Then profile.

```text
correct
 ↓
measure
 ↓
find bottleneck
 ↓
understand bottleneck
 ↓
optimize
 ↓
measure again
```

## 3.1 Framework-Level Optimization

Investigate:

* unnecessary Python loops
* vectorization
* temporary tensor allocations
* copies vs views
* contiguous layouts
* broadcasting
* memory usage
* gradient graph memory

Use TrenTorch as a useful comparison here.

---

## 3.2 Kernel-Level Exploration

Selectively move downward:

```text
Python API
 ↓
backend abstraction
 ↓
NumPy
 ↓
C/C++ extension
 ↓
CPU kernels / SIMD
 ↓
CUDA extension
 ↓
CUDA kernels
 ↓
Triton
```

Do not rewrite everything.

Choose individual operations such as:

* elementwise ops
* reductions
* softmax
* LayerNorm
* matmul
* attention

and understand how performance changes at each layer.

---

## 3.3 Distributed Optimization

Return to the distributed runtime and improve:

* communication volume
* communication/computation overlap
* gradient bucketing
* sharding
* memory usage
* pipeline bubbles
* collective selection
* load balancing
* checkpoint efficiency

At this point the project connects:

```text
ML mathematics
      +
autograd
      +
framework design
      +
computer architecture
      +
GPU programming
      +
distributed systems
```

---

# 4. Definition of Done

There is no requirement that `all-ml-layers-from-scratch` become production quality.

The project succeeds when you can trace:

```text
Python model code
       ↓
Module / Parameter
       ↓
Tensor operations
       ↓
Autograd graph
       ↓
Backend operations
       ↓
Device execution
       ↓
Distributed placement
       ↓
Collective communication
       ↓
Parallel training strategy
       ↓
Hardware / kernels
```

and understand what each abstraction is doing and why it exists.

That is the stack you are building.
