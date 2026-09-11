"""
**Connection Map**:
```
Layers → Losses → Autograd
(predictions) (error measurement) (learning signals)
```

Let's implement our loss functions with proper numerical stability and clear educational structure.

### Log-Softmax: The Numerically Stable Foundation

Before implementing loss functions, we need a reliable way to compute log-softmax. 
This function is the numerically stable backbone of classification losses.

### Why Log-Softmax Matters

Naive softmax can explode with large numbers:
```
Naive approach:
  logits = [100, 200, 300]
  exp(300) = 1.97 × 10^130  ← This breaks computers!

Stable approach:
  max_logit = 300
  shifted = [-200, -100, 0]  ← Subtract max
  exp(0) = 1.0  ← Manageable numbers
```

### The Log-Sum-Exp Trick Visualization

```
Original Computation:           Stable Computation:

logits: [a, b, c]              logits: [a, b, c]
   ↓                              ↓
exp(logits)                    max_val = max(a,b,c)
   ↓                              ↓
sum(exp(logits))               shifted = [a-max, b-max, c-max]
   ↓                              ↓
log(sum)                       exp(shifted)  ← All ≤ 1.0
   ↓                              ↓
logits - log(sum)              sum(exp(shifted))
                                  ↓
                               log(sum) + max_val
                                  ↓
                               logits - (log(sum) + max_val)
```

Both give the same result, but the stable version never overflows!


## Systems Analysis: Understanding Loss Function Performance

Loss functions seem simple, but they have important computational and numerical properties that affect training performance. Let's analyze the systems aspects.

### Computational Complexity Analysis

Different loss functions have different computational costs, especially at scale:

```
Computational Cost Comparison (Batch Size B, Classes C):

MSELoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Subtraction    │ O(B)           │
│ Squaring       │ O(B)           │
│ Mean           │ O(B)           │
│ Total          │ O(B)           │
└────────────────┴────────────────┘

CrossEntropyLoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Max (stability)│ O(B*C)         │
│ Exponential    │ O(B*C)         │
│ Sum            │ O(B*C)         │
│ Log            │ O(B)           │
│ Indexing       │ O(B)           │
│ Total          │ O(B*C)         │
└────────────────┴────────────────┘

Cross-entropy is C times more expensive than MSE!
For ImageNet (C=1000), CE is 1000x more expensive than MSE.
```

### Memory Layout and Access Patterns

```
Memory Usage Patterns:

MSE Forward Pass:              CE Forward Pass:

Input:  [B] predictions       Input:  [B, C] logits
       │                             │
       │ subtract                    │ subtract max
       v                             v
Temp:  [B] differences        Temp1: [B, C] shifted
       │                             │
       │ square                      │ exponential
       v                             v
Temp:  [B] squared            Temp2: [B, C] exp_vals
       │                             │
       │ mean                        │ sum along C
       v                             v
Output: [1] scalar            Temp3: [B] sums
                                     │
Memory: 3*B*sizeof(float)            │ log + index
                                     v
                              Output: [1] scalar

                              Memory: (3*B*C + 2*B)*sizeof(float)
```

### Production Context: How Loss Functions Scale

Understanding how loss functions behave in production helps make informed engineering decisions about model architecture and training strategies.

**Loss Function Scaling Challenges**

As models grow larger, loss function bottlenecks become critical:

```
Scaling Challenge Matrix:

                    │ Small Model     │ Large Model      │ Production Scale  │
                    │ (MNIST)         │ (ImageNet)       │ (GPT/BERT)        │
────────────────────┼─────────────────┼──────────────────┼───────────────────┤
Classes (C)         │ 10              │ 1,000            │ 50,000+           │
Batch Size (B)      │ 64              │ 256              │ 2,048             │
Memory (CE)         │ 2.5 KB          │ 1 MB             │ 400 MB            │
Memory (MSE)        │ 0.25 KB         │ 1 KB             │ 8 KB              │
Bottleneck          │ None            │ Softmax compute  │ Vocabulary memory │

Memory grows as B*C for cross-entropy!
At scale, vocabulary (C) dominates everything.
```

**Engineering Optimizations in Production**

```
Common Production Optimizations:

1. Hierarchical Softmax:
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Full Softmax:       │     │ Hierarchical:       │
   │ O(V) per sample     │ →   │ O(log V) per sample │
   │ 50k classes = 50k   │     │ 50k classes = 16    │
   │ operations          │     │ operations          │
   └─────────────────────┘     └─────────────────────┘

2. Sampled Softmax:
   Instead of computing over all 50k classes,
   sample 1k negative classes + correct class.
   50× speedup for training!

3. Label Smoothing:
   Instead of hard targets [0, 0, 1, 0],
   use soft targets [0.1, 0.1, 0.7, 0.1].
   Improves generalization.

4. Lower Precision:
   Using smaller data types can reduce memory
   (e.g., FP16 uses 2 bytes instead of 4).
```
"""

import numpy as np
rng = np.random.default_rng(7)
from typing import Optional

from core.tensor import Tensor
from core.activations import ReLU
from core.layers import Linear

# Constant for numerical stability
EPSILON = 1e-7  # Small value to prevent log(0) and numerical instability

def log_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Compute log-softmax with numerical stability."""
    # STEP 1: Find max along dimension for numerical stability
    max_vals = np.max(x.data, axis=dim, keepdims=True)

    # STEP2 2: Subtract max to prevent overflow
    shifted = x.data - max_vals

    # STEP 3: Compute log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=dim, keepdims=True))

    # STEP 4: Return log_softmax = input - max - log_sum_exp
    result = x.data - max_vals - log_sum_exp
    return Tensor(result)


class MSELoss:
    """Mean Squared Error loss for regression tasks."""

    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute mean squared error between predictions and targets."""
        squared_diff = (predictions.data - targets.data)**2
        mse = np.mean(squared_diff)
        return Tensor(mse)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        """Compute gradients (PLACEHOLDER)"""
        pass


class CrossEntropyLoss:
    """Cross Entropy loss for multi-class classification."""

    def __init__(self):
        pass

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss between logits and target class indices."""

        # STEP 1: Compute log-softmax for numerical stability
        log_probs = log_softmax(logits, dim=-1)

        # STEP 2: Select log-probabilities for correct classes
        batch_size = logits.shape[0]
        num_classes = logits.shape[-1]
        target_indices = targets.data.astype(int)

        out_of_range = (target_indices < 0) | (target_indices >= num_classes)
        if np.any(out_of_range):
            bad_values = np.unique(target_indices[out_of_range])
            raise ValueError(
                f"CrossEntropyLoss target index out of range: {bad_values.tolist()}"
                f"  Valid range for {num_classes} classes is [0, {num_classes - 1}]"
            )

        # Select correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]

        # STEP 3: Return negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)
        return Tensor(cross_entropy)

    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        return self.forward(logits, targets)

    def backward(self) -> Tensor:
        pass


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        pass

    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute binary cross-entropy loss."""

        # STEP 1: Clamp predictions to avoid numerical issues with log(0) and log(1)
        eps = EPSILON
        clamped_preds = np.clip(predictions.data, eps, 1 - eps)

        # STEP 2: Compute binary cross-entropy
        # BCE = -(targets * log(preds) + (1-targets) * log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1 - clamped_preds)

        bce_per_sample = -(targets.data * log_preds + (1 - targets.data) * log_one_minus_preds)

        # STEP 3: Return mean across all samples
        bce_loss = np.mean(bce_per_sample)
        return Tensor(bce_loss)

    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return self.forward(predictions, targets)

    def backward(self) -> Tensor:
        pass



        
    