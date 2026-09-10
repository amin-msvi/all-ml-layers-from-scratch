"""
# Module 02: Activations - Intelligence Through Nonlinearity

**Connection Map**:
```
Tensor → Activations → Layers
(data)   (intelligence) (architecture)
```

**Dependency Flow**:
```
Module 01 (Tensor) → Module 02 (Activations) → Module 03 (Layers)
     ↓                      ↓                         ↓
  Foundation          Nonlinearity              Architecture
```


## Implementation: Building Activation Functions

### Implementation Pattern

Each activation follows this structure:
```python
class ActivationName:
    def forward(self, x: Tensor) -> Tensor:
        # Apply mathematical transformation
        # Return new Tensor with result
```

"""


import numpy as np
rng = np.random.default_rng(7)
from typing import Optional

from .tensor import Tensor

# Constant for numerical comparisons
TOLERANCE = 1e-10  # Small tolerance for fp comparisons in tests

class Sigmoid:
    """Sigmoid activation: sigma(x) = 1/(1 + exp(-x))"""

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        """Apply sigmoid activation element-wise."""

        x_data = x.data
        with np.errstate(over="ignore", invalid="ignore"):
            result = np.where(
                x_data >= 0,
                1.0 / (1.0 + np.exp(-x_data)),
                np.exp(x_data) / (1.0 + np.exp(x_data))
            )
        return Tensor(result)
        
    def __call__(self, x: Tensor) -> Tensor:
        """Allows the activation to be called like a function."""
        return self.forward(x)


class ReLU:
    """ReLU activation: f(x) = max(0, x)"""

    def parameters(self):
        """Return empty list (activations have no learnable parameters)."""
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.maximum(0, x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Tanh:
    """Tanh activation: f(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""

    def parameters(self):
        """Return empty list (activations have no learnable parameters)"""
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class GELU:
    """
    GELU - The Smooth Modern Choice

    GELU (Gaussian Error Linear Unit) is a smooth approximation to ReLU that's 
    become popular in modern architectures like transformers. Unlike ReLU's sharp corner,
    GELU is smooth everywhere.

    
    Mathematical Definition:
    The exact GELU multiplies x by Φ(x), the probability that a standard
    normal random variable is ≤ x (an S-curve from 0 to 1):
    ```
    GELU(x) = x · Φ(x)
    ```
    
    Two common approximations exist:
    ```
    Tanh:    0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])   ← more accurate
    Sigmoid: x · σ(1.702x)                            ← faster, used here
    ```
    
    We use the sigmoid form because it reuses sigmoid and the single constant 1.702
    is empirically fitted so that σ(1.702x) ≈ Φ(x).

    Used in GPT, BERT, and other modern architectures. The smoothness helps with 
    optimization compared to ReLU's sharp corner.
    """

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        return Sigmoid()(x * 1.702) * x

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Softmax:
    """
    Softmax activation: f(x_i) = e^(x_i) / Σ(e^(x_j))

    Converts any vector to a probability distribution.
    Sum of all outputs equals 1.0.
    """

    def parameters(self):
        return []

    def forward(self, x: Tensor, dim: int=-1) -> Tensor:
        # Numerical stability: Subtract max to prevent overflow
        x_max = np.max(x.data, axis=dim, keepdims=True)
        x_shifted = x.data - x_max

        # Computer exponentials
        exp_values = np.exp(x_shifted)

        # Sum along dimension
        exp_sum = np.sum(exp_values, axis=dim, keepdims=True)

        # Normalize to get probabilities
        result = exp_values / exp_sum
        return Tensor(result)

    def __call__(self, x: Tensor, dim: int=-1) -> Tensor:
        return self.forward(x, dim)

    
    
        
    