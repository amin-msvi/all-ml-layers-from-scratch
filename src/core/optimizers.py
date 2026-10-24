"""
# Model Optimizers

## 1. Implementation Strategy

Optimizer Base Class
    ↓
SGD (foundation algorithm):
    - Strategy: Always step downhilll
    - Problem: Can get stuck oscillating in narrow valleys
    - Solution: Add momentum to "coast" through oscillating
    ↓
SGD + Momentum (reduce oscillations)
    ↓
Adam (adaptive learning rates)
    - Strategy: Adapt step size for each parameter individually
    - Advantage: Different learning rates for different dimensions.
    - Key Insight: Some directions need big steps, others need small steps
    ↓
AdamW (proper weight decay)
    - Strategy: Adam + proper regularization
    - Fix: Separates optimization from regularization
    - Result: Better generalization and training stability

---

## 2. Foundations: Mathematics Background
At its core, optimization follows: `theta_new = theta_old - (alpha * direction)`
Where:
    - `theta`: parameters (your position in the lanscape)
    - `alpha`: step size (learning rate)
    - `direction`: where to step (gradient-based)

-> Sophisticated optimizers do much more than basic gradient descent.


### 2.1 Understanding Momentum: The Physics of Optimization

Momenmum in optimization works like momentum in physics. A ball rolling down
a hill doesn't immediately change direction when it hits a small bump -> It has 
momentum that carries it forward.

- SGD with Momentum Formula:
    ```
    >>> velocity = betha * previous_velocity + current_gradient
    >>> parameter = parameter - learning_rate * velocity
    ```
    where betha ≈ 0.9 means "90% memory of previous direction"
"""


import numpy as np
from typing import List, Union, Optional, Dict, Any

from core.tensor import Tensor

from core.autograd import enable_autograd
enable_autograd()

# Constants for optimizer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01  # Default learning rate for SGD
DEFAULT_LEARNING_RATE_ADAM = 0.001  # Default learning rate for Adam/AdamW
DEFAULT_MOMENTUM = 0.9  # Default momentum for SGD
DEFAULT_BETA1 = 0.9  # First moment decay rate for Adam
DEFAULT_BETA2 = 0.999  # Second moment decay rate for Adam
DEFAULT_EPS = 1e-8  # Small epsilon for numerical stability in Adam
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01  # Default weight decay for AdamW



class Optimizer:
    """
    Base class for all optimizers.

    This class defines the common interface that all optimizers must implement:
        - zero_grad(): Clear gradients from parameters
        - step(): Update parameters based on gradients
    """

    def __init__(self, params: List[Tensor]):
        """Initialize optimizer with parameters to optimize."""
        # Validate and store parameters
        if not isinstance(params, list):
            params = list(params)

        # Store parameters
        self.params = params

        # Ensure parameters participate in autograd once it is enabled.
        # Only initialize grad if it isn't set yet, a caller may have already
        # computed and assigned a gradient before constructing the optimizer.
        for param in self.params:
            if isinstance(param, Tensor):
                param.requires_grad = True
                if not hasattr(param, 'grad'):
                    param.grad = None
        self.step_count = 0  # For algorithms that need step counting

    def zero_grad(self):
        """Clear gradients from all parameters."""

        for param in self.params:
            param.grad = None

    def step(self):
        """
        Update parameters based on gradients

        This is abstract - each optimizer implements its own update rule.
        """
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f"{self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
        )

    def _extract_gradient(self, param: Tensor) -> np.ndarray:
        """
        Extract gradient data as a NumPy array from a parameter.

        Gradients can be stored as either a Tensor (with .data attribute)
        or as a raw NumPy array (from autograd). This helper normalizes
        both cases to a plain NumPy array for optimizer math.

        EXAMPLE:
        >>> param = Tensor([1.0, 2.0], requires_grad=True)
        >>> param.grad = Tensor([0.1, 0.2])
        >>> optimizer._extract_gradient(param)
        array([0.1, 0.2])
        """
        grad = param.grad

        if isinstance(grad, Tensor):
            return grad.data
        return grad


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum and weight decay.

    SGD is the foundational optimization algorithm that moves parameters
    in the direction opposite to gradient. With momentum, it remembers
    previous updates to reduce oscillations and accelerate convergence.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_SGD,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """Initialize SGD optimizer."""
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Initialize momentum buffer (created lazily)
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        """
        Check if this optimizer uses momentum.

        This explicit API method replaces the need for hasattr() checks
        in checkpointing code

        Returns:
            bool: True if momentum is enabled (momentum > 0), False otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.has_momentum()
            True
        """
        return self.momentum > 0

    def get_momentum_state(self) -> Optional[List]:
        """
        Get momentum buffers for checkpointing.

        This explicit API method provides safe access to momentum buffers
        without using hasattr(), making the API contract clear.

        Returns:
            Optional[List]: List of momentum buffers if momentum is enabled,
                            None otherwise

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> optimizer.step()  # Initialize buffers
            >>> state = optimizer.get_momentum_state()
            >>> # Later: optimizer.set_momentum_state(state)
        """
        if not self.has_momentum():
            return None
        return [buf.copy() if buf is not None else None
            for buf in self.momentum_buffers]

    def set_momentum_state(self, state: Optional[List]) -> None:
        """
        Restore momentum buffers from checkpointing.

        This explicit API method provides safe restoration of momentum state
        without using hasattr().

        Args:
            state: List of momentum buffers or None

        EXAMPLE:
            >>> optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> state = optimizer.get_momentum_state()
            >>> # Training interruption...
            >>> new_optimizer = SGD(params, lr=0.01, momentum=0.9)
            >>> new_optimizer.set_momentum_state(state)
        """
        if state is None or not self.has_momentum():
            return

        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length mismatch\n"
                f"State has {len(state)} buffers, but optimizer has {len(self.momentum_buffers)} parameters\n"
            )

        for i, buf in enumerate(state):
            if buf is not None:
                self.momentum_buffers[i] = buf.copy()

    def step(self):
        """Perform SGD update step with momentum."""

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)
    
            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data
    
            # Update momentum buffer
            if self.momentum != 0:
                if self.momentum_buffers[i] is None:
                    # Initialize momentum buffer
                    self.momentum_buffers[i] = np.zeros_like(param.data)
    
                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
                grad_data = self.momentum_buffers[i]

            # Update parameter: param = param - lr * grad
            param.data = param.data - self.lr * grad_data

        # Increment step counter
        self.step_count += 1


class Adam(Optimizer):
    """
    Adam optimizer with adaptive learning rates.

    Adam computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.
    This makes it effective for problems with sparse gradients or noisy data.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = 0.0
    ):
        """Initiate Adam optimizer."""
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (created lazily)
        self.m_buffers = [None for _ in self.params]  # First moment (mean)
        self.v_buffers = [None for _ in self.params]  # Second moment (variance)

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """
        Update first and second moment estimates with bias correction.

        Moment Updates - EMA and Bias Correction
        
        Adam tracks two running statistics per parameter: a first moment (mean of
        gradients) and a second moment (mean of squared gradients). Both use
        exponential moving averages (EMA) and need bias correction because they
        start from zero.
        
        ```
        grad_data ──→ m = β₁ * m + (1-β₁) * grad       (direction EMA)
                 └──→ v = β₂ * v + (1-β₂) * grad²      (magnitude EMA)
        
                     m̂ = m / (1 - β₁^t)                 (bias-corrected mean)
                     v̂ = v / (1 - β₂^t)                 (bias-corrected variance)
        ```
        
        This helper isolates the EMA + bias correction math so that `step()`
        only has to compose: extract gradient, update moments, apply update.

        EXAMPLE:
        >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))
        >>> # m_hat ≈ grad (after bias correction at step 1)
        >>> # v_hat ≈ grad^2 (after bias correction at step 1)
        """
        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)

        # Update biased first moment estimate
        self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data

        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)

        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count

        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2

        return m_hat, v_hat

    def step(self):
        """
        Perform Adam update step by composing helpers.

        Formula: `θ_t = θ_{t-1} - lr * m̂_t / (√v̂_t + ε)`
        """
        # Increment step counter first (needed for bias correction)
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Apply weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            # Update moments and get bias-corrected estimates
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Update parameter
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    AdamW fixes a bug in Adam's weight decay implementation by decoupling
    weight decay from the gradient-based update. This leads to better
    regularization and is the preferred version for most applications.
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = DEFAULT_LEARNING_RATE_ADAM,
        betas: tuple = (DEFAULT_BETA1, DEFAULT_BETA2),
        eps: float = DEFAULT_EPS,
        weight_decay: float = DEFAULT_WEIGHT_DECAY_ADAMW,
    ):
        super().__init__(params)

        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moment buffers (same as Adam)
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]

    def _update_moments(self, i: int, grad_data: np.ndarray) -> tuple:
        """
        Update first and second moment estimates with bias correction for AdamW.
    
        Identical math to Adam's _update_moments: EMA of gradient and squared
        gradient, with bias correction. The key difference is in how step()
        calls this -- AdamW passes pure gradients without weight decay mixed in.
    
        EXAMPLE:
        >>> m_hat, v_hat = self._update_moments(0, np.array([0.1, 0.2]))
        """
        # Initialize buffers if needed
        if self.m_buffers[i] is None:
            self.m_buffers[i] = np.zeros_like(grad_data)
            self.v_buffers[i] = np.zeros_like(grad_data)
    
        # Update biased first moment estimate
        self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1 - self.beta1) * grad_data
    
        # Update biased second moment estimate
        self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1 - self.beta2) * (grad_data ** 2)
    
        # Compute bias correction
        bias_correction1 = 1 - self.beta1 ** self.step_count
        bias_correction2 = 1 - self.beta2 ** self.step_count
    
        # Compute bias-corrected moments
        m_hat = self.m_buffers[i] / bias_correction1
        v_hat = self.v_buffers[i] / bias_correction2
    
        return m_hat, v_hat

    def step(self):
        """
        Perform AdamW update step by composing helpers with decoupled weight decay.

        KEY DIFFERENCE from Adam:
        - NO weight decay added to gradient before moment update
        - Weight decay applied directly to parameter AFTER gradient update
        """
        self.step_count += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Extract gradient using shared helper
            grad_data = self._extract_gradient(param)

            # Update moments using PURE gradients (no weight decay mixed in)
            m_hat, v_hat = self._update_moments(i, grad_data)

            # Apply gradient-based update
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Apply decoupled weight decay (separate from gradient update)
            if self.weight_decay != 0:
                param.data = param.data * (1 - self.lr * self.weight_decay)
