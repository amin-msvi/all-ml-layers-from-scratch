"""
## Introduction: What is Automatic Differentiation?

Automatic differentiation (autograd) is the magic that makes neural networks learn.
Instead of manually computing gradients for every parameter, autograd tracks operations and automatically computes gradients via the chain rule.

### The Challenge
To train a model, I need:
```
Loss = f(W₃, f(W₂, f(W₁, x)))
∂Loss/∂W₁ = ?  ∂Loss/∂W₂ = ?  ∂Loss/∂W₃ = ?
```

### The Solution: Computational Graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
```

INSIGHTS:
    - Each [operation] stores how to compute its backward pass.
      The chain rule automatically flows gradients through the entire graph.
    - Each operation records how to compute its backward pass. The chain rule connects them all.

### Mathematical Foundation
For composite functions: f(g(x)), the derivative is:
```
df/dx = (df/dg) × (dg/dx)
```

### Computational Graph Example
```
Simple computation: L = (x * y + 5)²

Forward Pass:
  x=2 ──┐
        ├──[×]──→ z=6 ──[+5]──→ w=11 ──[²]──→ L=121
  y=3 ──┘

Backward Pass (Chain Rule in Action):

Let's create the nodes in the computation graph first:
L = (x * y + 5)²:
    -> z = x * y
    -> w = (z + 5)
    -> L = w²

∂L/∂x = ∂L/∂w × ∂w/∂z × ∂z/∂x
      =   2w  ×   1   ×   y
      = 2(11) ×   1   ×   3 = 66

∂L/∂y = ∂L/∂w × ∂w/∂z × ∂z/∂y
      =  2w   ×   1   ×  x
      = 2(11) ×   1   ×  2 =  44

Gradient Flow Visualization:
  ∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←──[²]←── ∇L=1
  ∇y=44 ←──┘
```

### Memory Layout During Backpropagation
```
Computation Graph Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Forward Pass (stored for backward)                      │
├─────────────────────────────────────────────────────────┤
│ Node 1: x=2 (leaf, requires_grad=True) │ grad: None→66  │
│ Node 2: y=3 (leaf, requires_grad=True) │ grad: None→44  │
│ Node 3: z=x*y (MulFunction)            │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (AddFunction)            │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w² (PowFunction)             │ grad: 1        │
│         saved: (w=11)                  │ inputs: [w]    │
└─────────────────────────────────────────────────────────┘

Memory Cost: 2× parameters (data + gradients) + graph overhead

====================================================================================================

## Implementation: Building the Autograd Engine

Let's implement the autograd system step by step. We'll enhance the existing Tensor class and create supporting infrastructure.

### The Function Architecture
Every differntiable operation needs two things:
1. **Forward pass**: Compute the result
2. **Backward pass**: Compute gradient for inputs


```
Function Class Design:
┌─────────────────────────────────────┐
│ Function (Base Class)               │
├─────────────────────────────────────┤
│ • saved_tensors    ← Store data     │
│ • apply()          ← Compute grads  │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Backward│ │Backward│ │Backward│ │Backward│
└────────┘ └────────┘ └────────┘ └────────┘
```
Each operation inherits from Function and implements specific gradient rules.

### Function Base Class - The Foundation of Autograd

The Function class is the foundation that makes autograd possible.
Every differentiable operation (addition, multiplication, etc.) inherits from this class.

**Why Functions Matter**:
    - The remember inputs needed for backward pass
    - They implement gradient computation via apply()
    - They connect to form computation graphs
    - They enable the chain rule to flow gradients

**The Pattern**:
    - Forward:   inputs --> Function.forward() --> output
    - Backward:  grad_output --> Function.apply() --> grad_inputs

This pattern enables the chain rule to flow gradients through complex computations.
"""


import inspect

import numpy as np
rng = np.random.default_rng(7)
from typing import Optional, List, Tuple
import sys
import os

from core.tensor import Tensor


# ===== Global Gradient Tracking Flag =====
# Why this exists: During inference or parameter updates, we don't need to build
# computation graphs. Skipping graph construction saves memory and time.
# This matches PyTorch's torch.no_grad() behavior.
_GRAD_TRACKING_ENABLED = True

EPSILON = 1e-7  # Small perturbation for numerical gradient computation


class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients (add, multiply, matmul, etc.)
    will inherit from this class and implement the apply() method.

    **Key Concepts**:
        - **saved_tensors**: Store inputs needed for backward pass
        - **apply()**: Compute gradients using chain rule
    """

    def __init__(self, *tensors):
        """
        Initialize function with input tensors.

        Args:
            *tensors: Input tensors that will be saved for backward pass
        """
        self.saved_tensors = tensors

    def apply(self, grad_output):
        """
        Compute gradients for inputs.

        Args:
            grad_output: Gradient flowing backward from the output

        Returns:
            Tuple of gradients for each input tensor
        """
        raise NotImplementedError("Each Function must implement apply() method.")


def _reduce_broadcast_grad(grad, original_shape):
    """
    Reduce gradient to match original tensor shape after broadcasting.

    EXAMPLES:
    >>> # Bias case: (32, 128) -> (128,)
    >>> grad = np.ones((32, 128))  # just an example
    >>> reduced = _reduce_broadcast_grad(grad, (128,))
    >>> reduced.shape  # (128,)

    >>> # Singleton case: (32, 10, 5) -> (10, 1)
    >>> grad = np.ones((32, 10, 5))
    >>> reduced = _reduced_broadcast_grad(grad, (10, 1))
    >>> reduced.shape  # (10, 1)
    """

    # STEP 1: Remove leading dimensions that weren't in original tensor
    # Example: grad (32, 128) with original (128,) -> sum over axis 0
    while grad.ndim > len(original_shape):
        grad = grad.sum(axis=0)

    # STEP 2: Collapse dimensions where original had size 1
    # Example: grad (10, 5) with original (10, 1) -> sum over axis 1 with keepdims
    for i in range(len(original_shape)):
        if original_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad


class AddBackward(Function):
    """
    Gradient computation for tensor addition.

    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1

    **Key Insight:** Addition distributes gradients equally to both inputs.
    The gradient flowing backward is passed unchanged to each input.

    **Broadcasting Handling**: When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.
    """

    def apply(self, grad_output):
        """
        Compute gradients for addition.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two inputs.

        **Mathematical Foundation:**
            - ∂(a+b)/∂a = 1 → grad_a = grad_output
            - ∂(a+b)/∂b = 1 → grad_b = grad_output


        EXAMPLE (Same Shape):
        >>> a = Tensor([1, 2, 3], requires_grad=True)
        >>> b = Tensor([4, 5, 6], requires_grad=True)
        >>> z = a + b  # z = [5, 7, 9]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # Result: grad_a = [1, 1, 1], grad_b = [1, 1, 1]

        EXAMPLE (Broadcasting):
        >>> x = Tensor(np.ones((32, 128)), requires_grad=True)
        >>> bias = Tensor(np.zeros(128), requires_grad=True)
        >>> y = x + bias  # bias broadcasts to (32, 128)
        >>> # During backward: grad_output shape (32, 128)
        >>> # grad_bias must be reduced to (128,) by summing over batch
        >>> # Result: grad_x shape (32, 128), grad_bias shape (128,)
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input
        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class MulBackward(Function):
    """
    Gradient computation for tensor multiplication.

    **Mathematical Rule:** If z = a * b (element-wise), then
        - ∂z/∂a = b  (gradient wrt `a` equals the other input)
        - ∂z/∂b = a  (gradient wrt `b` equals the other input)

        By chain rule:
            ∂Loss/∂a = grad_output * b
            ∂Loss/∂b = grad_output * a

    **Key Insight:** Each input's gradient equals the gradient output
    multiplied by the OTHER input's value (product rule).

    **Applications:** Used in weight scaling, binary masking,
    and anywhere element-wise multiplication occurs.
    """
    def apply(self, grad_output):
        """
        Compute gradients for multiplication.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two inputs

        EXAMPLE:
        >>> a = Tensor([2, 3], requires_grad=True)
        >>> b = Tensor([4, 5], requires_grad=True)
        >>> z = a * b  # z = [8, 15]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1] * [4, 5] = [4, 5]
        >>> # grad_b = [1, 1] * [2, 3] = [2, 3]
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output * b
        if isinstance(a, Tensor) and a.requires_grad:
            if isinstance(b, Tensor):
                grad_a = grad_output * b.data
            else:
                grad_a = grad_output * b

            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        # Gradient for second input: grad_output * a
        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = grad_output * a.data
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class SubBackward(Function):
    """
    Gradient computation for tensor subtraction.

    **Mathematical Rule:** If z = a - b, then ∂z/∂a = 1 and ∂z/∂b = -1
    """
    def apply(self, grad_output):
        """
        Compute gradients for subtraction.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) where grad_b is negated

        EXAMPLE:
        >>> a = Tensor([5, 7], requires_grad=True)
        >>> b = Tensor([2, 3], requires_grad=True)
        >>> z = a - b  # z = [3, 4]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1], grad_b = -[1, 1] = [-1, -1]
        """

        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            grad_a = grad_output  # ∂(a-b)/∂a = 1
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            grad_b = -grad_output  # ∂(a-b)/∂b = -1 (note the negative!)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b


class DivBackward(Function):
    """
    Gradient computation for tensor division.

    **Mathematical Rule:** If z = a / b, then:
    - ∂z/∂a = 1/b
    - ∂z/∂b = -a/b²
    """

    def apply(self, grad_output):
        """
        Compute gradients for division using quotient rule.

        Returns:
            Tuple of (grad_a, grad_b)

        EXAMPLE:
        >>> a = Tensor([8.0, 12.0], requires_grad=True)
        >>> b = Tensor([2.0, 3.0], requires_grad=True)
        >>> z = a / b  # z = [4.0, 4.0]
        >>> # During backward: grad_output = [1, 1]
        >>> # grad_a = [1, 1] / [2, 3] = [0.5, 0.333...]
        >>> # grad_b = -[1, 1] * [8, 12] / ([2, 3]²) = [-2, -1.333...]
        """
        ### BEGIN SOLUTION
        a, b = self.saved_tensors
        grad_a = grad_b = None

        if isinstance(a, Tensor) and a.requires_grad:
            # ∂(a/b)/∂a = 1/b
            if isinstance(b, Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b
            # Handle broadcasting: reduce gradient to match original shape
            grad_a = _reduce_broadcast_grad(grad_a, a.data.shape)

        if isinstance(b, Tensor) and b.requires_grad:
            # ∂(a/b)/∂b = -a/b²
            grad_b = -grad_output * a.data / (b.data ** 2)
            # Handle broadcasting: reduce gradient to match original shape
            grad_b = _reduce_broadcast_grad(grad_b, b.data.shape)

        return grad_a, grad_b
        

class MatmulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    **Mathematical Rule:** If Z = A @ B, then:
        - ∂Z/∂A = grad_Z @ B.T
        - ∂Z/∂B = A.T @ grad_Z

    **Key Insight:** Matrix multiplication gradients involve transposing
    one input and multiplying with the gradient output.

    **Applications:** Core operation in neural networks for computing outputs
    in linear layers and combining feature representations.
    """
    def apply(self, grad_output):
        """
        Compute gradients for matrix multiplication.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a, grad_b) for the two matrix inputs

        **Mathematical Foundation:**
        - ∂(A@B)/∂A = grad_output @ B.T
        - ∂(A@B)/∂B = A.T @ grad_output

        **Batched Operation:** For 3D+ tensors, we transpose only the last two
        dimensions using np.swapaxes, preserving batch dimensions.

        EXAMPLE:
        >>> A = Tensor([[1, 2]], requires_grad=True)    # (1, 2)
        >>> B = Tensor([[3], [4]], requires_grad=True)  # (2, 1)
        >>> C = A @ B  # (1, 1), result = [[11]]
        >>> # During backward: grad_output = [[1]]
        >>> # grad_A = [[1]] @ [[3, 4]] = [[3, 4]]
        >>> # grad_B = [[1, 2]].T @ [[1]] = [[1], [2]]
        """
        a, b = self.saved_tensors
        grad_a = grad_b = None

        # Gradient for first input: grad_output @ b.T
        if isinstance(a, Tensor) and a.requires_grad:
            if b.data.ndim >= 2:
                # Batched: transpose only the last two dims
                b_T = np.swapaxes(b.data, -2, -1)
                grad_a = np.matmul(grad_output, b_T)
            else:
                # 1D b: A(m,k) @ b(k,) -> out(m,)
                # grad_A = outer(grad_output, b): (m,) x (k,) -> (m, k)
                grad_a = np.outer(grad_output, b.data)

        # Gradient for second input: a.T @ grad_output
        if isinstance(b, Tensor) and b.requires_grad:
            if a.data.ndim >= 2:
                # Batched: transpose only the last two dims
                a_T = np.swapaxes(a.data, -2, -1)
                grad_b = np.matmul(a_T, grad_output)
            else:
                # 1D a: a(k,) @ B(k,n) -> out(n,)
                # grad_B = outer(a, grad_output): (k,) x (n,) -> (k, n)
                grad_b = np.outer(a.data, grad_output)


class TransposeBackward(Function):
    """
    Gradient computation for transpose operation.

    **Mathematical Rule:** If Y = X.T, then:
    - ∂Y/∂X = grad_Y.T

    **Key Insight:** The gradient of transpose is just transpose the gradient!
    This is because transpose is a linear operation that just rearranges elements.

    **Applications:** Used in weight gradient computation (W.T), data reshaping,
    and any operation that needs to swap matrix dimensions.
    """

    def __init__(self, tensor, dim0, dim1):
        """
        Args:
            tensor: Input tensor
            dim0: First dimension to swap (None for default)
            dim1: Second dimension to swap (None for default)
        """
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1

    def apply(self, grad_output):
        """
        Compute gradient for transpose.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.T)/∂X = grad_output.T
        - Just transpose the gradient back!

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)
        >>> Y = X.transpose()  # [[1, 3], [2, 4]]
        >>> # During backward: grad_output = [[a, b], [c, d]]
        >>> # grad_X = grad_output.T = [[a, c], [b, d]]
        """
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                # Default: transpose last two dimensions
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2], axes[-1] = axes[-1], axes[-2]
                    grad_x = np.transpose(grad_output, axes)
            else:
                # Specific dimensions: swap them back
                axes = list(range(grad_output.ndim))
                axes[self.dim0], axes[self.dim1] = axes[self.dim1], axes[self.dim0]
                grad_x = np.transpose(grad_output, axes)

        return (grad_x,)


class PermuteBackward(Function):
    """
    Gradient computation for arbitrary axis permutation (general transpose).

    **Mathematical Rule:** If Y = X.permute(axes), then:
    - ∂Y/∂X = grad_Y.permute(inverse_axes)

    **Example:** If axes = (0, 2, 1, 3), the inverse is (0, 2, 1, 3) (self-inverse).
    More generally, if axes = (2, 0, 1), the inverse is (1, 2, 0).

    **Key Insight:** To reverse a permutation, we need to know where each axis went.
    If axis i went to position axes[i], then in the inverse, position axes[i] should go to i.

    **Applications:** Rearranging tensor dimensions for different computation
    patterns (e.g., swapping batch and feature dimensions).
    """

    def __init__(self, tensor, axes):
        """
        Args:
            tensor: Input tensor
            axes: Tuple of axis indices defining the permutation
        """
        super().__init__(tensor)
        self.axes = axes
        # Compute inverse permutation: if axes[i] = j, then inverse_axes[j] = i
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self, grad_output):
        """
        Compute gradient for permutation.

        The gradient is permuted back using the inverse permutation.

        **Mathematical Foundation:**
        - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)

        EXAMPLE:
        >>> X = Tensor([[[1, 2], [3, 4]]], requires_grad=True)  # (1, 2, 2)
        >>> Y = X.permute((0, 2, 1))  # Swap last two dims → (1, 2, 2)
        >>> # During backward: inverse_axes computed in __init__
        >>> # grad_X = np.transpose(grad_output, inverse_axes)
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Permute gradient back to original axis order
            grad_x = np.transpose(grad_output, self.inverse_axes)

        return (grad_x,)


class SliceBackward(Function):
    """
    Gradient computation for tensor slicing/indexing operations.

    **Mathematical Rule:** If Y = X[key], then:
    - ∂Loss/∂X[key] = grad_output
    - ∂Loss/∂X[other positions] = 0

    **Key Insight:** Slicing is a masking operation. The backward
    places gradients back into the original tensor positions, with
    zeros everywhere else.

    **Applications:** Sequence slicing, batch selection, and selecting subsets
    of tensor data for computation.

    **Examples:**
    >>> x = Tensor([1, 2, 3, 4, 5], requires_grad=True)
    >>> y = x[:3]  # Slice first 3 elements
    >>> loss = y.sum()
    >>> loss.backward()
    >>> # x.grad = [1, 1, 1, 0, 0] - gradients only for sliced positions
    """

    def __init__(self, tensor, key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index, slice, tuple of slices, etc.)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self, grad_output):
        """
        Compute gradient for slicing operation.

        Args:
            grad_output: Gradient flowing backward from sliced output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - Slicing extracts a subset of elements
        - Backward scatters gradients back to original positions
        - Unsliced positions receive zero gradient

        **Example:**
        If X = [a, b, c, d, e] and Y = X[1:4] = [b, c, d]
        Then dL/dX = [0, dL/db, dL/dc, dL/dd, 0]

        EXAMPLE:
        >>> X = Tensor([1, 2, 3, 4, 5], requires_grad=True)
        >>> Y = X[:3]  # Slice first 3 elements → [1, 2, 3]
        >>> # During backward: grad_output = [1, 1, 1]
        >>> # grad_X = [1, 1, 1, 0, 0] (gradients only for sliced positions)
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Create gradient array with same shape as original tensor
            grad_input = np.zeros(self.original_shape, dtype=np.float32)

            # Place gradients back into the sliced positions
            # This is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)


class ReshapeBackward(Function):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape), then:
    - ∂Y/∂X = grad_Y.reshape(X.shape)

    **Key Insight:** Reshape just rearranges the same elements.
    The gradient is simply reshaped back to the original shape!

    **Applications:** Flattening tensors for linear layers, reshaping
    between convolutional and dense layers.
    """

    def __init__(self, tensor, original_shape):
        """
        Args:
            tensor: Input tensor
            original_shape: Shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """
        Compute gradient for reshape.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple with single gradient for input tensor

        **Mathematical Foundation:**
        - ∂(X.reshape(...))/∂X = grad_output.reshape(X.shape)
        - Just reshape the gradient back!

        EXAMPLE:
        >>> X = Tensor([[1, 2], [3, 4]], requires_grad=True)  # (2, 2)
        >>> Y = X.reshape(4)  # [1, 2, 3, 4]
        >>> # During backward: grad_output = [1, 1, 1, 1]
        >>> # grad_X = grad_output.reshape((2, 2)) = [[1, 1], [1, 1]]
        """
        ### BEGIN SOLUTION
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x, Tensor) and x.requires_grad:
            # Reshape gradient back to original shape
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)


class SumBackward(Function):
    """
    Gradient computation for tensor sum.

    **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

    **Key Insight:** Sum distributes the gradient equally to all input elements.
    The gradient is broadcast from the reduced output back to input shape.

    **Applications:** Used in loss functions, mean operations, and
    anywhere tensor reduction occurs.
    """

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output):
        """
        Compute gradients for sum operation.

        Args:
            grad_output: Gradient flowing backward from output

        Returns:
            Tuple containing gradient for the input tensor

        **Mathematical Foundation:**
        - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output

        EXAMPLE:
        >>> X = Tensor([1, 2, 3], requires_grad=True)
        >>> Y = X.sum()  # Y = 6 (scalar)
        >>> # During backward: grad_output = 1 (scalar)
        >>> # grad_X = [1, 1, 1] * 1 = [1, 1, 1]
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # For axis-reduced sums, expand grad_output back along the summed
            # axis before broadcasting, so each row/column gets its own gradient.
            if self.axis is not None and not self.keepdims:
                grad_output = np.expand_dims(grad_output, axis=self.axis)
            return np.ones_like(tensor.data) * grad_output,
        return None,




class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """

    def __init__(self, input_tensor):
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for ReLU.

        EXAMPLE:
        >>> X = Tensor([-2, -1, 0, 1, 2], requires_grad=True)
        >>> Y = relu(X)  # [0, 0, 0, 1, 2]
        >>> # During backward: grad_output = [1, 1, 1, 1, 1]
        >>> # relu_mask = [0, 0, 0, 1, 1] (1 where x > 0)
        >>> # grad_X = [0, 0, 0, 1, 1]
        """
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # ReLU gradient: 1 if x > 0, else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad


class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation.

    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))
    """
    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to sigmoid
            output_tensor: Output of sigmoid (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """
        Compute gradient for sigmoid.

        EXAMPLE:
        >>> X = Tensor([0.0], requires_grad=True)
        >>> Y = sigmoid(X)  # Y = 0.5
        >>> # During backward: grad_output = 1
        >>> # σ'(0) = 0.5 * (1 - 0.5) = 0.25
        >>> # grad_X = 1 * 0.25 = 0.25
        """

        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # σ'(x) = σ(x) * (1 - σ(x))
            sigmoid_grad = self.output_data * (1 - self.output_data)
            return grad_output * sigmoid_grad

        return None,


class TanhBackward(Function):
    """
    Gradient computation for tanh activation.

    Tanh: tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
    Derivative: tanh'(x) = 1 - tanh(x)²
    """

    def __init__(self, input_tensor, output_tensor):
        """
        Initialize with both input and output.

        Args:
            input_tensor: Original input to tanh
            output_tensor: Output of tanh (saves recomputation)
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self, grad_output):
        """
        Compute gradient for tanh.

        EXAMPLE:
        >>> X = Tensor([0.0], requires_grad=True)
        >>> Y = tanh(X)  # Y = 0.0
        >>> # During backward: grad_output = 1
        >>> # tanh'(0) = 1 - 0² = 1
        >>> # grad_X = 1 * 1 = 1
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # tanh'(x) = 1 - tanh(x)²
            tanh_grad = 1 - self.output_data ** 2
            return grad_output * tanh_grad,
        return None,
    

class SoftmaxBackward(Function):
    """
    Gradient computation for softmax activation.

    Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    Derivative: ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    For gradient computation:
    grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    **Key Insight:** The gradient depends on all elements of softmax due to
    the normalization, not just the element being differentiated.
    """

    def __init__(self, input_tensor, output_tensor, dim=-1):
        """
        Initialize with input, output, and dimension.

        Args:
            input_tensor: Original input to softmax
            output_tensor: Output of softmax (needed for gradient)
            dim: Dimension along which softmax was applied
        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self, grad_output):
        """
        Compute gradient for softmax.

        Mathematical formula:
        ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum_j(∂L/∂y[j] * softmax[j]))

        This can be vectorized as:
        grad_x = softmax * (grad_y - sum(grad_y * softmax, keepdims=True))

        EXAMPLE:
        >>> X = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        >>> Y = softmax(X)  # [[0.09, 0.24, 0.67]] approximately
        >>> # During backward: grad_output = [[1, 0, 0]]
        >>> # sum_term = sum([1*0.09, 0*0.24, 0*0.67]) = 0.09
        >>> # grad_X[i] = softmax[i] * (grad_output[i] - sum_term)
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            # Compute sum(grad_output * softmax) along the softmax dimension
            sum_term = np.sum(grad_output * self.output_data, axis=self.dim, keepdims=True)

            # Softmax gradient: softmax * (grad_output - sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

            return (grad_x,)
        return (None,)


class GELUBackward(Function):
    """
    Gradient computation for GELU activation.

    GELU: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    **Key Insight:** GELU is smoother than ReLU, providing non-zero gradients
    for negative values, which helps training deep networks.
    """

    def __init__(self, input_tensor):
        """Initialize with input tensor."""
        super().__init__(input_tensor)

    def apply(self, grad_output):
        """
        Compute gradient for GELU.

        Mathematical formula (using approximation):
        ∂gelu/∂x ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * (...)

        Simplified: We compute the derivative numerically or use the formula.
        """
        ### BEGIN SOLUTION
        tensor, = self.saved_tensors

        if isinstance(tensor, Tensor) and tensor.requires_grad:
            x = tensor.data
            # GELU derivative using the sigmoid approximation (matches forward):
            # forward: gelu(x) = x * sigmoid(1.702 * x)
            # d/dx [x * sig(1.702x)] = sig(1.702x) + x * 1.702 * sig(1.702x) * (1 - sig(1.702x))
            # Numerically stable sigmoid (matches activations.Sigmoid.forward): each
            # branch keeps its exponent <= 0, so large |x| never overflows np.exp.
            z = 1.702 * x
            with np.errstate(over="ignore", invalid="ignore"):
                sig = np.where(
                    z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)),
                )
            gelu_grad = sig + x * 1.702 * sig * (1.0 - sig)

            return (grad_output * gelu_grad,)
        return (None,)


class MSEBackward(Function):
    """
    Gradient computation for Mean Squared Error Loss.

    MSE: L = mean((predictions - targets)²)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for MSE loss.

        EXAMPLE:
        >>> predictions = Tensor([2.0, 3.0], requires_grad=True)
        >>> targets = Tensor([1.0, 2.0])
        >>> loss = MSE(predictions, targets)  # (1² + 1²)/2 = 1.0
        >>> # During backward: grad_output = 1
        >>> # grad = 2 * ([2, 3] - [1, 2]) / 2 = [1, 1]
        """
        ### BEGIN SOLUTION
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            # Gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return grad * grad_output,
        return None,



class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = -[y*log(p) + (1-y)*log(1-p)]
    Derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)
    """

    def __init__(self, predictions, targets):
        """Initialize with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self, grad_output):
        """
        Compute gradient for BCE loss.

        EXAMPLE:
        >>> predictions = Tensor([0.7, 0.3], requires_grad=True)
        >>> targets = Tensor([1.0, 0.0])
        >>> loss = BCE(predictions, targets)
        >>> # During backward: grad = (p - y) / (p * (1-p) * N)
        """
        ### BEGIN SOLUTION
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPSILON
            p = np.clip(predictions.data, eps, 1 - eps)
            y = self.targets_data

            # Gradient: (p - y) / (p * (1-p) * N)
            grad = (p - y) / (p * (1 - p) * self.num_samples)

            return grad * grad_output,
        return None,


def _stable_softmax(logits_data):
    """
    Compute softmax probabilities with numerical stability.

    Computing softmax naively as `exp(x) / sum(exp(x))` overflows for large values.
    Subtracts the max value per row before exponentiating to prevent overflow.

    ```
    Naive (overflows):     softmax(x) = exp(x) / sum(exp(x))
    Stable (safe):         softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    
    Why it works:
    exp(x - max(x)) / sum(exp(x - max(x)))
    = exp(x) * exp(-max(x)) / (sum(exp(x)) * exp(-max(x)))
    = exp(x) / sum(exp(x))
    ```

    Args:
        logits_data: numpy array of shape (batch_size, num_classes)

    Returns:
        numpy array of softmax probabilities, same shape as input

    EXAMPLE:
    >>> logits = np.array([[2.0, 1.0, 0.1]])
    >>> probs = _stable_softmax(logits)
    >>> # probs ≈ [[0.659, 0.242, 0.099]]
    >>> # Each row sums to 1.0
    """
    ### BEGIN SOLUTION
    max_logits = np.max(logits_data, axis=1, keepdims=True)
    exp_logits = np.exp(logits_data - max_logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _one_hot_encode(targets, batch_size, num_classes):
    """
    Convert class indices to one-hot vectors.

    Args:
        targets: numpy array of integer class indices, shape (batch_size,)
        batch_size: number of samples
        num_classes: number of classes

    Returns:
        numpy array of one-hot vectors, shape (batch_size, num_classes)

    EXAMPLE:
    >>> targets = np.array([0, 2, 1])
    >>> one_hot = _one_hot_encode(targets, batch_size=3, num_classes=3)
    >>> # one_hot = [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    """
    ### BEGIN SOLUTION
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), targets] = 1.0
    return one_hot


class CrossEntropyBackward(Function):
    """
    Gradient computation for Cross-Entropy Loss.

    The cross-entropy gradient combines three sub-computations:
    1. **Stable softmax**: Convert raw logits to probabilities
    2. **One-hot encoding**: Convert target indices to indicator vectors
    3. **Gradient formula**: `(softmax - one_hot) / batch_size`

    CrossEntropy: L = -mean(log_softmax(logits)[targets])

    The gradient with respect to logits is remarkably elegant:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N

    This is one of the most beautiful results in machine learning:
    - The gradient is simply the difference between predictions and targets
    - It naturally scales with how wrong we are
    - It's numerically stable when computed via softmax

    ```
    Logits: [2.0, 1.0, 0.1]     Target: class 0
    
    Step 1 - Softmax:   [0.659, 0.242, 0.099]
    Step 2 - One-hot:   [1.000, 0.000, 0.000]
    Step 3 - Gradient:  [0.659-1, 0.242-0, 0.099-0] / 1 = [-0.341, 0.242, 0.099]
    ```
    """

    def __init__(self, logits, targets):
        """Initialize with logits and target class indices."""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self, grad_output):
        """
        Compute gradient for cross-entropy loss.

        Uses helper functions for each sub-computation:
        - _stable_softmax(): Converts logits to probabilities (numerically stable)
        - _one_hot_encode(): Converts target indices to one-hot vectors

        EXAMPLE:
        >>> logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
        >>> targets = Tensor([0])  # Correct class is 0
        >>> loss = CrossEntropy(logits, targets)
        >>> # softmax ≈ [0.66, 0.24, 0.10]
        >>> # one_hot = [1, 0, 0]
        >>> # grad = ([0.66, 0.24, 0.10] - [1, 0, 0]) / 1 = [-0.34, 0.24, 0.10]
        """
        ### BEGIN SOLUTION
        logits, = self.saved_tensors

        if isinstance(logits, Tensor) and logits.requires_grad:
            softmax = _stable_softmax(logits.data)
            one_hot = _one_hot_encode(self.targets_data, self.batch_size, self.num_classes)

            # Gradient: (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size

            return grad * grad_output,
        return None,


def is_grad_enabled():
    """Check if gradient tracking is currently enabled.

    Returns True when operations should build computation graphs,
    False when inside a no_grad() context.
    """
    return _GRAD_TRACKING_ENABLED


# TODO: Recall `context` how it works
class no_grad:
    """
    Context manager that disables gradient tracking.

    When entering this context, all operations will skip computation graph
    construction: Tensors produced inside will have requries_grad=False
    regardless of their inputs. This is essential for:
        1. **Inference:** No need to track gradients when making predictions
        2. **Parameter updates:** Optimizers modify weights without recording history
        3. **Memory savings** Skipping graph construction reduces memory usage

    Matches PyTorch's torch.no_grad() API.

    EXAMPLE:
    >>> x = Tensor([2.0], requires_grad=True)
    >>> with no_grad():
    ...    y = x * 2     # No graph built, y.requires_grad = False
    >>> z = x * 3        # Graph IS built (outside no_grad)
    >>> z.backward()     # Works normally

    EXAMPLE:
    >>> with no_grad():
    >>> ...  with no_grad():    # Inner context
    >>> ...        y = x * 2    # Still no graph
    >>> ...  z = x * 3          # Still no graph (outer context active)
    >>> w = x * 4               # Graph IS built (all contexts exited)
    
    """
    def __enter__(self):
        """Save previous state and disable gradient tracking."""
        global _GRAD_TRACKING_ENABLED
        # Save previous so nested context restore correctly
        self._prev_state = _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous gradient tracking state."""
        global _GRAD_TRACKING_ENABLED
        _GRAD_TRACKING_ENABLED = self._prev_state
        return False  # Don't suppress exceptions



def enable_autograd(quiet=False):
    """
    Enable gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.

    Args:
        quiet (bool): If True, suppress status message. Default: False

    What it does:
        - Replaces Tensor operations with gradient-tracking versions
        - Adds backward method for reverse-mode differenciation
        - Enables computation graph building
        - Maintains full backward compatibility

    After calling this:
        - Tensor operations will track computation graphs
        - backward() method becomes available
        - Gradients will flow through operations
        - requires_grad=True enables tracking per tensor

    EXAMPLE:
    >>> enable_autograd()  # Call once
    >>> x = Tensor([2.0], requires_grad=True)
    >>> y = x * 3
    >>> y.backward()
    >>> print(x.grad)  # [3.0]
    """
    if hasattr(Tensor, '_autograd_enabled'):
        # Silently return if alrady enabled - no need to warn
        return

    # STEP 1: Add gradient infrastructure to Tensor
    _original_init = Tensor.__init__

    def gradient_aware_init(self, data, requires_grad=False):
        """Extended Tensor that supports gradient tracking."""
        _original_init(self, data)
        self.requires_grad = requires_grad
        self.grad = None

    # Replace __init__ with gradient-aware version
    Tensor.__init__ = gradient_aware_init

    # Store original operations
    # These are guaranteed to exist from Module 01 (Tensor class)
    _original_add = Tensor.__add__
    _original_radd = Tensor.__radd__
    _original_sub = Tensor.__sub__
    _original_rsub = Tensor.__rsub__
    _original_mul = Tensor.__mul__
    _original_rmul = Tensor.__rmul__
    _original_div = Tensor.__truediv__
    _original_rdiv = Tensor.__rtruediv__
    _original_getitem = Tensor.__getitem__
    _original_sum = Tensor.sum

    # These methods are also guaranteed from Module 01 - trust Single Tensor Class
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    # Helper to safely check requires_grad (handles tensors created before enable_autograd)
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to False for pre-autograd tensors.

        Also returns False when inside a no_grad() context, since we should
        not build computation graphs there.
        """
        if not _GRAD_TRACKING_ENABLED:
            return False
        return getattr(tensor, 'requires_grad', False) if isinstance(tensor, Tensor) else False

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes (for tensors created before enable_autograd)."""
        if isinstance(tensor, Tensor):
            if not hasattr(tensor, 'requires_grad'):
                tensor.requires_grad = False
            if not hasattr(tensor, 'grad'):
                tensor.grad = None


    # Enhanced operations that track gradients
    def tracked_add(self, other):
        """
        Addition with gradient tracking.

        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        # Ensure self has gradient attributes
        _ensure_grad_attrs(self)

        # Keep a Tensor-typed copy for grad bookkeeping, but call the
        # original with `other` in its original form (matching
        # tracked_mul's pattern) so _original_add's own scalar branch
        # stays reachable instead of always seeing a pre-converted Tensor.
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        # Call original operation
        result = _original_add(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = AddBackward(self, other_tensor)

        return result

    def tracked_radd(self, other):
        """
        Scalar-left addition with gradient tracking.

        Delegates to the original __radd__ from Module 01 (which itself
        calls self.__add__(other)) rather than calling tracked_add
        directly, so Module 01's own __radd__ source actually executes.
        Addition is commutative, so this produces the same result either
        way, self.__add__ resolves to tracked_add regardless, since
        that's the currently-installed __add__ by the time this runs.
        """
        _ensure_grad_attrs(self)
        return _original_radd(self, other)

    def tracked_mul(self, other):
        """
        Multiplication with gradient tracking.

        Enhances the original __mul__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        result = _original_mul(self, other)
        _ensure_grad_attrs(result)

        # Pass other_tensor (always a Tensor) so MulBackward can call .data on it.
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = MulBackward(self, other_tensor)

        return result


    def tracked_rmul(self, other):
        """
        Scalar-left multiplication with gradient tracking.

        Delegates to the original __rmul__ from Module 01 (which itself
        calls self.__mul__(other)), same reasoning as tracked_radd above.
        """
        _ensure_grad_attrs(self)
        return _original_rmul(self, other)

    def tracked_matmul(self, other):
        """
        Matrix multiplication with gradient tracking.

        Enhances the original matmul method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        # Call original matmul from Module 01
        result = _original_matmul(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = MatmulBackward(self, other)

        return result

    def tracked_transpose(self, dim0=None, dim1=None):
        """
        Transpose with gradient tracking.

        Enhances the original transpose method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original transpose from Module 01
        result = _original_transpose(self, dim0, dim1)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = TransposeBackward(self, dim0, dim1)

        return result

    def tracked_reshape(self, *shape):
        """
        Reshape with gradient tracking.

        Enhances the original reshape method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)
        original_shape = self.shape

        # Call original reshape from Module 01
        result = _original_reshape(self, *shape)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = ReshapeBackward(self, original_shape)

        return result

    def tracked_sub(self, other):
        """
        Subtraction with gradient tracking.

        Enhances the original __sub__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Keep a Tensor-typed copy for grad bookkeeping, but call the
        # original with `other` in its original form (matching
        # tracked_mul's pattern) so _original_sub's own scalar branch
        # stays reachable instead of always seeing a pre-converted Tensor.
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        # Call original operation
        result = _original_sub(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = SubBackward(self, other_tensor)

        return result

    def tracked_rsub(self, other):
        """
        Scalar-left subtraction with gradient tracking.

        Unlike __radd__/__rmul__, __rsub__ computes a genuinely different
        formula (other - self, not self - other), so it can't just
        forward to tracked_sub without ever running Module 01's actual
        __rsub__ body. Calls the original directly (with other in
        whatever form the caller passed, so both its Tensor and
        non-Tensor branches stay reachable) to get the value, then
        attaches gradient tracking the same way tracked_sub(other, self)
        already did.
        """
        _ensure_grad_attrs(self)

        # Call original __rsub__ from Module 01: computes other - self
        result = _original_rsub(self, other)
        _ensure_grad_attrs(result)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = SubBackward(other, self)

        return result

    def tracked_div(self, other):
        """
        Division with gradient tracking.

        Enhances the original __truediv__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        # Keep a Tensor-typed copy for grad bookkeeping, but call the
        # original with `other` in its original form (matching
        # tracked_mul's pattern) so _original_div's own scalar branch
        # stays reachable instead of always seeing a pre-converted Tensor.
        if not isinstance(other, Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        # Call original operation
        result = _original_div(self, other)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self) or _get_requires_grad(other_tensor):
            result.requires_grad = True
            result._grad_fn = DivBackward(self, other_tensor)

        return result

    def tracked_rdiv(self, other):
        """
        Scalar-left division with gradient tracking.

        Same reasoning as tracked_rsub: __rtruediv__ computes a
        genuinely different formula (other / self, not self / other),
        so it calls the original directly to get the value (keeping
        both its Tensor and non-Tensor branches reachable), then
        attaches gradient tracking the same way tracked_div(other, self)
        already did.
        """
        _ensure_grad_attrs(self)

        # Call original __rtruediv__ from Module 01: computes other / self
        result = _original_rdiv(self, other)
        _ensure_grad_attrs(result)

        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        if _get_requires_grad(self) or _get_requires_grad(other):
            result.requires_grad = True
            result._grad_fn = DivBackward(other, self)

        return result

    def tracked_getitem(self, key):
        """
        Indexing/slicing with gradient tracking.

        Enhances the original __getitem__ method to build computation graphs
        when requires_grad=True for the input.
        """
        _ensure_grad_attrs(self)

        # Call original __getitem__ from Module 01
        result = _original_getitem(self, key)
        _ensure_grad_attrs(result)

        # Track gradient if needed
        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SliceBackward(self, key)

        return result

    def sum_op(self, axis=None, keepdims=False):
        """
        Sum operation with gradient tracking.

        Delegates to the original .sum() from Module 01 for the actual
        computation, rather than reimplementing it, so Module 01's own
        sum() source stays reachable and testable.
        """
        _ensure_grad_attrs(self)

        result = _original_sum(self, axis=axis, keepdims=keepdims)
        _ensure_grad_attrs(result)

        if _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = SumBackward(self, axis=axis, keepdims=keepdims)

        return result

    def _get_caller_tensor_names():
        """Map tensors to the variable names visible where backward() was called."""
        variable_names = {}
        current_frame = inspect.currentframe()
        backward_frame = current_frame.f_back if current_frame is not None else None
        caller_frame = backward_frame.f_back if backward_frame is not None else None

        if caller_frame is not None:
            for scope in (caller_frame.f_globals, caller_frame.f_locals):
                for name, value in scope.items():
                    if isinstance(value, Tensor):
                        names = variable_names.setdefault(id(value), [])
                        if name not in names:
                            names.append(name)

        del caller_frame, backward_frame, current_frame
        return variable_names

    def backward(self, gradient=None, retain_graph=False, show_graph=False):
        """
        Compute gradients via backpropagation.

        This is the key method that makes training possible!
        It implements reverse-mode automatic differentiation.

        **Algorithm:**
            1. Initialize gradient if not provided (for scalar outputs)
            2. Accumulate gradient in self.grad
            3. If this tensor has a _grad_fn, call it to propagate gradients
            4. Recursively call backward() on parent tensors
            5. Release computation graph (unless retain_graph=True)

        **Args:**
            gradient: External gradient to seed backpropagation. If None, assumes
                scalar output and uses ones_like as the seed.
            retain_graph: If False (default), releases the computation graph after
                backward to free memory. Set True if you need to call backward()
                multiple times on the same graph (e.g., for higher-order gradients).
                Matches PyTorch's retain_graph parameter.
            show_graph: If True, prints the computation graph before backpropagation.
                Tensor names are included when they are visible in the caller's scope.

        **Example:**
        >>> x = Tensor([2.0], requires_grad=True)
        >>> y = x * 3
        >>> y.backward()  # Computes gradients for x, then releases graph
        >>> print(x.grad)  # [3.0]
        >>> # y.backward()  # Would fail — graph already released!

        # To backward twice, use retain_graph=True:
        >>> y2 = x * 3
        >>> y2.backward(retain_graph=True)  # Graph kept alive
        >>> y2.backward()  # Works! (releases graph this time)
        """
        # Ensure gradient attributes exist
        _ensure_grad_attrs(self)

        if show_graph:
            print("Computation graph:")
            print("--------------------\n")
            _print_computation_graph(
                self, variable_names=_get_caller_tensor_names()
            )

        # Only compute gradients if required
        if not _get_requires_grad(self):
            return

        # Initialize gradient if not provided (for scalar outputs)
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalar tensor without gradient argument.\n"
                    f"  Tensor shape: {self.shape}\n"
                    f"  Issue: For non-scalar outputs, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function."
                )

        # Initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        # Handle broadcasting: sum gradient to match self.data shape
        # This happens when operations broadcast tensors (e.g., adding bias to batch)
        if gradient.shape != self.grad.shape:
            # Step 1: Remove extra leading dimensions added during forward pass
            # Example: gradient (batch_size, features) → self.grad (features,)
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)

            # Step 2: Sum over dimensions that were size-1 in original tensor
            # Example: bias with shape (1,) broadcast to (batch_size,) during forward
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i, keepdims=True)

        self.grad += gradient

        # Propagate gradients through computation graph
        # _grad_fn is set by autograd enhancement when tensor is created from an operation
        grad_fn = getattr(self, '_grad_fn', None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            # Recursively call backward on parent tensors
            for tensor, grad in zip(grad_fn.saved_tensors, grads):
                if isinstance(tensor, Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad, retain_graph=retain_graph)

            # Release computation graph to free memory (matches PyTorch's default)
            # Why: The graph stores references to all intermediate tensors. Without
            # cleanup, these references prevent garbage collection, causing memory
            # to grow linearly with the number of training steps.
            if not retain_graph:
                self._grad_fn = None

    def _print_computation_graph(
        tensor, prefix="", is_last=True, seen=None, variable_names=None, is_root=True
    ):
        """Print the computation graph from an output tensor to its inputs."""
        if seen is None:
            seen = set()
        if variable_names is None:
            variable_names = {}

        grad_fn = getattr(tensor, "_grad_fn", None)
        operation = type(grad_fn).__name__ if grad_fn is not None else "Leaf"
        connector = "" if is_root else ("└── " if is_last else "├── ")
        names = variable_names.get(id(tensor), [])
        name = f"name={','.join(names)} " if names else ""

        if id(tensor) in seen:
            print(f"{prefix}{connector}{operation} {name}shape={tensor.shape} [shared]")
            return

        seen.add(id(tensor))

        print(
            f"{prefix}{connector}{operation} "
            f"{name}"
            f"shape={tensor.shape} "
            f"requires_grad={getattr(tensor, 'requires_grad', False)}"
        )

        if grad_fn is None:
            return

        parents = [parent for parent in grad_fn.saved_tensors if isinstance(parent, Tensor)]
        child_prefix = prefix if is_root else prefix + ("    " if is_last else "│   ")
        for index, parent in enumerate(parents):
            _print_computation_graph(
                parent,
                prefix=child_prefix,
                is_last=index == len(parents) - 1,
                seen=seen,
                variable_names=variable_names,
                is_root=False,
            )

    def zero_grad(self):
        """
        Reset gradients to zero.

        Call this before each backward pass to prevent gradient accumulation
        from previous iterations.
        """
        self.grad = None

    # Install enhanced operations
    Tensor.__add__ = tracked_add
    Tensor.__radd__ = tracked_radd
    Tensor.__sub__ = tracked_sub
    Tensor.__rsub__ = tracked_rsub
    Tensor.__mul__ = tracked_mul
    Tensor.__rmul__ = tracked_rmul
    Tensor.__truediv__ = tracked_div
    Tensor.__rtruediv__ = tracked_rdiv
    Tensor.__getitem__ = tracked_getitem
    Tensor.matmul = tracked_matmul
    Tensor.transpose = tracked_transpose
    Tensor.reshape = tracked_reshape
    Tensor.sum = sum_op
    Tensor.backward = backward
    Tensor.zero_grad = zero_grad

    # Patch activations and losses to track gradients
    try:
        from core.activations import Sigmoid, ReLU, Softmax, GELU, Tanh
        from core.losses import BinaryCrossEntropyLoss, MSELoss, CrossEntropyLoss

        # Store original methods
        _original_sigmoid_forward = Sigmoid.forward
        _original_relu_forward = ReLU.forward
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_tanh_forward = Tanh.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        _original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self, x):
            """Sigmoid with gradient tracking."""
            # Delegate to the original, numerically stable implementation
            # (same pattern as tracked_softmax_forward/tracked_gelu_forward
            # below) rather than reimplementing sigmoid with a formula that
            # overflows np.exp for large negative x.
            result = _original_sigmoid_forward(self, x)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SigmoidBackward(x, result)

            return result

        def tracked_tanh_forward(self, x):
            """Tanh with gradient tracking."""
            result_data = np.tanh(x.data)
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = TanhBackward(x, result)

            return result

        def tracked_relu_forward(self, x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0, x.data)
            result = Tensor(result_data)

            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result

        def tracked_softmax_forward(self, x, dim=-1):
            """Softmax with gradient tracking."""
            # Call original forward to get result using Tensor operations
            result = _original_softmax_forward(self, x, dim=dim)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = SoftmaxBackward(x, result, dim)

            return result

        def tracked_gelu_forward(self, x):
            """GELU with gradient tracking."""
            # Call original forward to get result
            result = _original_gelu_forward(self, x)

            # Attach the correct gradient function
            if _GRAD_TRACKING_ENABLED and x.requires_grad:
                result.requires_grad = True
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self, predictions, targets):
            """Binary cross-entropy with gradient tracking."""
            # Call original forward to get result using the module's own implementation
            result = _original_bce_forward(self, predictions, targets)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = BCEBackward(predictions, targets)

            return result

        def tracked_mse_forward(self, predictions, targets):
            """MSE loss with gradient tracking."""
            # Call original forward to get result using the module's own implementation
            result = _original_mse_forward(self, predictions, targets)

            if _GRAD_TRACKING_ENABLED and predictions.requires_grad:
                result.requires_grad = True
                result._grad_fn = MSEBackward(predictions, targets)

            return result

        def tracked_ce_forward(self, logits, targets):
            """Cross-entropy loss with gradient tracking."""
            # Call original forward to get result using the module's own implementation
            result = _original_ce_forward(self, logits, targets)

            if _GRAD_TRACKING_ENABLED and logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits, targets)

            return result

        # Install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        Tanh.forward = tracked_tanh_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        # Activations/losses not yet available (happens during module development)
        pass

    # Mark as enabled
    Tensor._autograd_enabled = True

    if not quiet:
        print("Autograd enabled! Tensors now track gradients.")
        print("   - Operations build computation graphs")
        print("   - backward() computes gradients")
        print("   - requires_grad=True enables tracking")
