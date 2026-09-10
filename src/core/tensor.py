"""

## Fundamentals
The Tensor class will support all fundamental operations that NN need:

1. Core Operations
    1.1. Element-wise:
        - Addition (+)
        - Substraction (-)
        - Multiplication (*)
        - Division (/)
    1.2. Matrix Ops
        - Matrix Mult (@)
        - transpose
    1.3. Shape Ops
        - .reshape()
        - .sum()
        - .mean()
        - .max()

2. Broadcasting: Making Tensors Work Together
Broadcasting automatically aligns tensors of different shapes for operations

Examples:
    - Scalar + Vector:
        5 + [1, 2, 3] --> [5, 5, 5] + [1, 2, 3] = [6, 7, 8]
    - Matrix + Vector (row-wise):
        [[1, 2]]   [10]   [[1, 2]]   [[10, 10]]   [[11, 12]]
        [[3, 4]] + [10] = [[3, 4]] + [[10, 10]] = [[13, 14]]     

## Implementation

### Tensor Class Architecture
1. Core Attributes:
    • data: np.array (the numbers)
    • shape: tuple (dimensions)
    • size: int (total elements)
    • dtype: type (float32)
2. Arithmetic Operations:
    • __add__
    • __sub__
    • __mul__
    • __truediv__
    • matmul()
3. Shape Operations
    • reshape()
    • transpose()
    • sum()
    • mean()
    • max()
    • __getitem__ (indexing)
4. Utility Methods:
    • __repr__()
    • __str__()
    • numpy()
    • memory_footprint()
    • ndim
    • numel()
    • contiguous()


## Tensor Initiation Process
Input Data --> Validation --> NumPy Array --> Tensor Wrapper --> Ready for Operations
 [1,2,3]         types          np.array         shape=(3,)         + - * / @ ...
"""

import numpy as np
rng = np.random.default_rng(7)

# Constants for memory calculations
BYTES_PER_FLOAT32 = 4
KB_TO_BYTES = 1024
MB_TO_BYTES = 1024 * 1024


class Tensor:
    """Educational tensor - the foundation of machine learning computation.

    This class provides the core data structure for all ML operations:
    - data: The actual numerical values (NumPy array)
    - shape: Dimensions of the tensor
    - ndim: Number of dimensions (0=scalar, 1=vector, 2=matrix, ...)
    - size: Total number of elements
    - dtype: Data type (float32)

    All arithmetic, matrix, and shape operations are built on this foundation.
    """

    def __init__(self, data):
        if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], Tensor):
            data = np.stack([t.data for t in data])
        self.data = np.array(data, dtype=np.float32)
        self.shape = self.data.shape
        self.size = self.data.size
        self.dtype = self.data.dtype

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.shape})"

    def __str__(self):
        return f"Tensor({self.data})"

    def numpy(self):
        return self.data

    def memory_footprint(self):
        """Calculate exact memory usage in bytes

        Systems Concept: Understanding memory footprint is fundamental to ML systems.
        Before running any operation, engineers should know how much memory is required.

        Returns:
            int: Memory usage in bytes (e.g., 1000x1000 float32 = 4MB)
        """
        return self.data.nbytes

    @property
    def ndim(self):
        return len(self.shape)

    def numel(self):
        """Return total number of elements (Pytorch-compatible)"""
        return self.size

    def contiguous(self):
        """Return a contiguous copy of the tensor data (Pytorch-compatible)"""
        return Tensor(np.ascontiguousarray(self.data))

    def view(self, *shape):
        """Alias for reshape, matching PyTorch's Tensor.view() for contiguous tensors."""
        return self.reshape(*shape)

    def masked_fill(self, mask, value):
        """Fill positions where mask is True with value, matching PyTorch's masked_fill.

        Used in transformer attention to set padding positions to -inf before softmax.

        Args:
            mask: A Tensor or numpy array of booleans, same shape as self (or broadcastable)
            value: Scalar fill value (e.g. float('-inf') for attention masking)

        Returns:
            New Tensor with masked positions replaced by value.
        """
        mask_array = mask.data.astype(bool) if isinstance(mask, Tensor) else np.asarray(mask, dtype=bool)
        result = self.data.copy()
        result[mask_array] = value
        return Tensor(result)

    def __add__(self, other):
        """Add two tensors element-wise with broadcasting support."""
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        """Support natural scalar arithmetic: scaler + tensor."""
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        return Tensor(other - self.data)

    def __mul__(self, other):
        """Mumtiply two tensors element-wise (NOT matrix multiplication)"""
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        return Tensor(self.data * other)

    def __rmul__(self, other):
        """Support natural scalar arithmetic: scalar * tensor."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Divide two tensors element-wise."""
        if isinstance(other, Tensor):
            return Tensor(self.data / other.data)
        return Tensor(self.data / other)

    def __rtruediv__(self, other):
        """Support natural scalar arithmetic: scalar / tensor."""
        if isinstance(other, Tensor):
            return Tensor(other.data / self.data)
        return Tensor(other / self.data)

    def _validate_matmul_shapes(self, other):
        """Validate that two tensors are compatible for matrix multiplication."""

        if not isinstance(other, Tensor):
            raise TypeError(
                f"Matrix multiplication requires Tensor, got {type(other).__name__}\n"
                f"Wrap your data: Tensor({other}) @ other_tensor"
            )

        if len(self.shape) == 0 or len(other.shape) == 0:
            raise ValueError(
                f"Matrix multiplication requires at least 1D tensors\n"
                f"Got shape: {self.shape} @ {other.shape}\n"
            )

        if len(self.shape) >= 2 and len(other.shape) >= 2:
            if self.shape[-1] != other.shape[-2]:
                raise ValueError(
                    f"Inner dimensions don't match: {self.shape[-1]} vs {other.shape[-2]} "
                    f"for {self.shape} @ {other.shape}\n"
                )

    def matmul(self, other):
        """Matrix multiplication of two tensors.

        Validates shapes via _validate_matmul_shapes, then computes the product.
        2D matrix: explicit nested loops so I can see how exactly each output element is a dot
        product of a row and a column.
        3D+ matrix: delegated to np.matmul
        """

        self._validate_matmul_shapes(other)

        a = self.data
        b = other.data

        # 2D metrices with explicit loops
        if len(a.shape) == 2 and len(b.shape) == 2:
            M, K = a.shape
            K2, N = b.shape
            result_data = np.zeros((M, N), dtype=a.dtype)

            for i in range(M):
                for j in range(N):
                    result_data[i, j] = np.dot(a[i, :], b[:, j])
        else:
            result_data = np.matmul(a, b)

        return Tensor(result_data)

    def __matmul__(self, other):
        """Enable @ operator for matrix multiplication."""
        return self.matmul(other)

    def __getitem__(self, key):
        """Enable indexing and slicing operations on Tensors."""
        result_data = self.data[key]
        if not isinstance(result_data, np.ndarray):
            result_data = np.array(result_data)
        return Tensor(result_data)

    def reshape(self, *shape):
        """Reshape tensor to new dimenions."""

        # Handling the tuple/list calling style
        # Convert it to a convention: A.reshape((3, 4)) -> A.reshape(3, 4)
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)

        if -1 in shape:
            if shape.count(-1) > 1:
                raise ValueError(
                    f"Cannot reshape {self.shape} with multiple unknown dimensions."
                )
            idx = shape.index(-1)

            known_size = 1
            for i, dim in enumerate(shape):
                if i != idx:
                    known_size *= dim

            if self.size % known_size != 0:
                raise ValueError(
                    f"Cannot infer -1 dimension: {self.size} elements is not"
                    f"divisible by the known dimensions product {known_size}"
                )

            shape[idx] = self.size // known_size
        
        
        # Check compatibility of new size against tensor size
        # Reshape must preserve total element count
        if self.size != np.prod(shape):
            raise ValueError(
                f"Element count mismatch: {self.size} elements vs {np.prod(shape)} elements. "
                f"{self.shape} tensor can't be reshaped to {shape}."
            )
        return Tensor(self.data.reshape(shape))


    def transpose(self, dim0=None, dim1=None):
        """Transpose tensor dimensions."""

        # By default -2 and -1 axes get swaped 
        if dim0 is None and dim1 is None:
            if len(self.shape) < 2:
                return Tensor(self.data.copy())
            else:
                axes = list(range(len(self.shape)))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                transposed_data = np.transpose(self.data, axes)
        else:
            if dim0 is None or dim1 is None:
                provided = f"dim0={dim0}" if dim1 is None else f"dim1={dim1}"
                missing = "dim1" if dim1 is None else "dim0"
                raise ValueError(
                    f"Transpose requires both dimensions to be specified\n"
                    f"Got {provided}, but {missing} is None"
                )
            # Two dims are provided -> swap dim0 and dim1
            axes = list(range(len(self.shape)))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
            transposed_data = np.transpose(self.data, axes)
        return Tensor(transposed_data)
        

    def sum(self, axis=None, keepdims=False):
        """Sum tensor along specified axis."""
        result = np.sum(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def mean(self, axis=None, keepdims=False):
        """Compute mean of tensor along specified axis."""
        result = np.mean(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)

    def max(self, axis=None, keepdims=False):
        """Find maximum values along specified axis"""
        result = np.max(self.data, axis=axis, keepdims=keepdims)
        return Tensor(result)


"""
## Key Accomplishments
- Built a complete Tensor class with arithmetic operations, matrix multiplication, and shape manipulation
- Implemented broadcasting semantics that match NumPy for automatic shape alignment
- Created reduction operations (sum, mean, max) for aggregating data across dimensions
- Discovered cache performance implications through memory layout analysis

## Systems Insights Discovered
- Memory layout matters: Row-wise access is faster than column-wise due to cache locality
- Broadcasting efficiency: NumPy handles shape alignment without explicit data copying
- Matrix multiplication is the computational foundation of linear transformations
- Shape validation provides clear error messages at minimal performance cost
"""
