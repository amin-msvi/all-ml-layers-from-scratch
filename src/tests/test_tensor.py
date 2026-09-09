from tensor import Tensor, BYTES_PER_FLOAT32, KB_TO_BYTES, MB_TO_BYTES
import numpy as np


rng = np.random.default_rng(7)

def test_unit_tensor_creation():
    """Test Tensor creation with various data types."""

    # Test scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1
    assert scalar.dtype == np.float32
    
    # Test vector creation
    vector = Tensor([1, 2, 3])
    assert np.array_equal(vector.data, np.array([1, 2, 3], dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3

    # Test matrix creation
    matrix = Tensor([[1, 2], [3, 4]])
    assert np.array_equal(matrix.data, np.array([[1, 2], [3, 4]], dtype=np.float32))
    assert matrix.shape == (2, 2)
    assert matrix.size == 4

    # Test 3D tensor creation
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2, 2, 2)
    assert tensor_3d.size == 8

    # Test PyTorch-compatible utility methods
    assert scalar.ndim == 0, "Scalar should be 0-dimensional"
    assert vector.ndim == 1, "Vector should be 1-dimensional"
    assert matrix.ndim == 2, "Matrix should be 2-dimensional"
    assert tensor_3d.ndim == 3, "3D tensor should be 3-dimensional"

    assert scalar.numel() == 1, "Scalar has 1 element"
    assert vector.numel() == 3, "Vector has 3 elements"
    assert matrix.numel() == 4, "2x2 matrix has 4 elements"

    # Test contiguous returns a copy with same data
    contig = matrix.contiguous()
    assert np.array_equal(contig.data, matrix.data)
    assert contig.data is not matrix.data, "contiguous() should return a copy"

    print("Tensor creation works correctly!")


def test_unit_arithmetic_operations():
    """🧪 Test arithmetic operations with broadcasting."""

    # Test tensor + tensor
    a = Tensor([1, 2, 3])
    b = Tensor([4, 5, 6])
    result = a + b
    assert np.array_equal(result.data, np.array([5, 7, 9], dtype=np.float32))

    # Test tensor + scalar
    result = a + 10
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))

    # Scalar on the left should behave naturally too
    result = 10 + a
    assert np.array_equal(result.data, np.array([11, 12, 13], dtype=np.float32))

    # Test broadcasting with different shapes (matrix + vector)
    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])
    result = matrix + vector
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Broadcasting pitfall: verify shapes match before element-wise ops
    # In ML, predictions (batch, features) minus targets (features,) broadcasts
    # silently -> the same target row repeats for every sample. Always check!
    predictions = Tensor(np.ones((4, 3)))   # 4 samples, 3 features
    targets_good = Tensor(np.zeros((4, 3)))  # correct: same shape
    targets_bad = Tensor(np.zeros((3,)))     # dangerous: missing batch dim
    assert predictions.shape == targets_good.shape, "Matching shapes — safe"
    assert predictions.shape != targets_bad.shape, (
        f"Shape mismatch: {predictions.shape} vs {targets_bad.shape}. "
        f"NumPy broadcasts silently — this is almost always a bug in ML code."
    )

    # Test substraction (data centering)
    result = b - a
    assert np.array_equal(result.data, np.array([3, 3, 3], dtype=np.float32))

    # Test multiplication (scaling)
    result = a * 2
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    result = 2 * a
    assert np.array_equal(result.data, np.array([2, 4, 6], dtype=np.float32))

    # Test division (normalization)
    result = b / 2
    assert np.array_equal(result.data, np.array([2.0, 2.5, 3.0], dtype=np.float32))

    result = 12 / b
    assert np.allclose(result.data, np.array([3.0, 12.0 / 5.0, 2.0], dtype=np.float32))

    # Order matters for subtraction, but scalar-left arithmetic should still work
    result = 10 - a
    assert np.array_equal(result.data, np.array([9, 8, 7], dtype=np.float32))

    # Test chaining operations (common in ML pipelines)
    normalized = (a - 2) / 2  # Center and scale
    expected = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    assert np.allclose(normalized.data, expected)

    print("Arithmetic operations work correctly!")


def test_unit_validate_matmul_shapes():

    # Valid shapes should pass without error
    a = Tensor([[1, 2], [3, 4]])  # 2x2
    b = Tensor([[5, 6], [7, 8]])  # 2x2
    a._validate_matmul_shapes(b)  # No exception

    # Valid rectangular shapes
    c = Tensor([[1, 2, 3]])       # 1x3
    d = Tensor([[1], [2], [3]])   # 3x1
    c._validate_matmul_shapes(d)  # No exception (inner dim 3 matches)

    # Check 1: TypeError when other is not a Tensor
    try:
        a._validate_matmul_shapes([[1, 2], [3, 4]])
        assert False, "Should have raised TypeError for non-Tensor"
    except TypeError as e:
        assert "requires Tensor" in str(e)
        assert "list" in str(e)

    # Check 2: ValueError when either operand is a 0D scalar
    try:
        scalar = Tensor(5.0)
        scalar._validate_matmul_shapes(a)
        assert False, "Should have raised ValueError for 0D tensor"
    except ValueError as e:
        assert "at least 1D" in str(e)

    # Check 3: ValueError when inner dimensions don't match
    try:
        incompatible_a = Tensor([[1, 2]])          # 1x2
        incompatible_b = Tensor([[1], [2], [3]])   # 3x1
        incompatible_a._validate_matmul_shapes(incompatible_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "Inner dimensions don't match" in str(e)
        assert "2 vs 3" in str(e)

    print("Matmul shape validation works correctly!")

def test_unit_matrix_multiplication():

    # Test 2x2 matrix multiplication (basic case)
    a = Tensor([[1, 2], [3, 4]])  # 2x2
    b = Tensor([[5, 6], [7, 8]])  # 2x2
    result = a.matmul(b)
    # Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test rectangular matrices (common in data transformations)
    c = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3 (like samples=2, features=3)
    d = Tensor([[7, 8], [9, 10], [11, 12]])  # 3x2 (like features=3, outputs=2)
    result = c.matmul(d)
    # Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    expected = np.array([[58, 64], [139, 154]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test matrix-vector multiplication (common in linear transforms)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
    vector = Tensor([1, 2, 3])  # 1D vector
    result = matrix.matmul(vector)
    # Expected: [1*1+2*2+3*3, 4*1+5*2+6*3] = [14, 32]
    expected = np.array([14, 32], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Test @ operator sugar
    result_at = a @ b
    assert np.array_equal(result_at.data, np.array([[19, 22], [43, 50]], dtype=np.float32))

    print("Matrix multiplication works correctly!")


def test_unit_shape_manipulation():

    # Test basic reshape (flatten → matrix)
    tensor = Tensor([1, 2, 3, 4, 5, 6])  # Shape: (6,)
    reshaped = tensor.reshape(2, 3)      # Shape: (2, 3)
    assert reshaped.shape == (2, 3)
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped.data, expected)

    # Test reshape with tuple (alternative calling style)
    reshaped2 = tensor.reshape((3, 2))   # Shape: (3, 2)
    assert reshaped2.shape == (3, 2)
    expected2 = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    assert np.array_equal(reshaped2.data, expected2)

    # Test reshape with -1 (automatic dimension inference)
    auto_reshaped = tensor.reshape(2, -1)  # Should infer -1 as 3
    assert auto_reshaped.shape == (2, 3)

    # Test reshape validation - should raise error for incompatible sizes
    try:
        tensor.reshape(2, 2)  # 6 elements can't fit in 2×2=4
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Element count mismatch" in str(e)
        assert "6 elements vs 4 elements" in str(e)

    # Test matrix transpose (most common case)
    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    transposed = matrix.transpose()          # (3, 2)
    assert transposed.shape == (3, 2)
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype=np.float32)
    assert np.array_equal(transposed.data, expected)

    # Test 1D transpose (should be identity)
    vector = Tensor([1, 2, 3])
    vector_t = vector.transpose()
    assert np.array_equal(vector.data, vector_t.data)

    # Non-symmetric shape so a wrong axis swap produces the wrong shape and data.
    tensor_3d = Tensor(np.arange(24).reshape(2, 3, 4))  # (2, 3, 4)
    swapped = tensor_3d.transpose(0, 2)  # (4, 3, 2)
    assert swapped.shape == (4, 3, 2), (
        f"transpose(0, 2) on (2,3,4) should give (4,3,2), got {swapped.shape}"
    )
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert swapped.data[k, j, i] == tensor_3d.data[i, j, k], (
                    f"Data mismatch at [{k},{j},{i}]: expected {tensor_3d.data[i,j,k]}, "
                    f"got {swapped.data[k,j,i]}"
                )

    # Test common reshape pattern (flatten multi-dimensional data)
    batch_images = Tensor(rng.random((2, 3, 4)))  # (batch=2, height=3, width=4)
    flattened = batch_images.reshape(2, -1)  # (batch=2, features=12)
    assert flattened.shape == (2, 12)

    print("Shape manipulation works correctly!")


def test_unit_reduction_operations():

    matrix = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)

    # Test sum all elements (common for loss computation)
    total = matrix.sum()
    assert total.data == 21.0  # 1+2+3+4+5+6
    assert total.shape == ()   # Scalar result

    # Test sum along axis 0 (columns) - batch dimension reduction
    col_sum = matrix.sum(axis=0)
    expected_col = np.array([5, 7, 9], dtype=np.float32)  # [1+4, 2+5, 3+6]
    assert np.array_equal(col_sum.data, expected_col)
    assert col_sum.shape == (3,)

    # Test sum along axis 1 (rows) - feature dimension reduction
    row_sum = matrix.sum(axis=1)
    expected_row = np.array([6, 15], dtype=np.float32)  # [1+2+3, 4+5+6]
    assert np.array_equal(row_sum.data, expected_row)
    assert row_sum.shape == (2,)

    # Test mean (average loss computation)
    avg = matrix.mean()
    assert np.isclose(avg.data, 3.5)  # 21/6
    assert avg.shape == ()

    # Test mean along axis (per-column statistics)
    col_mean = matrix.mean(axis=0)
    expected_mean = np.array([2.5, 3.5, 4.5], dtype=np.float32)  # [5/2, 7/2, 9/2]
    assert np.allclose(col_mean.data, expected_mean)

    # Test max (finding largest value)
    maximum = matrix.max()
    assert maximum.data == 6.0
    assert maximum.shape == ()

    # Test max along axis (argmax-like operation)
    row_max = matrix.max(axis=1)
    expected_max = np.array([3, 6], dtype=np.float32)  # [max(1,2,3), max(4,5,6)]
    assert np.array_equal(row_max.data, expected_max)

    # Test keepdims (important for broadcasting)
    sum_keepdims = matrix.sum(axis=1, keepdims=True)
    assert sum_keepdims.shape == (2, 1)  # Maintains 2D shape
    expected_keepdims = np.array([[6], [15]], dtype=np.float32)
    assert np.array_equal(sum_keepdims.data, expected_keepdims)

    # Test 3D reduction (averaging across spatial dimensions)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    spatial_mean = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert spatial_mean.shape == (2,)  # One value per batch item

    print("Reduction operations work correctly!")


def test_analyze_memory_layout():
    """Demonstrate cache effects with row vs column access patterns."""
    print("Analyzing Memory Access Patterns...")
    print("=" * 60)

    # Create a moderately-sized matrix (large enough to show cache effects)
    size = 2000
    matrix = Tensor(rng.random((size, size)))

    import time

    print(f"\nTesting with {size}×{size} matrix ({matrix.size * BYTES_PER_FLOAT32 / MB_TO_BYTES:.1f} MB)")
    print("-" * 60)

    # Test 1: Row-wise access (cache-friendly)
    # Memory layout: [row0][row1][row2]... stored contiguously
    print("\nTest 1: Row-wise Access (Cache-Friendly)")
    start = time.time()
    row_sums = []
    for i in range(size):
        row_sum = matrix.data[i, :].sum()  # Access entire row sequentially
        row_sums.append(row_sum)
    row_time = time.time() - start
    print(f"   Time: {row_time*1000:.1f}ms")
    print("   Access pattern: Sequential (follows memory layout)")

    # Test 2: Column-wise access (cache-unfriendly)
    # Must jump between rows, poor spatial locality
    print("\nTest 2: Column-wise Access (Cache-Unfriendly)")
    start = time.time()
    col_sums = []
    for j in range(size):
        col_sum = matrix.data[:, j].sum()  # Access entire column with large strides
        col_sums.append(col_sum)
    col_time = time.time() - start
    print(f"   Time: {col_time*1000:.1f}ms")
    print(f"   Access pattern: Strided (jumps {size * BYTES_PER_FLOAT32} bytes per element)")

    # Calculate slowdown
    slowdown = col_time / row_time
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE IMPACT:")
    print(f"   Slowdown factor: {slowdown:.2f}× ({col_time/row_time:.1f}× slower)")
    print(f"   Cache misses cause {(slowdown-1)*100:.0f}% performance loss")

    # Educational insights
    print("\n💡 KEY INSIGHTS:")
    print("   1. Memory layout matters: Row-major (C-style) storage is sequential")
    print("   2. Cache lines are ~64 bytes: Row access loads nearby elements \"for free\"")
    print("   3. Column access misses cache: Must reload from DRAM every time")
    print(f"   4. This is O(n) algorithm but {slowdown:.1f}× different wall-clock time!")

    print("\n🚀 REAL-WORLD IMPLICATIONS:")
    print("   • Image processing libraries use specific memory formats for cache efficiency")
    print("   • Matrix multiplication optimized with blocking (tile into cache-sized chunks)")
    print(f"   • Transpose is expensive ({slowdown:.1f}×) because it creates a non-contiguous view with poor cache locality")
    print("   • Hardware-optimized libraries leverage memory layout for better performance")

    print("\n" + "=" * 60)


def test_module():
    """🧪 Module Test: Complete Integration

    Comprehensive test of entire module functionality.

    This final test runs before module summary to ensure:
    - All unit tests pass
    - Functions work together correctly
    - Module is ready for integration with TinyTorch
    """
    print("RUNNING MODULE INTEGRATION TEST")
    print("=" * 50)

    # Run all unit tests
    print("Running unit tests...")
    test_unit_tensor_creation()
    test_unit_arithmetic_operations()
    test_unit_validate_matmul_shapes()
    test_unit_matrix_multiplication()
    test_unit_shape_manipulation()
    test_unit_reduction_operations()

    print("\nRunning integration scenarios...")

    # Test realistic multi-stage computation
    print("Integration Test: Two-Stage Linear Transformation...")

    # Create input data (2 samples, 3 features)
    x = Tensor([[1, 2, 3], [4, 5, 6]])

    # First stage: 3 inputs → 4 intermediate values
    W1 = Tensor([[0.1, 0.2, 0.3, 0.4],
                 [0.5, 0.6, 0.7, 0.8],
                 [0.9, 1.0, 1.1, 1.2]])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])

    # Forward pass: hidden = xW1 + b1
    hidden = x.matmul(W1) + b1
    assert hidden.shape == (2, 4), f"Expected (2, 4), got {hidden.shape}"

    # Second stage: 4 intermediate → 2 outputs
    W2 = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    b2 = Tensor([0.1, 0.2])

    # Output stage: output = hiddenW2 + b2
    output = hidden.matmul(W2) + b2
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Verify data flows correctly (no NaN, reasonable values)
    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Output contains infinite values"

    print("Two-stage linear transformation works!")

    # Test complex shape manipulations
    print("Integration Test: Complex Shape Operations...")
    data = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    # Reshape to 3D tensor (simulating batch processing)
    tensor_3d = data.reshape(2, 2, 3)  # (batch=2, height=2, width=3)
    assert tensor_3d.shape == (2, 2, 3)

    # Spatial averaging (collapse height and width)
    pooled = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

    # Flatten to 2D
    flattened = tensor_3d.reshape(2, -1)  # (batch, features)
    assert flattened.shape == (2, 6)

    # Transpose for different operations
    transposed = tensor_3d.transpose()  # Should transpose last two dims
    assert transposed.shape == (2, 3, 2)

    print("Complex shape operations work!")

    # Test broadcasting edge cases
    print("Integration Test: Broadcasting Edge Cases...")

    # Scalar broadcasting
    scalar = Tensor(5.0)
    vector = Tensor([1, 2, 3])
    result = scalar + vector  # Should broadcast scalar to vector shape
    expected = np.array([6, 7, 8], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    # Matrix + vector broadcasting
    matrix = Tensor([[1, 2], [3, 4]])
    vec = Tensor([10, 20])
    result = matrix + vec
    expected = np.array([[11, 22], [13, 24]], dtype=np.float32)
    assert np.array_equal(result.data, expected)

    print("Broadcasting edge cases work!")