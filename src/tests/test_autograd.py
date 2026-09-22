def test_unit_reduce_broadcast_grad():
    """Test _reduce_broadcast_grad helper function."""
    print("Unit Test: _reduce_broadcast_grad...")
    
    # Test 1: Remove leading dimension
    grad = np.ones((32, 128))
    original_shape = (128,)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (128,), f"Expected (128,), got {reduced.shape}"
    assert np.allclose(reduced, np.ones(128) * 32), "Should sum over batch dimension"
    print("  ✓ Leading dimension reduction works")
    
    # Test 2: Collapse singleton dimension
    grad = np.ones((10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
    assert np.allclose(reduced, np.ones((10, 1)) * 5), "Should sum over singleton axis"
    print("  ✓ Singleton dimension reduction works")
    
    # Test 3: Both operations
    grad = np.ones((32, 10, 5))
    original_shape = (10, 1)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 1), f"Expected (10, 1), got {reduced.shape}"
    expected = np.ones((10, 1)) * 32 * 5  # Sum over batch and last dim
    assert np.allclose(reduced, expected), "Should handle multiple reductions"
    print("  ✓ Multiple dimension reduction works")
    
    # Test 4: No broadcasting (should return unchanged)
    grad = np.ones((10, 5))
    original_shape = (10, 5)
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (10, 5), f"Expected (10, 5), got {reduced.shape}"
    assert np.allclose(reduced, grad), "Should return unchanged when shapes match"
    print("  ✓ No-op case works")
    
    # Test 5: Scalar case (reduce to scalar)
    grad = np.ones((32, 128))
    original_shape = ()
    reduced = _reduce_broadcast_grad(grad, original_shape)
    assert reduced.shape == (), f"Expected scalar, got {reduced.shape}"
    assert np.allclose(reduced, 32 * 128), "Should sum to scalar"
    print("  ✓ Scalar reduction works")
    
    print("_reduce_broadcast_grad works correctly!")


def test_unit_function_classes():
    """Test Function classes."""
    print("Unit Test: Function Classes...")

    # Note: We manually set requires_grad here because enable_autograd() hasn't been
    # called yet. This tests the Backward classes in isolation before the full
    # autograd system is enabled.

    # Test AddBackward
    a = Tensor([1, 2, 3])
    a.requires_grad = True  # Set manually (enable_autograd not called yet)
    b = Tensor([4, 5, 6])
    b.requires_grad = True
    add_func = AddBackward(a, b)
    grad_output = np.array([1, 1, 1])
    grad_a, grad_b = add_func.apply(grad_output)
    assert np.allclose(grad_a, grad_output), f"AddBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, grad_output), f"AddBackward grad_b failed: {grad_b}"

    # Test MulBackward
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(grad_output)
    assert np.allclose(grad_a, b.data), f"MulBackward grad_a failed: {grad_a}"
    assert np.allclose(grad_b, a.data), f"MulBackward grad_b failed: {grad_b}"

    # Test MatmulBackward
    a_mat = Tensor([[1, 2], [3, 4]])
    a_mat.requires_grad = True
    b_mat = Tensor([[5, 6], [7, 8]])
    b_mat.requires_grad = True
    matmul_func = MatmulBackward(a_mat, b_mat)
    grad_output = np.ones((2, 2))
    grad_a, grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape, f"MatmulBackward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape, f"MatmulBackward grad_b shape: {grad_b.shape}"

    print("Function classes work correctly!")


def test_unit_broadcast_gradients():
    """Test gradient broadcasting reduction."""
    print("Unit Test: Broadcasting in Gradients...")

    # Note: We set requires_grad manually here because enable_autograd() hasn't
    # been called yet. This tests the Backward classes in isolation before the
    # full autograd system upgrades the Tensor constructor to accept requires_grad.

    # Scenario 1: Bias-like broadcasting (most common case)
    # Shape: (batch, features) + (features,) → (batch, features)
    x = Tensor(rng.standard_normal((4, 3)))
    x.requires_grad = True
    bias = Tensor(np.ones(3))
    bias.requires_grad = True
    
    add_func = AddBackward(x, bias)
    grad_output = np.ones((4, 3))
    grad_x, grad_bias = add_func.apply(grad_output)
    
    # Check shapes
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    assert grad_bias.shape == bias.data.shape, \
        f"Expected grad_bias shape {bias.data.shape}, got {grad_bias.shape}"
    
    # Check values: bias gradient should sum over batch dimension
    expected_bias_grad = np.ones(3) * 4  # Sum of 4 ones = 4
    assert np.allclose(grad_bias, expected_bias_grad), \
        f"Expected bias grad {expected_bias_grad}, got {grad_bias}"
    
    print("  ✓ Bias-like broadcasting works")
    
    # Scenario 2: Scalar broadcasting
    # Shape: (3, 4) + scalar → (3, 4)
    x = Tensor(rng.standard_normal((3, 4)))
    x.requires_grad = True
    scalar_val = 5.0
    
    add_func = AddBackward(x, scalar_val)
    grad_output = np.ones((3, 4))
    grad_x, grad_scalar = add_func.apply(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    # Scalar gradients don't get reduced (not a Tensor)
    
    print("  ✓ Scalar broadcasting works")
    
    # Scenario 3: Multiple dimension broadcasting
    # Shape: (32, 10, 5) + (10, 1) → (32, 10, 5)
    x = Tensor(rng.standard_normal((32, 10, 5)))
    x.requires_grad = True
    y = Tensor(rng.standard_normal((10, 1)))
    y.requires_grad = True
    
    mul_func = MulBackward(x, y)
    grad_output = np.ones((32, 10, 5))
    grad_x, grad_y = mul_func.apply(grad_output)
    
    assert grad_x.shape == x.data.shape, \
        f"Expected grad_x shape {x.data.shape}, got {grad_x.shape}"
    assert grad_y.shape == y.data.shape, \
        f"Expected grad_y shape {y.data.shape}, got {grad_y.shape}"
    
    print("  ✓ Multi-dimension broadcasting works")
    
    # Scenario 4: Test all operations (Add, Mul, Sub, Div)
    a = Tensor(rng.standard_normal((8, 16)))
    a.requires_grad = True
    b = Tensor(rng.standard_normal(16))
    b.requires_grad = True
    
    # Test Addition
    add_func = AddBackward(a, b)
    grad_a, grad_b = add_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"AddBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Multiplication
    mul_func = MulBackward(a, b)
    grad_a, grad_b = mul_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"MulBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Subtraction
    sub_func = SubBackward(a, b)
    grad_a, grad_b = sub_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"SubBackward: Expected (16,), got {grad_b.shape}"
    
    # Test Division
    div_func = DivBackward(a, b)
    grad_a, grad_b = div_func.apply(np.ones((8, 16)))
    assert grad_b.shape == (16,), f"DivBackward: Expected (16,), got {grad_b.shape}"
    
    print("  ✓ All operations handle broadcasting correctly")
    
    # Scenario 5: Real-world case - Linear layer gradient
    # Simulates: output = input @ weight + bias
    # where bias is (out_features,) and output is (batch, out_features)
    batch_size, out_features = 32, 128
    output_grad = rng.standard_normal((batch_size, out_features))
    bias = Tensor(np.zeros(out_features))
    bias.requires_grad = True

    # In real Linear layer, bias gradient comes from output gradient
    activations = Tensor(np.zeros((batch_size, out_features)))
    activations.requires_grad = True
    add_func = AddBackward(activations, bias)
    _, grad_bias = add_func.apply(output_grad)
    
    assert grad_bias.shape == (out_features,), \
        f"Linear layer bias: Expected ({out_features},), got {grad_bias.shape}"
    
    # Verify gradient is sum over batch
    expected = output_grad.sum(axis=0)
    assert np.allclose(grad_bias, expected), \
        "Bias gradient should equal sum over batch dimension"
    
    print("  ✓ Real-world Linear layer scenario works")
    
    print("Broadcasting gradient tests pass!")


def test_unit_stable_softmax():
    """Test stable softmax helper."""
    print("Testing stable softmax helper...")

    # Basic correctness
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = _stable_softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0), f"Softmax should sum to 1, got {probs.sum(axis=1)}"

    # Verify correct values
    expected = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    assert np.allclose(probs, expected), f"Softmax values wrong: {probs}"

    # Numerical stability: large values that would overflow naive exp()
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs_large = _stable_softmax(large_logits)
    assert not np.any(np.isnan(probs_large)), "Stable softmax should handle large values"
    assert not np.any(np.isinf(probs_large)), "Stable softmax should not overflow"
    assert np.allclose(probs_large.sum(axis=1), 1.0), "Large softmax should sum to 1"

    # Batch dimension
    batch_logits = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    batch_probs = _stable_softmax(batch_logits)
    assert batch_probs.shape == (3, 2), f"Expected (3, 2), got {batch_probs.shape}"
    assert np.allclose(batch_probs.sum(axis=1), np.ones(3)), "Each row should sum to 1"

    print("  Stable softmax helper works correctly!")



def test_unit_one_hot_encode():
    """Test one-hot encoding helper."""
    print("Testing one-hot encoding helper...")

    # Basic test
    targets = np.array([0, 2, 1])
    result = _one_hot_encode(targets, batch_size=3, num_classes=3)
    expected = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert np.allclose(result, expected), f"One-hot encoding wrong: {result}"

    # Single sample
    result_single = _one_hot_encode(np.array([1]), batch_size=1, num_classes=4)
    assert result_single.shape == (1, 4), f"Expected (1, 4), got {result_single.shape}"
    assert result_single[0, 1] == 1.0, "Target class should be 1.0"
    assert result_single.sum() == 1.0, "Should have exactly one 1.0"

    # Each row sums to 1
    targets_batch = np.array([0, 1, 2, 3, 4])
    result_batch = _one_hot_encode(targets_batch, batch_size=5, num_classes=5)
    assert np.allclose(result_batch.sum(axis=1), np.ones(5)), "Each row should sum to 1"

    print("  One-hot encoding helper works correctly!")


def test_unit_tensor_autograd():
    """🔬 Test Tensor autograd enhancement."""
    print("🔬 Unit Test: Tensor Autograd Enhancement...")

    # Test simple gradient computation
    x = Tensor([2.0], requires_grad=True)
    y = x * 3
    z = y + 1  # z = 3x + 1, so dz/dx = 3

    z.backward()
    assert np.allclose(x.grad, [3.0]), f"Expected [3.0], got {x.grad}"

    # Test matrix multiplication gradients
    a = Tensor([[1.0, 2.0]], requires_grad=True)  # 1x2
    b = Tensor([[3.0], [4.0]], requires_grad=True)  # 2x1
    c = a.matmul(b)  # 1x1, result = [[11.0]]

    c.backward()
    assert np.allclose(a.grad, [[3.0, 4.0]]), f"Expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(b.grad, [[1.0], [2.0]]), f"Expected [[1.0], [2.0]], got {b.grad}"

    # Test computation graph with multiple operations
    x = Tensor([1.0, 2.0], requires_grad=True)
    y = x * 2      # y = [2, 4]
    z = y.sum()    # z = 6

    z.backward()
    assert np.allclose(x.grad, [2.0, 2.0]), f"Expected [2.0, 2.0], got {x.grad}"

    print("✅ Tensor autograd enhancement works correctly!")


