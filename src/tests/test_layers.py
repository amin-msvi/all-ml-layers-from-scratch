import numpy as np

from core.tensor import Tensor
from core.layers import Linear, Dropout, Sequential

INIT_SCALE_FACTOR = 1.0  # LeCun-style initialization: sqrt(1/fan_in)

rng = np.random.default_rng(7)

# 1. Linear layer tests
def test_unit_linear_layer():
    """Test Linear layer implementation."""
    print("Unit Test: Linear Layer...")

    # Test layer creation
    layer = Linear(784, 256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784, 256)
    assert layer.bias.shape == (256,)

    # Test LeCun-style initialization (weights should be reasonably scaled)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(INIT_SCALE_FACTOR / 784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f"Weight std {weight_std} not close to expected {expected_std}"

    # Test bias initialization (should be zeros)
    assert np.allclose(layer.bias.data, 0), "Bias should be initialized to zeros"

    # Test forward pass
    x = Tensor(rng.standard_normal((32, 784)))  # Batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32, 256), f"Expected shape (32, 256), got {y.shape}"

    # Test no bias option
    layer_no_bias = Linear(10, 5, bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1  # Only weight, no bias

    # Test parameters method
    params = layer.parameters()
    assert len(params) == 2  # Weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("Linear layer works correctly!")


def test_unit_edge_cases_linear():
    """Test Linear layer edge cases."""
    print("Edge Case Tests: Linear Layer...")

    layer = Linear(10, 5)

    # Test single sample (should handle 2D input)
    x_2d = Tensor(rng.standard_normal((1, 10)))
    y = layer.forward(x_2d)
    assert y.shape == (1, 5), "Should handle single sample"

    # Test zero batch size (edge case)
    x_empty = Tensor(rng.standard_normal((0, 10)))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0, 5), "Should handle empty batch"

    # Test numerical stability with large weights
    layer_large = Linear(10, 5)
    layer_large.weight.data = np.ones((10, 5)) * 100  # Large but not extreme
    x = Tensor(np.ones((1, 10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data)), "Should not produce NaN with large weights"
    assert not np.any(np.isinf(y.data)), "Should not produce Inf with large weights"

    # Test with no bias
    layer_no_bias = Linear(10, 5, bias=False)
    x = Tensor(rng.standard_normal((4, 10)))
    y = layer_no_bias.forward(x)
    assert y.shape == (4, 5), "Should work without bias"

    print("Edge cases handled correctly!")


def test_unit_parameter_collection_linear():
    """Test Linear layer parameter collection."""
    print("Parameter Collection Test: Linear Layer...")

    layer = Linear(10, 5)

    # Verify parameter collection works
    params = layer.parameters()
    assert len(params) == 2, "Should return 2 parameters (weight and bias)"
    assert params[0].shape == (10, 5), "First param should be weight"
    assert params[1].shape == (5,), "Second param should be bias"

    # Test layer without bias
    layer_no_bias = Linear(10, 5, bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, "Should return 1 parameter (weight only)"

    print("Parameter collection works correctly!")


# 2. Dropout layer tests

def test_unit_should_apply_dropout():
    """Test _should_apply_dropout decision logic."""
    print("Unit Test: Dropout Decision Logic...")

    # Standard dropout (p=0.5) in training mode should apply
    d = Dropout(0.5)
    assert d._should_apply_dropout(training=True) is True, \
        "Dropout(0.5) should apply during training"

    # Same dropout in inference mode should NOT apply
    assert d._should_apply_dropout(training=False) is False, \
        "Dropout should not apply during inference"

    # Zero dropout (p=0) should never apply, even in training
    d_zero = Dropout(0.0)
    assert d_zero._should_apply_dropout(training=True) is False, \
        "Dropout(0.0) should never apply (no neurons to drop)"

    # Full dropout (p=1.0) in training mode should apply
    d_full = Dropout(1.0)
    assert d_full._should_apply_dropout(training=True) is True, \
        "Dropout(1.0) should apply during training"

    # Full dropout in inference mode should NOT apply
    assert d_full._should_apply_dropout(training=False) is False, \
        "Even Dropout(1.0) should not apply during inference"

    print("Dropout decision logic works correctly!")


def test_unit_generate_dropout_mask():
    """Test _generate_dropout_mask output properties."""
    print("Unit Test: Dropout Mask Generation...")

    d = Dropout(0.5)
    rng = np.random.default_rng(7)
    mask = d._generate_dropout_mask((1000,))

    # Shape must match the requested shape
    assert mask.shape == (1000,), f"Expected shape (1000,), got {mask.shape}"

    # Every element must be either 0.0 or 2.0 (= 1/(1-0.5))
    unique_vals = set(np.unique(mask.data))
    assert unique_vals <= {0.0, 2.0}, \
        f"Mask values should be {{0.0, 2.0}}, got {unique_vals}"

    # Statistically, about 50% should survive (3-sigma tolerance)
    non_zero = np.count_nonzero(mask.data)
    std_err = np.sqrt(1000 * 0.5 * 0.5)
    assert 500 - 3 * std_err < non_zero < 500 + 3 * std_err, \
        f"Expected ~500 survivors, got {non_zero}"

    # Test with different dropout probability
    d2 = Dropout(0.3)
    rng = np.random.default_rng(7)
    mask2 = d2._generate_dropout_mask((2000,))

    # Values should be 0.0 or 1/(1-0.3) ≈ 1.4286
    expected_scale = 1.0 / 0.7
    non_zero_vals = mask2.data[mask2.data != 0.0]
    assert np.allclose(non_zero_vals, expected_scale), \
        f"Surviving values should be {expected_scale:.4f}, got {np.unique(non_zero_vals)}"

    # About 70% should survive for p=0.3
    survival_rate = np.count_nonzero(mask2.data) / 2000
    assert 0.60 < survival_rate < 0.80, \
        f"Expected ~70% survival for p=0.3, got {survival_rate:.1%}"

    print("Dropout mask generation works correctly!")


def test_unit_dropout_layer():
    """Test Dropout layer implementation."""
    print("Unit Test: Dropout Layer...")

    # Test dropout creation
    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    # Test inference mode (should pass through unchanged)
    x = Tensor([1, 2, 3, 4])
    y_inference = dropout.forward(x, training=False)
    assert np.array_equal(x.data, y_inference.data), "Inference should pass through unchanged"

    # Test training mode with zero dropout (should pass through unchanged)
    dropout_zero = Dropout(0.0)
    y_zero = dropout_zero.forward(x, training=True)
    assert np.array_equal(x.data, y_zero.data), "Zero dropout should pass through unchanged"

    # Test training mode with full dropout (should zero everything)
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x, training=True)
    assert np.allclose(y_full.data, 0), "Full dropout should zero everything"

    # Test training mode with partial dropout
    # Note: This is probabilistic, so we test statistical properties
    rng = np.random.default_rng(7)  # For reproducible test
    x_large = Tensor(np.ones((1000,)))  # Large tensor for statistical significance
    y_train = dropout.forward(x_large, training=True)

    # Count non-zero elements (approximately 50% should survive)
    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500
    # Use 3-sigma bounds: std = sqrt(n*p*(1-p)) = sqrt(1000*0.5*0.5) ≈ 15.8
    std_error = np.sqrt(1000 * 0.5 * 0.5)
    lower_bound = expected - 3 * std_error  # ≈ 453
    upper_bound = expected + 3 * std_error  # ≈ 547
    assert lower_bound < non_zero_count < upper_bound, \
        f"Expected {expected}±{3*std_error:.0f} survivors, got {non_zero_count}"

    # Test scaling (surviving elements should be scaled by 1/(1-p) = 2.0)
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0  # 1.0 / (1 - 0.5)
    assert np.allclose(surviving_values, expected_value), f"Surviving values should be {expected_value}"

    # Test no parameters
    params = dropout.parameters()
    assert len(params) == 0, "Dropout should have no parameters"

    # Test invalid probability
    try:
        Dropout(-0.1)
        assert False, "Should raise ValueError for negative probability"
    except ValueError:
        pass

    try:
        Dropout(1.1)
        assert False, "Should raise ValueError for probability > 1"
    except ValueError:
        pass

    print("Dropout layer works correctly!")



# 3. Sequential layer
