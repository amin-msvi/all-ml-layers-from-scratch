def test_unit_extract_gradient():
    """Test _extract_gradient handles Tensor and ndarray gradients."""
    print("Unit Test: Gradient Extraction...")

    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = Optimizer([param])

    # Case 1: Gradient is a Tensor
    param.grad = Tensor([0.1, 0.2])
    grad_data = optimizer._extract_gradient(param)
    assert isinstance(grad_data, np.ndarray), "Should return ndarray from Tensor grad"
    assert np.allclose(grad_data, [0.1, 0.2])

    # Case 2: Gradient is a raw NumPy array
    param.grad = np.array([0.3, 0.4])
    grad_data = optimizer._extract_gradient(param)
    assert isinstance(grad_data, np.ndarray), "Should return ndarray from ndarray grad"
    assert np.allclose(grad_data, [0.3, 0.4])

    # Case 3: Multi-dimensional gradient
    param_2d = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    opt_2d = Optimizer([param_2d])
    param_2d.grad = Tensor([[0.1, 0.2], [0.3, 0.4]])
    grad_data_2d = opt_2d._extract_gradient(param_2d)
    assert grad_data_2d.shape == (2, 2)
    assert np.allclose(grad_data_2d, [[0.1, 0.2], [0.3, 0.4]])

    print("Gradient extraction works correctly!")


def test_unit_optimizer_base():
    """Test base Optimizer functionality."""
    print("Unit Test: Base Optimizer...")

    # Create test parameters
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([[3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    # Create optimizer first (optimizer.__init__ resets grad to None)
    optimizer = Optimizer([param1, param2])

    # Test parameter storage
    assert len(optimizer.params) == 2
    assert optimizer.params[0] is param1
    assert optimizer.params[1] is param2
    assert optimizer.step_count == 0

    # Add gradients AFTER creating optimizer to test zero_grad properly
    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([[0.3, 0.4], [0.5, 0.6]])

    # Test zero_grad
    optimizer.zero_grad()
    assert param1.grad is None
    assert param2.grad is None

    # Test that optimizer accepts any tensor (no validation required)
    # Gradient tracking is handled by the autograd module
    regular_param = Tensor([1.0])
    opt = Optimizer([regular_param])
    assert len(opt.params) == 1

    print("Base Optimizer works correctly!")


def test_unit_sgd_optimizer():
    """Test SGD optimizer implementation."""
    print("Unit Test: SGD Optimizer...")

    # Test basic SGD without momentum
    param = Tensor([1.0, 2.0], requires_grad=True)
    optimizer = SGD([param], lr=0.1)
    # Set gradient AFTER creating optimizer (optimizer.__init__ resets grad to None)
    param.grad = Tensor([0.1, 0.2])
    original_data = param.data.copy()

    optimizer.step()

    # Expected: param = param - lr * grad = [1.0, 2.0] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    expected = original_data - 0.1 * np.array([0.1, 0.2])
    assert np.allclose(param.data, expected)
    assert optimizer.step_count == 1

    # Test SGD with momentum
    param2 = Tensor([1.0, 2.0], requires_grad=True)
    optimizer_momentum = SGD([param2], lr=0.1, momentum=0.9)
    # Set gradient AFTER creating optimizer
    param2.grad = Tensor([0.1, 0.2])

    # First step: v = 0.9 * 0 + [0.1, 0.2] = [0.1, 0.2]
    optimizer_momentum.step()
    expected_first = np.array([1.0, 2.0]) - 0.1 * np.array([0.1, 0.2])
    assert np.allclose(param2.data, expected_first)

    # Second step with same gradient
    param2.grad = Tensor([0.1, 0.2])
    optimizer_momentum.step()
    # v = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
    expected_momentum = np.array([0.19, 0.38])
    expected_second = expected_first - 0.1 * expected_momentum
    assert np.allclose(param2.data, expected_second, rtol=1e-5)

    # Test weight decay
    param3 = Tensor([1.0, 2.0], requires_grad=True)
    optimizer_wd = SGD([param3], lr=0.1, weight_decay=0.01)
    # Set gradient AFTER creating optimizer
    param3.grad = Tensor([0.1, 0.2])
    optimizer_wd.step()

    # grad_with_decay = [0.1, 0.2] + 0.01 * [1.0, 2.0] = [0.11, 0.22]
    expected_wd = np.array([1.0, 2.0]) - 0.1 * np.array([0.11, 0.22])
    assert np.allclose(param3.data, expected_wd)

    print("SGD optimizer works correctly!")