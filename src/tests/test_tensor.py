import operator

import numpy as np
import pytest

from tensor import Tensor


@pytest.mark.parametrize(
    "data, expected_shape, expected_size",
    [
        pytest.param(5.0, (), 1, id="scalar"),
        pytest.param([1, 2, 3], (3,), 3, id="vector"),
        pytest.param([[1, 2], [3, 4]], (2, 2), 4, id="matrix"),
        pytest.param(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            (2, 2, 2),
            8,
            id="three-dimensional",
        ),
    ],
)
def test_tensor_creation(data, expected_shape, expected_size):
    tensor = Tensor(data)

    np.testing.assert_array_equal(
        tensor.data,
        np.array(data, dtype=np.float32),
    )
    assert tensor.dtype == np.dtype(np.float32)
    assert tensor.shape == expected_shape
    assert tensor.size == expected_size
    assert tensor.ndim == len(expected_shape)
    assert tensor.numel() == expected_size


def test_contiguous_returns_independent_contiguous_copy():
    tensor = Tensor([[1, 2], [3, 4]])

    result = tensor.contiguous()

    np.testing.assert_array_equal(result.data, tensor.data)
    assert result.data.flags.c_contiguous
    assert not np.shares_memory(result.data, tensor.data)


@pytest.mark.parametrize(
    "operation, left, right, expected",
    [
        pytest.param(
            operator.add, [1, 2, 3], [4, 5, 6], [5, 7, 9], id="add"
        ),
        pytest.param(
            operator.sub, [4, 5, 6], [1, 2, 3], [3, 3, 3], id="subtract"
        ),
        pytest.param(
            operator.mul, [1, 2, 3], [4, 5, 6], [4, 10, 18], id="multiply"
        ),
        pytest.param(
            operator.truediv, [4, 5, 6], [1, 2, 3], [4, 2.5, 2], id="divide"
        ),
    ],
)
def test_tensor_arithmetic(operation, left, right, expected):
    a = Tensor(left)
    b = Tensor(right)

    result = operation(a, b)

    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "operation, scalar, scalar_on_left, expected",
    [
        pytest.param(operator.add, 10, False, [14, 15, 16], id="tensor-plus-scalar"),
        pytest.param(operator.add, 10, True, [14, 15, 16], id="scalar-plus-tensor"),
        pytest.param(operator.sub, 10, False, [-6, -5, -4], id="tensor-minus-scalar"),
        pytest.param(operator.sub, 10, True, [6, 5, 4], id="scalar-minus-tensor"),
        pytest.param(operator.mul, 2, False, [8, 10, 12], id="tensor-times-scalar"),
        pytest.param(operator.mul, 2, True, [8, 10, 12], id="scalar-times-tensor"),
        pytest.param(operator.truediv, 2, False, [2, 2.5, 3], id="tensor-over-scalar"),
        pytest.param(operator.truediv, 12, True, [3, 2.4, 2], id="scalar-over-tensor"),
    ],
)
def test_scalar_arithmetic(operation, scalar, scalar_on_left, expected):
    tensor = Tensor([4, 5, 6])

    if scalar_on_left:
        result = operation(scalar, tensor)
    else:
        result = operation(tensor, scalar)

    assert isinstance(result, Tensor)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-7)


def test_add_broadcasts_vector_across_matrix_rows():
    matrix = Tensor([[1, 2], [3, 4]])
    vector = Tensor([10, 20])

    result = matrix + vector

    np.testing.assert_array_equal(result.data, [[11, 22], [13, 24]])


def test_add_broadcasts_scalar_tensor():
    scalar = Tensor(5.0)
    vector = Tensor([1, 2, 3])

    result = scalar + vector

    np.testing.assert_array_equal(result.data, [6, 7, 8])


@pytest.mark.parametrize(
    "left, right, expected",
    [
        pytest.param(
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[19, 22], [43, 50]],
            id="square",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8], [9, 10], [11, 12]],
            [[58, 64], [139, 154]],
            id="rectangular",
        ),
        pytest.param(
            [[1, 2, 3]],
            [[1], [2], [3]],
            [[14]],
            id="single-row-and-column",
        ),
        pytest.param(
            [[1, 2, 3], [4, 5, 6]],
            [1, 2, 3],
            [14, 32],
            id="matrix-vector",
        ),
    ],
)
def test_matmul(left, right, expected):
    a = Tensor(left)
    b = Tensor(right)

    result = a.matmul(b)

    assert isinstance(result, Tensor)
    np.testing.assert_array_equal(result.data, expected)


def test_matmul_operator():
    a = Tensor([[1, 2], [3, 4]])
    b = Tensor([[5, 6], [7, 8]])

    result = a @ b

    np.testing.assert_array_equal(result.data, [[19, 22], [43, 50]])


def test_matmul_rejects_non_tensor_operand():
    tensor = Tensor([[1, 2], [3, 4]])

    with pytest.raises(TypeError, match="requires Tensor, got list"):
        tensor.matmul([[5, 6], [7, 8]])


@pytest.mark.parametrize(
    "left, right",
    [
        pytest.param(5.0, [[1, 2], [3, 4]], id="scalar-left"),
        pytest.param([[1, 2], [3, 4]], 5.0, id="scalar-right"),
    ],
)
def test_matmul_rejects_scalar_operands(left, right):
    a = Tensor(left)
    b = Tensor(right)

    with pytest.raises(ValueError, match="at least 1D"):
        a.matmul(b)


def test_matmul_rejects_incompatible_inner_dimensions():
    a = Tensor([[1, 2]])
    b = Tensor([[1], [2], [3]])

    with pytest.raises(ValueError, match="Inner dimensions don't match: 2 vs 3"):
        a.matmul(b)


@pytest.mark.parametrize(
    "shape_args, expected",
    [
        pytest.param(
            (2, 3),
            [[1, 2, 3], [4, 5, 6]],
            id="separate-dimensions",
        ),
        pytest.param(
            ((3, 2),),
            [[1, 2], [3, 4], [5, 6]],
            id="tuple",
        ),
        pytest.param(
            (2, -1),
            [[1, 2, 3], [4, 5, 6]],
            id="inferred-dimension",
        ),
    ],
)
def test_reshape(shape_args, expected):
    tensor = Tensor([1, 2, 3, 4, 5, 6])

    result = tensor.reshape(*shape_args)

    assert result.shape == np.shape(expected)
    np.testing.assert_array_equal(result.data, expected)


def test_reshape_rejects_incompatible_element_count():
    tensor = Tensor([1, 2, 3, 4, 5, 6])

    with pytest.raises(
        ValueError,
        match="Element count mismatch: 6 elements vs 4 elements",
    ):
        tensor.reshape(2, 2)


def test_reshape_flattens_each_batch_item():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4))

    result = tensor.reshape(2, -1)

    assert result.shape == (2, 12)
    np.testing.assert_array_equal(result.data, np.arange(24).reshape(2, 12))


def test_transpose_matrix():
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = tensor.transpose()

    assert result.shape == (3, 2)
    np.testing.assert_array_equal(result.data, [[1, 4], [2, 5], [3, 6]])


def test_transpose_vector_preserves_values_and_shape():
    tensor = Tensor([1, 2, 3])

    result = tensor.transpose()

    assert result.shape == (3,)
    np.testing.assert_array_equal(result.data, [1, 2, 3])


def test_transpose_swaps_requested_axes():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4))

    result = tensor.transpose(0, 2)

    assert result.shape == (4, 3, 2)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert result.data[k, j, i] == tensor.data[i, j, k]


def test_transpose_defaults_to_last_two_axes():
    tensor = Tensor(np.arange(24).reshape(2, 3, 4))

    result = tensor.transpose()

    assert result.shape == (2, 4, 3)
    for i in range(2):
        for j in range(3):
            for k in range(4):
                assert result.data[i, k, j] == tensor.data[i, j, k]


@pytest.mark.parametrize(
    "method, axis, keepdims, expected",
    [
        pytest.param("sum", None, False, 21.0, id="sum-all"),
        pytest.param("sum", 0, False, [5, 7, 9], id="sum-columns"),
        pytest.param("sum", 1, False, [6, 15], id="sum-rows"),
        pytest.param("sum", 1, True, [[6], [15]], id="sum-keepdims"),
        pytest.param("mean", None, False, 3.5, id="mean-all"),
        pytest.param("mean", 0, False, [2.5, 3.5, 4.5], id="mean-columns"),
        pytest.param("max", None, False, 6.0, id="max-all"),
        pytest.param("max", 1, False, [3, 6], id="max-rows"),
    ],
)
def test_reduction(method, axis, keepdims, expected):
    tensor = Tensor([[1, 2, 3], [4, 5, 6]])

    result = getattr(tensor, method)(axis=axis, keepdims=keepdims)

    assert isinstance(result, Tensor)
    assert result.shape == np.shape(expected)
    np.testing.assert_allclose(result.data, expected, rtol=1e-6, atol=1e-7)


def test_mean_reduces_multiple_axes():
    tensor = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    result = tensor.mean(axis=(1, 2))

    assert result.shape == (2,)
    np.testing.assert_allclose(result.data, [2.5, 6.5], rtol=1e-6, atol=1e-7)
