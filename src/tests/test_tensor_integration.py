import numpy as np

from tensor import Tensor


def test_center_and_scale():
    tensor = Tensor([1, 2, 3])

    result = (tensor - 2) / 2

    np.testing.assert_allclose(
        result.data,
        [-0.5, 0.0, 0.5],
        rtol=1e-6,
        atol=1e-7,
    )


def test_two_stage_linear_transformation():
    x = Tensor([[1, 2, 3], [4, 5, 6]])
    w1 = Tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ])
    b1 = Tensor([0.1, 0.2, 0.3, 0.4])
    w2 = Tensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
    ])
    b2 = Tensor([0.1, 0.2])

    hidden = x.matmul(w1) + b1
    output = hidden.matmul(w2) + b2

    assert hidden.shape == (2, 4)
    np.testing.assert_allclose(
        hidden.data,
        [[3.9, 4.6, 5.3, 6.0], [8.4, 10.0, 11.6, 13.2]],
        rtol=1e-6,
        atol=1e-7,
    )
    assert output.shape == (2, 2)
    np.testing.assert_allclose(
        output.data,
        [[8.72, 10.8], [18.98, 23.4]],
        rtol=1e-6,
        atol=1e-7,
    )


def test_reshape_then_spatial_mean():
    tensor = Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    result = tensor.reshape(2, 2, 3).mean(axis=(1, 2))

    assert result.shape == (2,)
    np.testing.assert_allclose(result.data, [3.5, 9.5], rtol=1e-6, atol=1e-7)
