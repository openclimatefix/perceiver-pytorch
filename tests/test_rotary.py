import torch

from perceiver_pytorch.rotary import SinusoidalEmbeddings, apply_rotary_emb, rotate_every_two


def test_rotate_every_two():
    """
    Test for rotate every two
    :return:
    """

    x = torch.randn(5, 4, 4)
    y = rotate_every_two(x)

    assert y.shape == torch.Size([5, 4, 4])
    assert y[0, 0, 0] == -x[0, 0, 1]
    assert y[0, 0, 1] == x[0, 0, 0]


def test_apply_rotary_emb():
    """
    Check that 'apply_rotary_emb' works correctly
    :return:
    """

    sinu_pos = torch.randn(1, 4, 10)
    q = torch.randn(5, 4, 10)
    k = torch.randn(5, 4, 10)

    q, k = apply_rotary_emb(q, k, sinu_pos=sinu_pos)


def test_torch_sinusoidal_mbeddings():
    model = SinusoidalEmbeddings(dim=128)

    y = model(torch.randn(4, 10))
    assert y.shape[-1] == 128
    assert y.shape[-2] == 4
