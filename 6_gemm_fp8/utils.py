import torch
import triton
import triton.language as tl
from config import BLOCK_SIZE


def reconstruct(x_fp8, x_scale):
    assert len(x_fp8.shape) == 2
    assert len(x_scale.shape) == 2

    M, N = x_fp8.shape
    Ms, Ns = x_scale.shape
    if M == Ms:
        assert triton.cdiv(N, BLOCK_SIZE) == Ns
        x_scale_bc = (
            x_scale.view(M, Ns, 1)
            .broadcast_to(M, Ns, BLOCK_SIZE)
            .reshape(M, Ns * BLOCK_SIZE)[:, :N]
        )
    elif N == Ns:
        assert triton.cdiv(M, BLOCK_SIZE) == Ms
        x_scale_bc = (
            x_scale.view(Ms, 1, N)
            .broadcast_to(Ms, BLOCK_SIZE, N)
            .reshape(Ms * BLOCK_SIZE, N)[:M, :]
        )
    else:
        assert triton.cdiv(N, Ns) == triton.cdiv(M, Ms)
        x_scale_bc = (
            x_scale.view(Ms, 1, Ns, 1)
            .broadcast_to(Ms, BLOCK_SIZE, Ns, BLOCK_SIZE)
            .transpose(1, 2)
            .reshape(Ms * BLOCK_SIZE, Ns * BLOCK_SIZE)[:M, :N]
        )
    return x_fp8.to(torch.float) * x_scale_bc


def generate_range(start, num_steps, until):
    """
    generates floating-point-like range
    """
    current = start
    step = start
    arr = []
    while current < until:
        arr.append(current)
        current += step
        if current == step * num_steps:
            step *= 2

    return arr


def msb_index(v: int):
    # De Brujin sequence
    assert v > 0, f"{v} should be positive"
    assert (v & (v - 1)) == 0, f"{v} should be power of two"
    bit_position_LUT = [
        0,
        1,
        16,
        2,
        29,
        17,
        3,
        22,
        30,
        20,
        18,
        11,
        13,
        4,
        7,
        23,
        31,
        15,
        28,
        21,
        19,
        10,
        12,
        6,
        14,
        27,
        9,
        5,
        26,
        8,
        25,
        24,
    ]
    return bit_position_LUT[((v * 0x06EB_14F9) & 0xFFFF_FFFF) >> 27]


def is_power_of_two(x: int):
    return (x & (x - 1)) == 0


def broadcast(x, block_size):
    assert len(x.shape) == 2
    M, N = x.shape
    x = x.reshape([M, N, 1, 1])
    x = x.broadcast_to([M, N, block_size, block_size])
    x = x.transpose(1, 2)
    x = x.reshape(M * block_size, N * block_size)
    return x


@triton.jit
def fp32_maxexp(x, axis=None, keep_dims=False):
    x_i32 = tl.cast(x, dtype=tl.int32, bitcast=True)
    x_exp = (x_i32 & 0x7F80_0000) >> 23  # extract exponent from 1-8-23
    return tl.max(x_exp, axis=axis, keep_dims=keep_dims) - 127  # fp32 bias 127


@triton.jit
def fp32_absmax(x, axis=None, keep_dims=False):
    x_i32 = tl.cast(x, dtype=tl.int32, bitcast=True)
    x_exp_man = x_i32 & 0x7FFF_FFFF
    return tl.cast(
        tl.max(x_exp_man, axis=axis, keep_dims=keep_dims),
        dtype=tl.float32,
        bitcast=True,
    )
