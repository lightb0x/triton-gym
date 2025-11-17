import torch
import triton
import triton.language as tl
from utils import is_power_of_two

_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn}, num_warps=nw)
    for (bm, bn, nw) in [
        (1, 1, 8),
    ]
]


@triton.jit
def fp32_maxexp(x):
    x_i32 = tl.cast(x, dtype=tl.int32, bitcast=True)
    x_exp = (x_i32 & 0x7F80_0000) >> 23  # extract exponent from 1-8-23
    return tl.max(x_exp) - 127  # fp32 bias 127


@triton.autotune(
    configs=_CONFIGS,
    key=["N"],
)
@triton.jit
def quantize_fp8_square_tile_kernel(
    # pointer i/o
    g_x,
    g_x_fp8,
    g_x_scale,
    # scalar param
    block_size: tl.constexpr,
    max_exp: tl.constexpr,
    # pointer shape
    M: tl.constexpr,
    N: tl.constexpr,
    # hyperparam
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    SM = tl.cdiv(M, block_size)
    SN = tl.cdiv(N, block_size)

    pid_m = pid // tl.cdiv(N, BN * block_size)
    pid_n = pid % tl.cdiv(N, BN * block_size)

    stride_m = N
    stride_n = 1

    stride_sm = SN
    stride_sn = 1

    for i_m in tl.static_range(0, BM):
        for i_n in tl.static_range(0, BN):
            offs_m = (
                pid_m * BM * block_size + i_m * block_size + tl.arange(0, block_size)
            )
            offs_n = (
                pid_n * BN * block_size + i_n * block_size + tl.arange(0, block_size)
            )

            g_x_ptrs = g_x + (offs_m[:, None] * stride_m + offs_n[None, :] * stride_n)
            mask_m = offs_m[:, None] < M
            mask_n = offs_n[None, :] < N
            x = tl.load(g_x_ptrs, mask=mask_m & mask_n)
            x_floorexp = fp32_maxexp(x)

            inv_scale = tl.cast(
                ((max_exp - x_floorexp + 127) & 0x0000_00FF) << 23,
                tl.float32,
                bitcast=True,
            )
            x_scale = tl.cast(
                ((x_floorexp - max_exp + 127) & 0x0000_00FF) << 23,
                tl.float32,
                bitcast=True,
            )
            x_fp8 = tl.cast(x * inv_scale, dtype=tl.float8e5)

            g_x_fp8_ptrs = g_x_fp8 + (
                offs_m[:, None] * stride_m + offs_n[None, :] * stride_n
            )
            tl.store(g_x_fp8_ptrs, x_fp8, mask=mask_m & mask_n)

            offs_sm = pid_m * BM + i_m + tl.arange(0, 1)
            offs_sn = pid_n * BN + i_n + tl.arange(0, 1)
            g_x_scale_ptrs = g_x_scale + (
                offs_sm[:, None] * stride_sm + offs_sn[None, :] * stride_sn
            )
            mask_sm = offs_sm[:, None] < SM
            mask_sn = offs_sn[None, :] < SN
            tl.store(g_x_scale_ptrs, x_scale, mask=mask_sm & mask_sn)

    return


def quantize_fp8_tensor_func(x, max_exp: int):
    absmax = x.abs().max()
    absmax_exp = torch.log2(absmax).floor()

    # compute tensor-wise absmax of `x`

    scale = 2 ** (absmax_exp - max_exp)
    x_fp8 = (x * 2 ** (max_exp - absmax_exp)).to(torch.float8_e5m2)

    return x_fp8, scale


def quantize_fp8_square_tile_func(x, block_size, max_exp: int):
    assert len(x.shape) >= 2, (
        f"input `x` must be at least 2-dim : got {len(x.shape)}-dim"
    )
    assert is_power_of_two(block_size), (
        f"`block_size` must be power-of-two : got {block_size}"
    )
    assert x.dtype == torch.float32, "`x` must be FP32"

    orig_shape = x.shape
    x = x.view(-1, x.shape[-1])

    M, N = x.shape
    x_fp8 = torch.empty_like(x, dtype=torch.float8_e5m2)
    x_scale = torch.empty(
        (triton.cdiv(M, block_size), triton.cdiv(N, block_size)),
        dtype=torch.float,
        device=x.device,
    )

    def grid(meta):
        """
        Each subprocess handles `(BM * block_size, BN * block_size)` elems
        """
        return (
            triton.cdiv(M, meta["BM"] * block_size)
            * triton.cdiv(N, meta["BN"] * block_size),
        )

    quantize_fp8_square_tile_kernel[grid](x, x_fp8, x_scale, block_size, max_exp, M, N)
    return x_fp8.view(orig_shape), x_scale


if __name__ == "__main__":
    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
    seed = 42
    shape = (8, 4)
    block_size = 2

    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(shape, device=DEVICE)

    print("tensor-wise quantization")
    x_fp8, x_scale = quantize_fp8_tensor_func(x, max_exp=9)

    assert x_fp8.dtype == torch.float8_e5m2

    x_recon = x_fp8.to(torch.float) * x_scale
    if not torch.allclose(x, x_recon):
        print(f"orig: {x}")
        print(f"recon: {x_recon}")

        print(f"fp8: {x_fp8}")
        print(f"scale: {x_scale}")

    print("square-tile-wise quantization")

    x_fp8, x_scale = quantize_fp8_square_tile_func(x, block_size=block_size, max_exp=9)
    assert x_fp8.dtype == torch.float8_e5m2

    x_scale_br = x_scale.reshape((*x_scale.shape, 1, 1)).broadcast_to(
        (
            *x_scale.shape,
            block_size,
            block_size,
        )
    )
    x_scale_brt = x_scale_br.transpose(1, 2).reshape(shape)

    x_recon = x_fp8.to(torch.float) * x_scale_brt
    if not torch.allclose(x, x_recon):
        print(f"orig: {x}")
        print(f"recon: {x_recon}")

        print(f"fp8: {x_fp8}")
        print(f"scale: {x_scale}")
