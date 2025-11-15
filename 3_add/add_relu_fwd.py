import torch
import triton
import triton.language as tl

_CONFIGS = [
    triton.Config({"BN": bn}, num_warps=nw) for (bn, nw) in [
        (4096, 8),
        (8192, 8),
        (4096, 4),
        (8192, 4),
    ]
]

# triton subroutine
@triton.jit
def binary_f32_mask(x, len_x: tl.constexpr):
    """
    Args:
        x ([f32], len=BN)
        len_x (integer, value=BN)

    Returns:
        [u8], len=BN
            packed mask for ReLU backward
            - 8b'00  if x < 0
            - 8b'01  if x == 0
            - 8b'02  if x > 0
    """
    # mask = tl.zeros(len_x, dtype=tl.uint8)

    # bin_x = tl.cast(x, tl.uint32, bitcast=True)
    # x_nosign = bin_x & 0x7fff_ffff
    # x_nosign = x_nosign | (x_nosign >> 1)
    # x_nosign = x_nosign | (x_nosign >> 2)
    # x_nosign = x_nosign | (x_nosign >> 4)
    # x_nosign = x_nosign | (x_nosign >> 8)
    # x_nosign = (x_nosign | (x_nosign >> 16)) & 0x0000_0001
    # # `x_nosign` is
    # # 0 if x is +-0
    # # 1 o.w.

    # x_sign = (bin_x & 0x8000_0000) >> 30
    # # `x_sign` is
    # # 0 if x is positive
    # # 2 o.w.

    # # x0 --> 01
    # # 01 --> 10
    # # 11 --> 00

    mask = tl.cast(x >= 0, tl.uint8) + tl.cast(x > 0, tl.uint8)

    return mask


@triton.autotune(
    configs=_CONFIGS,
    key=["N"],
)
@triton.jit
def add_relu_kernel(
    # pointer i/o
    g_lhs,
    g_rhs,
    g_out,
    g_mask,
    # pointer shape
    N,
    # hyperparams
    BN: tl.constexpr
):
    pid = tl.program_id(axis=0)

    offs = pid * BN + tl.arange(0, BN)
    mask = offs < N

    g_lhs_ptrs = g_lhs + offs
    g_rhs_ptrs = g_rhs + offs

    lhs = tl.load(g_lhs_ptrs, mask=mask)
    rhs = tl.load(g_rhs_ptrs, mask=mask)

    added = tl.add(lhs, rhs)

    # added = added.reshape(BN // 4, 4).reduce(axis=-1, combine_fn=binary_f32_mask)

    bin_mask = binary_f32_mask(added, BN)

    result = tl.maximum(added, 0)

    g_out_ptrs = g_out + offs
    g_mask_ptrs = g_mask + offs

    tl.store(g_out_ptrs, result, mask=mask)
    tl.store(g_mask_ptrs, bin_mask, mask=mask)

    return


def add_relu_fwd_func(lhs, rhs):
    """
    computes `relu(lhs + rhs)` elem-wise

    Args:
        lhs (ND Tensor)
        rhs (ND Tensor)

    Returns:
        relu(lhs + rhs)  (ND Tensor)
        (lhs + rhs > 0)  (ND Tensor)
    """
    # result = lhs + rhs

    assert lhs.numel() == rhs.numel()
    # assert lhs.numel() % 4 == 0  # for mask packing to `uint8`
    assert len(lhs.shape) == len(rhs.shape)
    assert all([lhs.shape[i] == rhs.shape[i] for i in range(len(lhs.shape))])
    assert lhs.dtype == rhs.dtype == torch.float32

    orig_shape = lhs.shape

    # Flatten to 1D
    lhs = lhs.view(-1)
    rhs = rhs.view(-1)

    N = lhs.numel()

    result = torch.empty_like(lhs)
    # lhs + rhs  < 0  --> 2b'00
    # lhs + rhs == 0  --> 2b'01
    # lhs + rhs  > 0  --> 2b'10
    mask = torch.empty(N, dtype=torch.uint8, device=result.device)

    # subprocess grid
    # every process handles "BN" elements out of total "N" elems
    grid = lambda meta: (
        triton.cdiv(N, meta["BN"]),
    )

    add_relu_kernel[grid](lhs, rhs, result, mask, N)

    return result.view(orig_shape), mask

if __name__=="__main__":
    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    gen = torch.Generator(DEVICE).manual_seed(42)
    t1 = torch.randn(4, 4, generator=gen, device=DEVICE)
    t2 = torch.randn(4, 4, generator=gen, device=DEVICE)
    t1[0, 3] = 0
    t2[0, 3] = 0
    result, mask = add_relu_fwd_func(t1, t2)
    print(torch.isclose(torch.nn.functional.relu(t1 + t2), result))
    print(mask)
