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

@triton.autotune(
    configs=_CONFIGS,
    key=["N"],
)
@triton.jit
def add_kernel(
    # pointer i/o
    g_lhs,
    g_rhs,
    g_out,
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

    result = tl.add(lhs, rhs)

    g_out_ptrs = g_out + offs

    tl.store(g_out_ptrs, result, mask=mask)

    return

def add_fwd_func(lhs, rhs):
    """
    computes `lhs + rhs` elem-wise

    Args:
        lhs (ND Tensor)
        rhs (ND Tensor)

    Returns:
        lhs + rhs (ND Tensor)
    """

    # result = lhs + rhs

    assert lhs.numel() == rhs.numel()
    assert len(lhs.shape) == len(rhs.shape)
    assert all([lhs.shape[i] == rhs.shape[i] for i in range(len(lhs.shape))])

    orig_shape = lhs.shape

    # Flatten to 1D
    lhs = lhs.view(-1)
    rhs = rhs.view(-1)

    N = lhs.numel()

    result = torch.empty_like(lhs)

    # subprocess grid
    # every process handles "BN" elements out of total "N" elems
    grid = lambda meta: (
        triton.cdiv(N, meta["BN"]),
    )

    add_kernel[grid](lhs, rhs, result, N)

    return result.view(orig_shape)

if __name__=="__main__":
    # test drive
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

    gen = torch.Generator(DEVICE).manual_seed(42)
    t1 = torch.randn(4, 4, generator=gen, device=DEVICE)
    t2 = torch.randn(4, 4, generator=gen, device=DEVICE)
    print(torch.isclose(t1 + t2, add_fwd_func(t1, t2)))
